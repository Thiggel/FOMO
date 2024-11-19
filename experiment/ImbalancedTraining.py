import os
import json
from typing import Optional, Dict, Any
import torch
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint


class CheckpointState:
    def __init__(self):
        self.current_cycle = 0
        self.initial_train_ds_size = None
        self.additional_data_state = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_cycle": self.current_cycle,
            "initial_train_ds_size": self.initial_train_ds_size,
            "additional_data_state": self.additional_data_state,
        }

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "CheckpointState":
        state = cls()
        state.current_cycle = state_dict.get("current_cycle", 0)
        state.initial_train_ds_size = state_dict.get("initial_train_ds_size", None)
        state.additional_data_state = state_dict.get("additional_data_state", {})
        return state


class ImbalancedTraining:
    def __init__(
        self,
        args: dict,
        trainer_args: dict,
        ssl_method: L.LightningModule,
        datamodule: L.LightningDataModule,
        checkpoint_callback: L.Callback,
        checkpoint_filename: str,
        save_class_distribution: bool = False,
        run_idx: int = 0,
    ):
        self.args = args
        self.run_idx = run_idx
        self.trainer_args = trainer_args
        self.ssl_method = ssl_method
        self.datamodule = datamodule
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_filename = checkpoint_filename
        self.save_class_distribution = save_class_distribution
        self.n_epochs_per_cycle = args.n_epochs_per_cycle
        self.max_cycles = args.max_cycles
        self.ood_test_split = args.ood_test_split

        # Initialize checkpoint state
        self.state = CheckpointState()
        if self.datamodule is not None:
            self.state.initial_train_ds_size = len(self.datamodule.train_dataset)

        # Create checkpoint directory
        checkpoint_dir = os.environ.get("BASE_CACHE_DIR", "checkpoints")
        self.checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_filename)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # State file path
        self.state_file = os.path.join(self.checkpoint_dir, "training_state.json")

    def save_state(self) -> None:
        """Save current training state"""
        state_dict = self.state.to_dict()
        with open(self.state_file, "w") as f:
            json.dump(state_dict, f)

    def load_state(self) -> None:
        """Load training state if it exists"""
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                state_dict = json.load(f)
            self.state = CheckpointState.from_dict(state_dict)

    def get_checkpoint_path(self, cycle: int, epoch: Optional[int] = None) -> str:
        """Get path for checkpoint file"""
        if epoch is not None:
            return os.path.join(
                self.checkpoint_dir, f"cycle_{cycle}_epoch_{epoch}.ckpt"
            )
        return os.path.join(self.checkpoint_dir, f"cycle_{cycle}_final.ckpt")

    def save_checkpoint(self, cycle: int, epoch: Optional[int] = None) -> None:
        """Save a complete checkpoint including model, optimizer, and training state"""
        checkpoint = {
            "model_state_dict": self.ssl_method.state_dict(),
            "optimizer_state_dict": self.ssl_method.optimizers().state_dict(),
            "cycle": cycle,
            "epoch": epoch,
            "state": self.state.to_dict(),
        }

        if self.datamodule is not None:
            checkpoint["datamodule_state"] = self.datamodule.state_dict()

        torch.save(checkpoint, self.get_checkpoint_path(cycle, epoch))
        self.save_state()

    def load_checkpoint(self, cycle: int, epoch: Optional[int] = None) -> None:
        """Load a complete checkpoint"""
        checkpoint_path = self.get_checkpoint_path(cycle, epoch)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)

        # Load model and optimizer states
        self.ssl_method.load_state_dict(checkpoint["model_state_dict"])
        self.ssl_method.optimizers().load_state_dict(checkpoint["optimizer_state_dict"])

        # Load training state
        self.state = CheckpointState.from_dict(checkpoint["state"])

        # Load datamodule state if available
        if "datamodule_state" in checkpoint and self.datamodule is not None:
            self.datamodule.load_state_dict(checkpoint["datamodule_state"])

    def pretrain_cycle(self, cycle_idx: int) -> None:
        """Modified pretrain_cycle with checkpointing"""
        trainer = L.Trainer(
            **self.trainer_args,
            callbacks=[
                *self.trainer_args.get("callbacks", []),
                ModelCheckpoint(
                    dirpath=self.checkpoint_dir,
                    filename=f"cycle_{cycle_idx}_epoch_{{epoch}}",
                    save_top_k=-1,  # Save all epochs
                    every_n_epochs=1,
                ),
            ],
        )

        # Find latest epoch checkpoint if exists
        latest_epoch = self._find_latest_epoch(cycle_idx)
        ckpt_path = (
            None
            if latest_epoch is None
            else self.get_checkpoint_path(cycle_idx, latest_epoch)
        )

        # Train
        trainer.fit(
            model=self.ssl_method, datamodule=self.datamodule, ckpt_path=ckpt_path
        )

        # Save cycle checkpoint
        self.save_checkpoint(cycle_idx)

        # Clean up intermediate checkpoints
        self._cleanup_epoch_checkpoints(cycle_idx)

        self.datamodule.set_dataloaders_none()

        if not self.args.ood_augmentation:
            return

        # Handle OOD data generation and augmentation
        if self.args.remove_diffusion:
            self._handle_remove_diffusion()
        else:
            self._handle_ood_generation(cycle_idx)

    def _find_latest_epoch(self, cycle_idx: int) -> Optional[int]:
        """Find the latest epoch checkpoint for a given cycle"""
        latest_epoch = None
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(f"cycle_{cycle_idx}_epoch_"):
                epoch = int(filename.split("_")[-1].split(".")[0])
                if latest_epoch is None or epoch > latest_epoch:
                    latest_epoch = epoch
        return latest_epoch

    def _cleanup_epoch_checkpoints(self, cycle_idx: int) -> None:
        """Clean up intermediate epoch checkpoints after cycle completion"""
        for filename in os.listdir(self.checkpoint_dir):
            if filename.startswith(f"cycle_{cycle_idx}_epoch_"):
                os.remove(os.path.join(self.checkpoint_dir, filename))

    def pretrain_imbalanced(self) -> None:
        """Modified pretrain_imbalanced with checkpoint handling"""
        # Load existing state if available
        self.load_state()

        visualization_dir = (
            f"visualizations/class_distributions/{self.checkpoint_filename}"
        )
        os.makedirs(visualization_dir, exist_ok=True)

        if self.save_class_distribution and self.state.current_cycle == 0:
            self.save_class_dist(
                self.datamodule.train_dataset, f"{visualization_dir}/initial_class_dist"
            )

        for cycle_idx in range(self.state.current_cycle, self.max_cycles):
            print(f"Run {self.run_idx + 1}/{self.args.num_runs}")
            print(f"Pretraining cycle {cycle_idx + 1}/{self.max_cycles}")

            self.pretrain_cycle(cycle_idx)
            self.state.current_cycle = cycle_idx + 1
            self.save_state()

            if self.save_class_distribution:
                self.save_class_dist(
                    self.datamodule.train_dataset,
                    f"{visualization_dir}/class_dist_after_cycle_{cycle_idx}",
                )
