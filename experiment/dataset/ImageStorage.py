import h5py
import numpy as np
from PIL import Image
import os
import io
from typing import Optional, Tuple
from filelock import FileLock
import threading
from concurrent.futures import ThreadPoolExecutor


class ImageStorage:
    def __init__(self, base_path: str, max_images_per_file: int = 1000):
        """
        Initialize the image storage system using HDF5 files.

        Args:
            base_path: Base directory for storing HDF5 files
            max_images_per_file: Maximum number of images per HDF5 file
        """
        self.base_path = base_path
        self.max_images_per_file = max_images_per_file
        self._locks = {}
        self._locks_lock = threading.Lock()

    def _get_cycle_path(self, cycle_idx: int) -> str:
        """Get the directory path for a specific training cycle."""
        return os.path.join(self.base_path, str(cycle_idx))

    def _get_hdf5_path(self, cycle_idx: int, file_idx: int) -> str:
        """Get the path for a specific HDF5 file."""
        cycle_path = self._get_cycle_path(cycle_idx)
        return os.path.join(cycle_path, f"images_{file_idx}.h5")

    def _get_lock(self, filepath: str) -> FileLock:
        """Get or create a lock for a specific file."""
        with self._locks_lock:
            if filepath not in self._locks:
                self._locks[filepath] = FileLock(filepath + ".lock")
            return self._locks[filepath]

    def _get_file_and_index(self, global_idx: int) -> Tuple[int, int]:
        """Convert a global image index to file index and local index."""
        file_idx = global_idx // self.max_images_per_file
        local_idx = global_idx % self.max_images_per_file
        return file_idx, local_idx

    def save_image(self, image: Image.Image, cycle_idx: int, global_idx: int) -> None:
        """
        Save a PIL Image to the appropriate HDF5 file.

        Args:
            image: PIL Image to save
            cycle_idx: Current training cycle index
            global_idx: Global index of the image
        """
        file_idx, local_idx = self._get_file_and_index(global_idx)

        # Ensure cycle directory exists
        cycle_path = self._get_cycle_path(cycle_idx)
        os.makedirs(cycle_path, exist_ok=True)

        h5_path = self._get_hdf5_path(cycle_idx, file_idx)
        lock = self._get_lock(h5_path)

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()

        with lock:
            with h5py.File(h5_path, "a") as f:
                if "images" not in f:
                    # Create dataset with maxshape for extensibility
                    f.create_dataset(
                        "images",
                        shape=(0,),
                        maxshape=(self.max_images_per_file,),
                        dtype=h5py.special_dtype(vlen=np.dtype("uint8")),
                    )
                    f.create_dataset(
                        "indices",
                        shape=(0,),
                        maxshape=(self.max_images_per_file,),
                        dtype="int64",
                    )

                images = f["images"]
                indices = f["indices"]

                # Resize if needed
                if local_idx >= len(images):
                    new_size = local_idx + 1
                    images.resize((new_size,))
                    indices.resize((new_size,))

                # Store image bytes and index
                images[local_idx] = np.frombuffer(img_bytes, dtype="uint8")
                indices[local_idx] = global_idx

    def load_image(self, cycle_idx: int, global_idx: int) -> Optional[Image.Image]:
        """
        Load an image from HDF5 storage.

        Args:
            cycle_idx: Training cycle index
            global_idx: Global index of the image

        Returns:
            PIL Image if found, None otherwise
        """
        file_idx, local_idx = self._get_file_and_index(global_idx)
        h5_path = self._get_hdf5_path(cycle_idx, file_idx)

        if not os.path.exists(h5_path):
            return None

        # No lock needed for reading
        with h5py.File(h5_path, "r") as f:
            if "images" not in f or local_idx >= len(f["images"]):
                return None

            img_bytes = f["images"][local_idx]
            return Image.open(io.BytesIO(img_bytes.tobytes()))

    def save_batch(
        self, images: list[Image.Image], cycle_idx: int, start_idx: int
    ) -> None:
        """
        Save a batch of images efficiently.

        Args:
            images: List of PIL Images to save
            cycle_idx: Current training cycle index
            start_idx: Starting global index for the batch
        """
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, image in enumerate(images):
                futures.append(
                    executor.submit(self.save_image, image, cycle_idx, start_idx + i)
                )
            # Wait for all saves to complete
            for future in futures:
                future.result()

    def clear_cycle(self, cycle_idx: int) -> None:
        """Remove all HDF5 files for a specific cycle."""
        cycle_path = self._get_cycle_path(cycle_idx)
        if os.path.exists(cycle_path):
            for filename in os.listdir(cycle_path):
                if filename.endswith(".h5"):
                    filepath = os.path.join(cycle_path, filename)
                    with self._get_lock(filepath):
                        if os.path.exists(filepath):
                            os.remove(filepath)
            os.rmdir(cycle_path)
