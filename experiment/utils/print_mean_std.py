import torch


def get_mean_std(results: list[dict]) -> dict:
    tensor_data = torch.tensor([list(d.values()) for d in results])

    mean = tensor_data.mean(dim=0)
    std_dev = tensor_data.std(dim=0)

    return {
        key: {"mean": m.item(), "std": s.item()}
        for key, m, s in zip(results[0].keys(), mean, std_dev)
    }


def print_mean_std(results: list[dict]) -> None:
    mean_std = get_mean_std(results)

    for key, values in mean_std.items():
        print(f"{key} - Mean: {values['mean']}, Std: {values['std']}")
