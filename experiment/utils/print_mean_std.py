import torch


def print_mean_std(results: list[dict]) -> None:
    tensor_data = torch.tensor([list(d.values()) for d in results])

    mean = tensor_data.mean(dim=0)
    std_dev = tensor_data.std(dim=0)

    for key, m, s in zip(results[0].keys(), mean, std_dev):
        print(f"{key} - Mean: {m.item()}, Std: {s.item()}")
