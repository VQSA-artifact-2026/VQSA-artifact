from __future__ import annotations

import copy
import os
import random
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# ============================================================
# 1. Global configuration
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
    "figure.dpi": 150,
    "savefig.dpi": 400,
})


@dataclass
class FLConfig:
    dataset_name: str = "MNIST"
    num_clients: int = 4
    rounds: int = 20
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    bits: int = 8
    clip_bound: float = 1.0
    max_train_samples: int = 12000
    max_test_samples: int = 2000
    save_dir: str = "./qfl_task_consistency_outputs"


# ============================================================
# 2. Model definition
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# 3. Dataset and DataLoader
# ============================================================
def build_dataset(dataset_name: str, root: str, train: bool):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if dataset_name == "MNIST":
        return datasets.MNIST(root=root, train=train, download=True, transform=transform)
    if dataset_name == "FashionMNIST":
        return datasets.FashionMNIST(root=root, train=train, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")



def build_federated_loaders(cfg: FLConfig) -> Tuple[List[DataLoader], DataLoader]:
    data_root = os.path.join(cfg.save_dir, "data")

    train_full = build_dataset(cfg.dataset_name, root=data_root, train=True)
    test_full = build_dataset(cfg.dataset_name, root=data_root, train=False)

    train_subset = Subset(train_full, list(range(min(cfg.max_train_samples, len(train_full)))))
    test_subset = Subset(test_full, list(range(min(cfg.max_test_samples, len(test_full)))))

    part_size = len(train_subset) // cfg.num_clients
    lengths = [part_size] * (cfg.num_clients - 1)
    lengths.append(len(train_subset) - sum(lengths))

    client_subsets = random_split(
        train_subset,
        lengths,
        generator=torch.Generator().manual_seed(SEED),
    )

    client_loaders = [DataLoader(ds, batch_size=cfg.batch_size, shuffle=True) for ds in client_subsets]
    test_loader = DataLoader(test_subset, batch_size=cfg.batch_size, shuffle=False)
    return client_loaders, test_loader


# ============================================================
# 4. State utilities
# ============================================================
def get_state_copy(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())



def load_state(model: nn.Module, state: "OrderedDict[str, torch.Tensor]") -> None:
    model.load_state_dict(state, strict=True)



def flatten_state(state: "OrderedDict[str, torch.Tensor]") -> torch.Tensor:
    return torch.cat([v.reshape(-1).float() for v in state.values()])



def state_sub(a: "OrderedDict[str, torch.Tensor]", b: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, a[k] - b[k]) for k in a.keys())



def state_add(a: "OrderedDict[str, torch.Tensor]", b: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, a[k] + b[k]) for k in a.keys())



def state_scalar_div(a: "OrderedDict[str, torch.Tensor]", value: float) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, a[k] / value) for k in a.keys())



def average_states(states: List["OrderedDict[str, torch.Tensor]"]) -> "OrderedDict[str, torch.Tensor]":
    avg = OrderedDict((k, torch.zeros_like(v)) for k, v in states[0].items())
    for state in states:
        for k in avg.keys():
            avg[k] += state[k]
    return state_scalar_div(avg, float(len(states)))


# ============================================================
# 5. Quantization / dequantization
# ============================================================
def quantize_tensor(x: torch.Tensor, bits: int, B: float) -> torch.Tensor:
    clipped = torch.clamp(x, -B, B)
    shifted = clipped + B
    L = (2 ** bits) - 1
    q = torch.round((shifted / (2 * B)) * L)
    return q.to(torch.int64)



def dequantize_tensor(q: torch.Tensor, bits: int, B: float) -> torch.Tensor:
    L = (2 ** bits) - 1
    return (q.float() / L) * (2 * B) - B



def quantize_state(state: "OrderedDict[str, torch.Tensor]", bits: int, B: float) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, quantize_tensor(v, bits, B)) for k, v in state.items())



def dequantize_state(state: "OrderedDict[str, torch.Tensor]", bits: int, B: float) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, dequantize_tensor(v, bits, B)) for k, v in state.items())



def semantic_equivalent_quantized_average(
    deltas: List["OrderedDict[str, torch.Tensor]"], bits: int, B: float
) -> Tuple["OrderedDict[str, torch.Tensor]", Dict[str, "OrderedDict[str, torch.Tensor]"]]:
    q_states = [quantize_state(delta, bits, B) for delta in deltas]

    q_avg = OrderedDict((k, torch.zeros_like(v)) for k, v in q_states[0].items())
    for q_state in q_states:
        for k in q_avg.keys():
            q_avg[k] += q_state[k]

    num_clients = len(q_states)
    for k in q_avg.keys():
        q_avg[k] = torch.round(q_avg[k].float() / num_clients).to(torch.int64)

    dq_avg = dequantize_state(q_avg, bits, B)
    return dq_avg, {"quantized_average": q_avg}


# ============================================================
# 6. Local training and evaluation
# ============================================================
def train_one_client_from_global(
    global_state: "OrderedDict[str, torch.Tensor]", loader: DataLoader, cfg: FLConfig
) -> "OrderedDict[str, torch.Tensor]":
    model = SimpleCNN().to(DEVICE)
    load_state(model, global_state)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(cfg.local_epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    local_state = get_state_copy(model)
    return state_sub(local_state, global_state)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ============================================================
# 7. Consistency metrics
# ============================================================
def cosine_similarity_state(state_a: "OrderedDict[str, torch.Tensor]", state_b: "OrderedDict[str, torch.Tensor]") -> float:
    a = flatten_state(state_a)
    b = flatten_state(state_b)
    denom = torch.norm(a, p=2) * torch.norm(b, p=2)
    if float(denom) == 0.0:
        return 1.0
    return float(torch.dot(a, b) / denom)



def mae_state(state_a: "OrderedDict[str, torch.Tensor]", state_b: "OrderedDict[str, torch.Tensor]") -> float:
    a = flatten_state(state_a)
    b = flatten_state(state_b)
    return float(torch.mean(torch.abs(a - b)))


# ============================================================
# 8. Single experiment run
# ============================================================
def run_task_level_consistency_experiment(cfg: FLConfig) -> pd.DataFrame:
    os.makedirs(cfg.save_dir, exist_ok=True)
    client_loaders, test_loader = build_federated_loaders(cfg)

    classical_model = SimpleCNN().to(DEVICE)
    quantized_model = SimpleCNN().to(DEVICE)
    quantized_model.load_state_dict(copy.deepcopy(classical_model.state_dict()))

    history: List[Dict[str, float]] = []

    # Round 0: record the initial models
    classical_global = get_state_copy(classical_model)
    quantized_global = get_state_copy(quantized_model)
    classical_loss, classical_acc = evaluate_model(classical_model, test_loader)
    quantized_loss, quantized_acc = evaluate_model(quantized_model, test_loader)
    history.append({
        "round": 0,
        "classical_test_loss": classical_loss,
        "classical_test_acc": classical_acc,
        "quantized_test_loss": quantized_loss,
        "quantized_test_acc": quantized_acc,
        "model_cossim": cosine_similarity_state(classical_global, quantized_global),
        "model_mae": mae_state(classical_global, quantized_global),
        "delta_cossim": 1.0,
        "delta_mae": 0.0,
        "bits": cfg.bits,
        "num_clients": cfg.num_clients,
        "dataset": cfg.dataset_name,
    })
    print(
        f"[{cfg.dataset_name} | m={cfg.bits} | Round 00] "
        f"Classical acc={classical_acc:.4f}, Quantized acc={quantized_acc:.4f}, "
        f"Model CosSim=1.000000, Model MAE=0.00000000"
    )

    for rnd in range(1, cfg.rounds + 1):
        classical_global = get_state_copy(classical_model)
        quantized_global = get_state_copy(quantized_model)

        classical_client_deltas = [train_one_client_from_global(classical_global, loader, cfg) for loader in client_loaders]
        avg_delta = average_states(classical_client_deltas)
        classical_next = state_add(classical_global, avg_delta)
        load_state(classical_model, classical_next)

        quantized_client_deltas = [train_one_client_from_global(quantized_global, loader, cfg) for loader in client_loaders]
        qavg_delta, _ = semantic_equivalent_quantized_average(quantized_client_deltas, bits=cfg.bits, B=cfg.clip_bound)
        quantized_next = state_add(quantized_global, qavg_delta)
        load_state(quantized_model, quantized_next)

        classical_loss, classical_acc = evaluate_model(classical_model, test_loader)
        quantized_loss, quantized_acc = evaluate_model(quantized_model, test_loader)

        row = {
            "round": rnd,
            "classical_test_loss": classical_loss,
            "classical_test_acc": classical_acc,
            "quantized_test_loss": quantized_loss,
            "quantized_test_acc": quantized_acc,
            "model_cossim": cosine_similarity_state(classical_next, quantized_next),
            "model_mae": mae_state(classical_next, quantized_next),
            "delta_cossim": cosine_similarity_state(avg_delta, qavg_delta),
            "delta_mae": mae_state(avg_delta, qavg_delta),
            "bits": cfg.bits,
            "num_clients": cfg.num_clients,
            "dataset": cfg.dataset_name,
        }
        history.append(row)

        print(
            f"[{cfg.dataset_name} | m={cfg.bits} | Round {rnd:02d}] "
            f"Classical acc={classical_acc:.4f}, Quantized acc={quantized_acc:.4f}, "
            f"Model CosSim={row['model_cossim']:.6f}, Model MAE={row['model_mae']:.8f}"
        )

    return pd.DataFrame(history)


# ============================================================
# 9. Batch execution
# ============================================================
def run_grid_experiments(
    datasets_to_run: List[str],
    num_clients_list: List[int],
    bits_list: List[int],
    rounds: int = 20,
    local_epochs: int = 1,
    batch_size: int = 64,
    lr: float = 1e-3,
    clip_bound: float = 1.0,
    max_train_samples: int = 12000,
    max_test_samples: int = 2000,
    save_root: str = "./qfl_task_consistency_outputs",
) -> pd.DataFrame:
    all_results: List[pd.DataFrame] = []

    for dataset_name in datasets_to_run:
        for num_clients in num_clients_list:
            for bits in bits_list:
                print("\n" + "=" * 80)
                print(f"Starting experiment: dataset={dataset_name}, n={num_clients}, m={bits}")
                print("=" * 80)

                run_dir = os.path.join(save_root, dataset_name, f"n{num_clients}_m{bits}")
                cfg = FLConfig(
                    dataset_name=dataset_name,
                    num_clients=num_clients,
                    rounds=rounds,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    bits=bits,
                    clip_bound=clip_bound,
                    max_train_samples=max_train_samples,
                    max_test_samples=max_test_samples,
                    save_dir=run_dir,
                )

                df = run_task_level_consistency_experiment(cfg)
                df["setting"] = df.apply(lambda row: f"n={int(row['num_clients'])}, m={int(row['bits'])}", axis=1)

                os.makedirs(run_dir, exist_ok=True)
                csv_path = os.path.join(run_dir, f"{dataset_name.lower()}_n{num_clients}_m{bits}.csv")
                config_path = os.path.join(run_dir, f"{dataset_name.lower()}_n{num_clients}_m{bits}_config.json")
                df.to_csv(csv_path, index=False)
                pd.Series(asdict(cfg)).to_json(config_path, force_ascii=False, indent=2)
                print(f"Metrics table saved: {csv_path}")
                all_results.append(df)

    merged = pd.concat(all_results, ignore_index=True)
    os.makedirs(save_root, exist_ok=True)
    merged_path = os.path.join(save_root, "all_metrics.csv")
    merged.to_csv(merged_path, index=False)
    print(f"\nMerged metrics table saved: {merged_path}")
    return merged


if __name__ == "__main__":
    all_df = run_grid_experiments(
        datasets_to_run=["MNIST", "FashionMNIST"],
        num_clients_list=[4],
        bits_list=[4, 8, 12, 16],
        rounds=20,
        local_epochs=1,
        batch_size=64,
        lr=1e-3,
        clip_bound=1.0,
        max_train_samples=12000,
        max_test_samples=2000,
        save_root="./qfl_task_consistency_outputs",
    )
    print(f"\nGenerated {len(all_df)} records.")
