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
    save_dir: str = "./qfl_preprocess_inconsistency_outputs"
    attack_client_idx: int = 0
    attack_noise_std: float = 0.05


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
    return OrderedDict((k, a[k] / value) for k, v in a.items())



def average_states(states: List["OrderedDict[str, torch.Tensor]"]) -> "OrderedDict[str, torch.Tensor]":
    avg = OrderedDict((k, torch.zeros_like(v)) for k, v in states[0].items())
    for state in states:
        for k in avg.keys():
            avg[k] += state[k]
    return state_scalar_div(avg, float(len(states)))



def add_gaussian_noise_to_state(
    state: "OrderedDict[str, torch.Tensor]",
    std: float,
    seed: int,
) -> "OrderedDict[str, torch.Tensor]":
    gen = torch.Generator().manual_seed(seed)
    noisy = OrderedDict()
    for k, v in state.items():
        noise = torch.randn(v.shape, generator=gen, dtype=v.dtype) * std
        noisy[k] = v + noise
    return noisy


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



def pairwise_consistency_metrics(states: List["OrderedDict[str, torch.Tensor]"]) -> Tuple[float, float]:
    if len(states) <= 1:
        return 0.0, 1.0

    maes: List[float] = []
    cossims: List[float] = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            maes.append(mae_state(states[i], states[j]))
            cossims.append(cosine_similarity_state(states[i], states[j]))
    return float(np.mean(maes)), float(np.mean(cossims))


# ============================================================
# 8. Single-round execution
# ============================================================
def run_one_round_with_client_states(
    client_states: List["OrderedDict[str, torch.Tensor]"],
    client_loaders: List[DataLoader],
    cfg: FLConfig,
) -> Tuple[List["OrderedDict[str, torch.Tensor]"], "OrderedDict[str, torch.Tensor]", Dict[str, float]]:
    client_deltas = [
        train_one_client_from_global(client_states[i], client_loaders[i], cfg)
        for i in range(cfg.num_clients)
    ]

    qavg_delta, _ = semantic_equivalent_quantized_average(client_deltas, bits=cfg.bits, B=cfg.clip_bound)
    next_client_states = [state_add(state, qavg_delta) for state in client_states]

    pairwise_mae, pairwise_cossim = pairwise_consistency_metrics(next_client_states)
    metrics = {
        "pairwise_mae": pairwise_mae,
        "pairwise_cossim": pairwise_cossim,
    }
    return next_client_states, qavg_delta, metrics


# ============================================================
# 9. Main experiment pipeline
# ============================================================
def run_preprocess_inconsistency_experiment(cfg: FLConfig) -> pd.DataFrame:
    os.makedirs(cfg.save_dir, exist_ok=True)
    client_loaders, test_loader = build_federated_loaders(cfg)

    # ===== Build initial states for the normal group =====
    base_model = SimpleCNN().to(DEVICE)
    base_global_state = get_state_copy(base_model)
    normal_client_states = [copy.deepcopy(base_global_state) for _ in range(cfg.num_clients)]

    # ===== Build initial states for the attack group: perturb one client only during preprocessing =====
    attack_client_states = [copy.deepcopy(base_global_state) for _ in range(cfg.num_clients)]
    attack_idx = cfg.attack_client_idx
    attack_client_states[attack_idx] = add_gaussian_noise_to_state(
        attack_client_states[attack_idx],
        std=cfg.attack_noise_std,
        seed=SEED + 1000 + attack_idx,
    )

    history: List[Dict[str, float]] = []

    def eval_global_from_state(state: "OrderedDict[str, torch.Tensor]") -> Tuple[float, float]:
        model = SimpleCNN().to(DEVICE)
        load_state(model, state)
        return evaluate_model(model, test_loader)

    # Round 0
    normal_mae0, normal_cossim0 = pairwise_consistency_metrics(normal_client_states)
    attack_mae0, attack_cossim0 = pairwise_consistency_metrics(attack_client_states)
    normal_loss0, normal_acc0 = eval_global_from_state(normal_client_states[0])
    attack_loss0, attack_acc0 = eval_global_from_state(attack_client_states[0])

    history.append({
        "round": 0,
        "normal_pairwise_mae": normal_mae0,
        "normal_pairwise_cossim": normal_cossim0,
        "normal_test_acc": normal_acc0,
        "normal_test_loss": normal_loss0,
        "attack_pairwise_mae": attack_mae0,
        "attack_pairwise_cossim": attack_cossim0,
        "attack_test_acc": attack_acc0,
        "attack_test_loss": attack_loss0,
        "attack_client_idx": cfg.attack_client_idx + 1,
        "attack_noise_std": cfg.attack_noise_std,
        "bits": cfg.bits,
        "num_clients": cfg.num_clients,
        "dataset": cfg.dataset_name,
    })

    print(
        f"[Round 00] Normal: MAE={normal_mae0:.8f}, CosSim={normal_cossim0:.6f}, Acc={normal_acc0:.4f} | "
        f"Attack: MAE={attack_mae0:.8f}, CosSim={attack_cossim0:.6f}, Acc={attack_acc0:.4f}"
    )

    for rnd in range(1, cfg.rounds + 1):
        normal_client_states, _, normal_metrics = run_one_round_with_client_states(
            normal_client_states, client_loaders, cfg
        )
        attack_client_states, _, attack_metrics = run_one_round_with_client_states(
            attack_client_states, client_loaders, cfg
        )

        normal_loss, normal_acc = eval_global_from_state(normal_client_states[0])
        attack_loss, attack_acc = eval_global_from_state(attack_client_states[0])

        row = {
            "round": rnd,
            "normal_pairwise_mae": normal_metrics["pairwise_mae"],
            "normal_pairwise_cossim": normal_metrics["pairwise_cossim"],
            "normal_test_acc": normal_acc,
            "normal_test_loss": normal_loss,
            "attack_pairwise_mae": attack_metrics["pairwise_mae"],
            "attack_pairwise_cossim": attack_metrics["pairwise_cossim"],
            "attack_test_acc": attack_acc,
            "attack_test_loss": attack_loss,
            "attack_client_idx": cfg.attack_client_idx + 1,
            "attack_noise_std": cfg.attack_noise_std,
            "bits": cfg.bits,
            "num_clients": cfg.num_clients,
            "dataset": cfg.dataset_name,
        }
        history.append(row)

        print(
            f"[Round {rnd:02d}] Normal: MAE={row['normal_pairwise_mae']:.8f}, "
            f"CosSim={row['normal_pairwise_cossim']:.6f}, Acc={row['normal_test_acc']:.4f} | "
            f"Attack: MAE={row['attack_pairwise_mae']:.8f}, "
            f"CosSim={row['attack_pairwise_cossim']:.6f}, Acc={row['attack_test_acc']:.4f}"
        )

    df = pd.DataFrame(history)
    csv_path = os.path.join(cfg.save_dir, "preprocess_inconsistency_metrics.csv")
    json_path = os.path.join(cfg.save_dir, "preprocess_inconsistency_config.json")
    df.to_csv(csv_path, index=False)
    pd.Series(asdict(cfg)).to_json(json_path, force_ascii=False, indent=2)
    print(f"\nMetrics table saved: {csv_path}")
    return df


# ============================================================
# 10. Plotting
# ============================================================
def plot_preprocess_inconsistency_results(df: pd.DataFrame, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # MAE
    plt.figure()
    plt.plot(df["round"], df["normal_pairwise_mae"], label="Normal")
    plt.plot(df["round"], df["attack_pairwise_mae"], label="Attack")
    plt.xlabel("Round")
    plt.ylabel("Pairwise MAE")
    plt.title("Model Consistency Error (MAE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pairwise_mae.png"))
    plt.close()

    # CosSim
    plt.figure()
    plt.plot(df["round"], df["normal_pairwise_cossim"], label="Normal")
    plt.plot(df["round"], df["attack_pairwise_cossim"], label="Attack")
    plt.xlabel("Round")
    plt.ylabel("Pairwise CosSim")
    plt.title("Model Consistency Similarity (CosSim)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pairwise_cossim.png"))
    plt.close()

    # Acc
    plt.figure()
    plt.plot(df["round"], df["normal_test_acc"], label="Normal")
    plt.plot(df["round"], df["attack_test_acc"], label="Attack")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.title("Task Accuracy Under Preprocessing Inconsistency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_accuracy.png"))
    plt.close()

    print(f"Plots saved to: {save_dir}")


if __name__ == "__main__":
    cfg = FLConfig(
        dataset_name="MNIST",
        num_clients=4,
        rounds=20,
        local_epochs=1,
        batch_size=64,
        lr=1e-3,
        bits=16,
        clip_bound=1.0,
        max_train_samples=12000,
        max_test_samples=2000,
        save_dir="./qfl_preprocess_inconsistency_outputs/MNIST_n4_m16",
        attack_client_idx=0,   # 0 means P1
        attack_noise_std=0.05, # Can be adjusted to 0.01 / 0.05 / 0.1
    )

    df = run_preprocess_inconsistency_experiment(cfg)
    plot_preprocess_inconsistency_results(df, cfg.save_dir)
    print(f"\nGenerated {len(df)} records.")
