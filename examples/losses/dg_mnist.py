"""MNIST classification via policy gradient — comparison of loss functions.

Frames MNIST classification as a bandit problem:
  - State: MNIST image (28x28)
  - Action: digit prediction (0-9)
  - Reward: +1 if correct, -1 if incorrect

Trains CE, PG (REINFORCE), GRPO, and DG — plots test error over training batches.

References:
  - Delightful Policy Gradient (arXiv:2603.14608)
  - Delightful Distributed Policy Gradient (arXiv:2603.20521)
  - DeepSeekMath / GRPO (arXiv:2402.03300)
"""

import argparse
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from torch.distributions import Categorical
from torchvision import datasets, transforms

from torchtrade.losses import DGLoss, GRPOLoss


def make_mlp(obs_dim=784, n_actions=10, hidden=128):
    """Shared architecture for all methods."""
    return nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )


def make_actor(obs_dim=784, n_actions=10, hidden=128):
    """Create a TorchRL actor wrapping the same MLP."""
    policy_net = TensorDictModule(
        make_mlp(obs_dim, n_actions, hidden),
        in_keys=["observation"],
        out_keys=["logits"],
    )
    actor = ProbabilisticTensorDictModule(
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=Categorical,
        return_log_prob=True,
    )
    return ProbabilisticTensorDictSequential(policy_net, actor)


def make_data_loader(train=True, batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(
        root="/tmp/mnist_data", train=train, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=2
    )


def compute_test_error(model, test_loader, device):
    """Compute test error rate (%) for a plain nn.Module."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return 100.0 * (1 - correct / total)


def compute_test_error_dg(loss_module, test_loader, device):
    """Compute test error rate (%) for DGLoss actor."""
    loss_module.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            td = TensorDict(
                {"observation": images},
                batch_size=[images.size(0)],
                device=device,
            )
            with loss_module.actor_network_params.to_module(loss_module.actor_network):
                td = loss_module.actor_network(td)
            preds = td["logits"].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    loss_module.train()
    return 100.0 * (1 - correct / total)


def train_ce(epochs=5, lr=1e-3, batch_size=256, eval_every=10, device=None):
    """Cross-Entropy (supervised). Returns metrics dict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = make_mlp().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = make_data_loader(train=True, batch_size=batch_size)
    test_loader = make_data_loader(train=False, batch_size=1000)

    metrics = {"batch_loss": [], "batch_test_error": [], "eval_steps": []}
    step = 0

    print(f"Training CE (cross-entropy) on {device}")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics["batch_loss"].append(loss.item())
            step += 1

            if step % eval_every == 0:
                err = compute_test_error(model, test_loader, device)
                metrics["batch_test_error"].append(err)
                metrics["eval_steps"].append(step)

        err = compute_test_error(model, test_loader, device)
        dt = time.time() - t0
        print(f"  Epoch {epoch}: test error {err:.2f}%, {dt:.1f}s")

    return metrics


def train_pg(epochs=5, lr=1e-3, batch_size=256, eval_every=10, device=None):
    """REINFORCE (PG) with mean baseline. Returns metrics dict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = make_mlp().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = make_data_loader(train=True, batch_size=batch_size)
    test_loader = make_data_loader(train=False, batch_size=1000)

    metrics = {"batch_loss": [], "batch_test_error": [], "eval_steps": []}
    step = 0

    print(f"Training PG (REINFORCE) on {device}")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            logits = model(images)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)

            reward = torch.where(
                actions == labels,
                torch.ones(1, device=device),
                -torch.ones(1, device=device),
            ).float()

            advantage = reward - reward.mean()
            loss = -(log_prob * advantage.detach()).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics["batch_loss"].append(loss.item())
            step += 1

            if step % eval_every == 0:
                err = compute_test_error(model, test_loader, device)
                metrics["batch_test_error"].append(err)
                metrics["eval_steps"].append(step)

        err = compute_test_error(model, test_loader, device)
        dt = time.time() - t0
        print(f"  Epoch {epoch}: test error {err:.2f}%, {dt:.1f}s")

    return metrics


def train_grpo(epochs=5, lr=1e-3, batch_size=256, eval_every=10,
               group_size=8, device=None):
    """GRPO with group-relative advantages. Returns metrics dict.

    For each image, samples `group_size` actions from the policy, computes
    rewards for each, then normalizes advantages within each group. This is
    the key GRPO mechanism: the batch has shape (group_size, B) and
    mean(0) computes per-image baselines across the group.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = make_actor().to(device)
    loss_module = GRPOLoss(
        actor_network=actor,
        entropy_bonus=True,
        entropy_coeff=0.01,
        epsilon_low=0.2,
        epsilon_high=0.2,
    )
    optimizer = torch.optim.Adam(
        loss_module.actor_network_params.values(True, True), lr=lr
    )
    train_loader = make_data_loader(train=True, batch_size=batch_size)
    test_loader = make_data_loader(train=False, batch_size=1000)

    metrics = {"batch_loss": [], "batch_test_error": [], "eval_steps": []}
    step = 0

    print(f"Training GRPO (G={group_size}) on {device}")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss_module.train()
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)  # (B, 784)
            labels = labels.to(device)  # (B,)
            B = images.size(0)

            # Expand each image G times: (G, B, 784)
            images_g = images.unsqueeze(0).expand(group_size, -1, -1)
            labels_g = labels.unsqueeze(0).expand(group_size, -1)

            # Sample G actions per image
            td = TensorDict(
                {"observation": images_g},
                batch_size=[group_size, B],
                device=device,
            )
            with torch.no_grad():
                with loss_module.actor_network_params.to_module(loss_module.actor_network):
                    td = loss_module.actor_network(td)

            actions = td["action"]  # (G, B)
            old_log_prob = td["action_log_prob"]  # (G, B)
            reward = torch.where(
                actions == labels_g,
                torch.ones(1, device=device),
                -torch.ones(1, device=device),
            ).unsqueeze(-1).float()  # (G, B, 1)

            # GRPO loss normalizes advantage via mean(0) across the group dim
            loss_td = TensorDict(
                {
                    "observation": images_g,
                    "action": actions,
                    "action_log_prob": old_log_prob,
                    "next": {"reward": reward},
                },
                batch_size=[group_size, B],
                device=device,
            )

            optimizer.zero_grad()
            output = loss_module(loss_td)
            total_loss = output["loss_objective"] + output["loss_entropy"]
            total_loss.backward()
            optimizer.step()

            metrics["batch_loss"].append(output["loss_objective"].item())
            step += 1

            if step % eval_every == 0:
                err = compute_test_error_dg(loss_module, test_loader, device)
                metrics["batch_test_error"].append(err)
                metrics["eval_steps"].append(step)

        err = compute_test_error_dg(loss_module, test_loader, device)
        dt = time.time() - t0
        print(f"  Epoch {epoch}: test error {err:.2f}%, {dt:.1f}s")

    return metrics


def train_dg(epochs=5, eta=1.0, baseline="mean", lr=1e-3, batch_size=256,
             eval_every=10, device=None):
    """Delightful Gradient. Returns metrics dict."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = make_actor().to(device)
    loss_module = DGLoss(
        actor_network=actor,
        eta=eta,
        baseline=baseline,
        entropy_bonus=True,
        entropy_coeff=0.01,
    )
    optimizer = torch.optim.Adam(
        loss_module.actor_network_params.values(True, True), lr=lr
    )
    train_loader = make_data_loader(train=True, batch_size=batch_size)
    test_loader = make_data_loader(train=False, batch_size=1000)

    metrics = {
        "batch_loss": [],
        "batch_test_error": [],
        "eval_steps": [],
        "batch_gate": [],
    }
    step = 0

    print(f"Training DG (eta={eta}, baseline={baseline}) on {device}")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss_module.train()
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            td = TensorDict(
                {"observation": images},
                batch_size=[images.size(0)],
                device=device,
            )
            with torch.no_grad():
                with loss_module.actor_network_params.to_module(loss_module.actor_network):
                    td = loss_module.actor_network(td)

            actions = td["action"]
            reward = torch.where(
                actions == labels,
                torch.ones(1, device=device),
                -torch.ones(1, device=device),
            ).unsqueeze(-1).float()

            loss_td = TensorDict(
                {
                    "observation": images,
                    "action": actions,
                    "next": {"reward": reward},
                },
                batch_size=[images.size(0)],
                device=device,
            )

            optimizer.zero_grad()
            output = loss_module(loss_td)
            total_loss = output["loss_objective"] + output["loss_entropy"]
            total_loss.backward()
            optimizer.step()

            metrics["batch_loss"].append(output["loss_objective"].item())
            metrics["batch_gate"].append(output["gate"].item())
            step += 1

            if step % eval_every == 0:
                err = compute_test_error_dg(loss_module, test_loader, device)
                metrics["batch_test_error"].append(err)
                metrics["eval_steps"].append(step)

        err = compute_test_error_dg(loss_module, test_loader, device)
        dt = time.time() - t0
        print(f"  Epoch {epoch}: test error {err:.2f}%, {dt:.1f}s")

    return metrics


def smooth(values, alpha=0.9):
    """Exponential moving average."""
    out = []
    val = values[0]
    for v in values:
        val = alpha * val + (1 - alpha) * v
        out.append(val)
    return out


def plot_results(ce, pg, grpo, dg, save_path="dg_vs_reinforce.png"):
    """Plot test error and loss for CE, PG, GRPO, DG."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        "CE vs PG vs GRPO vs DG — MNIST Bandit",
        fontsize=14, fontweight="bold", y=1.02,
    )

    ce_color = "#8b5cf6"    # purple
    pg_color = "#dc2626"    # red
    grpo_color = "#f59e0b"  # amber
    dg_color = "#2563eb"    # blue

    # --- (a) Test Error over training ---
    ax = axes[0]
    ax.plot(ce["eval_steps"], ce["batch_test_error"],
            color=ce_color, linewidth=1.8, label="CE (supervised)")
    ax.plot(pg["eval_steps"], pg["batch_test_error"],
            color=pg_color, linewidth=1.8, label="PG (REINFORCE)")
    ax.plot(grpo["eval_steps"], grpo["batch_test_error"],
            color=grpo_color, linewidth=1.8, label="GRPO")
    ax.plot(dg["eval_steps"], dg["batch_test_error"],
            color=dg_color, linewidth=1.8, label="DG (eta=1)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Test Error (%)")
    ax.set_title("(a) Test error")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- (b) Training loss per batch (smoothed) ---
    ax = axes[1]
    ax.plot(smooth(ce["batch_loss"]), color=ce_color, alpha=0.85, linewidth=1.2, label="CE")
    ax.plot(smooth(pg["batch_loss"]), color=pg_color, alpha=0.85, linewidth=1.2, label="PG")
    ax.plot(smooth(grpo["batch_loss"]), color=grpo_color, alpha=0.85, linewidth=1.2, label="GRPO")
    ax.plot(smooth(dg["batch_loss"]), color=dg_color, alpha=0.85, linewidth=1.2, label="DG")
    ax.plot(ce["batch_loss"], color=ce_color, alpha=0.07, linewidth=0.5)
    ax.plot(pg["batch_loss"], color=pg_color, alpha=0.07, linewidth=0.5)
    ax.plot(grpo["batch_loss"], color=grpo_color, alpha=0.07, linewidth=0.5)
    ax.plot(dg["batch_loss"], color=dg_color, alpha=0.07, linewidth=0.5)
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.set_title("(b) Training loss")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (c) DG gate over training ---
    ax = axes[2]
    gate_color = "#16a34a"
    ax.plot(smooth(dg["batch_gate"]), color=gate_color, alpha=0.85, linewidth=1.5,
            label="Gate sigma(delight/eta)")
    ax.plot(dg["batch_gate"], color=gate_color, alpha=0.08, linewidth=0.5)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="gate = 0.5")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Gate value")
    ax.set_title("(c) DG gate")
    ax.set_ylim(0.35, 0.65)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="DGLoss MNIST sanity check")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--baseline", type=str, default="mean", choices=["mean", "none"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Evaluate test error every N batches")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    ce_metrics = train_ce(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        eval_every=args.eval_every, device=device,
    )

    torch.manual_seed(42)
    pg_metrics = train_pg(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        eval_every=args.eval_every, device=device,
    )

    torch.manual_seed(42)
    grpo_metrics = train_grpo(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        eval_every=args.eval_every, device=device,
    )

    torch.manual_seed(42)
    dg_metrics = train_dg(
        epochs=args.epochs, eta=args.eta, baseline=args.baseline,
        lr=args.lr, batch_size=args.batch_size,
        eval_every=args.eval_every, device=device,
    )

    print(f"\n{'=' * 50}")
    print(f"Final test errors:")
    print(f"  CE:   {ce_metrics['batch_test_error'][-1]:.2f}%")
    print(f"  PG:   {pg_metrics['batch_test_error'][-1]:.2f}%")
    print(f"  GRPO: {grpo_metrics['batch_test_error'][-1]:.2f}%")
    print(f"  DG:   {dg_metrics['batch_test_error'][-1]:.2f}%")

    if not args.no_plot:
        plot_results(
            ce_metrics, pg_metrics, grpo_metrics, dg_metrics,
            save_path="examples/losses/dg_vs_reinforce.png",
        )

    assert dg_metrics["batch_test_error"][-1] < 20.0, "DGLoss failed sanity check"
    print("\nSanity check PASSED.")


if __name__ == "__main__":
    main()
