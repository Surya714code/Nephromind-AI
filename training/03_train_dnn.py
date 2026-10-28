"""
NephroMind AI — Step 3: Train Deep Neural Network (PyTorch MLP)
================================================================
Architecture:
  Input (8 features)
  → Dense 128 + ReLU + BatchNorm + Dropout(0.30)
  → Dense 64  + ReLU + BatchNorm + Dropout(0.25)
  → Dense 32  + ReLU + Dropout(0.20)
  → Dense 16  + ReLU
  → Dense 1   + Sigmoid

Training:
  - Adam optimizer
  - Binary Cross Entropy with class weights
  - EarlyStopping (patience=15)
  - ReduceLROnPlateau (patience=7)
  - Fixed seed: 42
  - 70% train / 15% val / 15% test (split already done in step 2)

Saves:
  models/nephromind_dnn.pt
  models/training_history.json
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

RANDOM_SEED  = 42
BATCH_SIZE   = 256
MAX_EPOCHS   = 200
LR           = 1e-3
PATIENCE_ES  = 15   # early stopping patience
PATIENCE_LR  = 7    # reduce-LR patience
LR_FACTOR    = 0.5
LR_MIN       = 1e-6

# ── Seed everything ──────────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print("NephroMind AI — DNN Training (PyTorch)")
print("=" * 60)
print(f"Device: {DEVICE}")

# ── Load prepared data ────────────────────────────────────────────────────────
X_train = np.load(MODELS_DIR / "X_train.npy")
X_val   = np.load(MODELS_DIR / "X_val.npy")
y_train = np.load(MODELS_DIR / "y_train.npy")
y_val   = np.load(MODELS_DIR / "y_val.npy")

with open(MODELS_DIR / "nephromind_features.json") as f:
    features_meta = json.load(f)

n_features = features_meta["n_features"]
print(f"\nFeatures ({n_features}): {features_meta['feature_names']}")
print(f"Train: {len(X_train)}  Val: {len(X_val)}")

# ── Class weights ────────────────────────────────────────────────────────────
classes = np.array([0, 1])
cw = compute_class_weight("balanced", classes=classes, y=y_train)
pos_weight = torch.tensor([cw[1] / cw[0]], dtype=torch.float32, device=DEVICE)
print(f"\nClass weights: neg={cw[0]:.4f}  pos={cw[1]:.4f}")
print(f"BCEWithLogitsLoss pos_weight: {pos_weight.item():.4f}")

# ── DataLoaders ───────────────────────────────────────────────────────────────
def make_loader(X, y, shuffle=False):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val, y_val, shuffle=False)


# ── Model Architecture ────────────────────────────────────────────────────────
class NephroMindDNN(nn.Module):
    """
    Tabular MLP for kidney disease risk classification.
    Input: 8 clinical/physiological features from Renal Sentinel hardware.
    Output: logit (sigmoid applied externally for prediction).
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 128 units
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.30),
            # Block 2: 64 units
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            # Block 3: 32 units
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),
            # Block 4: 16 units
            nn.Linear(32, 16),
            nn.ReLU(),
            # Output
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # shape: (batch,)


model = NephroMindDNN(n_features=n_features).to(DEVICE)
print(f"\nModel architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# ── Optimizer and Loss ────────────────────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=LR_FACTOR, patience=PATIENCE_LR,
    min_lr=LR_MIN
)

# ── Training loop ─────────────────────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "lr": []}
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0
best_state = None

print(f"\nTraining: max_epochs={MAX_EPOCHS}  batch={BATCH_SIZE}  "
      f"lr={LR}  early_stop_patience={PATIENCE_ES}")
print("-" * 60)

for epoch in range(1, MAX_EPOCHS + 1):
    # ── Train ──
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_losses.append(loss.item())

    # ── Validate ──
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss   = np.mean(val_losses)
    current_lr = optimizer.param_groups[0]["lr"]

    history["train_loss"].append(float(train_loss))
    history["val_loss"].append(float(val_loss))
    history["lr"].append(float(current_lr))

    # Log every 10 epochs
    if epoch % 10 == 0 or epoch <= 5:
        print(f"  Epoch {epoch:3d}/{MAX_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"lr={current_lr:.2e}")

    # ── Scheduler step ──
    scheduler.step(val_loss)

    # ── Early stopping ──
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE_ES:
        print(f"\n  Early stopping at epoch {epoch} "
              f"(best val_loss={best_val_loss:.4f} at epoch {best_epoch})")
        break

print(f"\nTraining complete. Best epoch: {best_epoch}  "
      f"Best val_loss: {best_val_loss:.4f}")

# ── Restore best weights ──────────────────────────────────────────────────────
model.load_state_dict(best_state)

# ── Save model ───────────────────────────────────────────────────────────────
torch.save({
    "model_state_dict": model.state_dict(),
    "n_features": n_features,
    "feature_names": features_meta["feature_names"],
    "model_version": "nephromind_dnn_v1",
    "architecture": "NephroMindDNN",
    "best_epoch": best_epoch,
    "best_val_loss": float(best_val_loss),
}, MODELS_DIR / "nephromind_dnn.pt")
print(f"\nSaved: models/nephromind_dnn.pt")

# ── Save training history ────────────────────────────────────────────────────
history["best_epoch"]   = best_epoch
history["best_val_loss"] = float(best_val_loss)
history["total_epochs"] = epoch

with open(MODELS_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)
print(f"Saved: models/training_history.json")

print(f"\n✓ DNN TRAINING COMPLETE")
