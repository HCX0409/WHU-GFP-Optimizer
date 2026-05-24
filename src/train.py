# train_brightness_v2.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import os

# ================= 路径配置 =================
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
EMB_PATH   = ROOT / "data/processed/gfp_esm2_emb.npy"
LABEL_PATH = ROOT / "data/processed/gfp_labels.parquet"
MODEL_DIR  = ROOT / "models"
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================================

print(f"🚀 使用设备: {DEVICE}")
X = np.load(EMB_PATH).astype(np.float32)
y = pd.read_parquet(LABEL_PATH)["Brightness"].values.astype(np.float32)
assert X.shape[0] == y.shape[0], "⚠️ 特征与标签数量不匹配！"

# 标准化（MLP 必需）
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分: 80% train | 10% val | 10% test
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# DataLoader
train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
val_ds   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=0)

# 🧠 MLP 架构（残差+BN+Dropout，专为稠密蛋白表征设计）
class BrightnessMLP(nn.Module):
    def __init__(self, in_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

model = BrightnessMLP().to(DEVICE)
criterion = nn.HuberLoss(delta=0.5)  # 对生物实验异常值鲁棒
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 🔄 训练循环
best_val_mae = np.inf
patience, wait = 15, 0
print("⏳ 开始训练 MLP (约 3~5 分钟)...")
for epoch in range(100):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    
    # 验证
    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            val_preds.append(model(xb.to(DEVICE)).cpu().numpy())
            val_trues.append(yb.numpy())
    val_preds = np.concatenate(val_preds)
    val_trues = np.concatenate(val_trues)
    val_mae = mean_absolute_error(val_trues, val_preds)
    scheduler.step(val_mae)
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "gfp_mlp_best.pth"))
        np.save(os.path.join(MODEL_DIR, "scaler_mean.npy"), scaler.mean_)
        np.save(os.path.join(MODEL_DIR, "scaler_scale.npy"), scaler.scale_)
        wait = 0
    else:
        wait += 1
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:03d} | TrainLoss: {train_loss/len(X_tr):.4f} | ValMAE: {val_mae:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    if wait >= patience:
        print(f"⏹️ 早停触发 @ Epoch {epoch+1}")
        break

# 📊 测试集评估
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gfp_mlp_best.pth"), weights_only=True))
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_te).to(DEVICE)).cpu().numpy()
r2 = r2_score(y_te, y_pred)
mae = mean_absolute_error(y_te, y_pred)
print("\n" + "="*40)
print(f"📈 测试集评估 (log10尺度) | R² = {r2:.4f} | MAE = {mae:.4f}")
print("="*40)
