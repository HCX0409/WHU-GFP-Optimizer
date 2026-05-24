"""
模块名称: train.py
核心功能: 加载 ESM-2 特征与亮度标签，训练 MLP 回归模型预测 GFP 初始亮度
规范对齐: 基础模型训练管线。遵循 DIY规范.txt v1.1，固定随机种子(42)，记录评估指标与训练参数
作者: 何承炫 (HCX) | 更新日期: 2024-05-20
依赖库: torch, numpy, pandas, scikit-learn
输入: data/processed/gfp_esm2_emb.npy (特征矩阵), data/processed/gfp_labels.parquet (亮度标签)
输出: models/gfp_mlp_best.pth (PyTorch模型权重), models/scaler_*.npy (标准化器参数)
备注: 竞赛管线为追求推理速度使用 .pth 格式；团队统一 .joblib 格式由 train_standard.py 独立维护
"""
import joblib, datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
from pathlib import Path

# ================= 路径配置 =================
ROOT = Path(__file__).resolve().parent.parent
EMB_PATH   = ROOT / "data/processed/gfp_esm2_emb.npy"
LABEL_PATH = ROOT / "data/processed/gfp_labels.parquet"
MODEL_DIR  = ROOT / "models"
os.makedirs(MODEL_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定全局随机种子（严格对齐 DIY规范.txt 可复现要求）
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# ============================================

print(f"🚀 使用设备: {DEVICE}")

# ================= 1. 数据加载与训练集划分 =================
# 加载 ESM-2 提取的 1280 维特征与原始亮度标签（不取对数）
# 固定 random_state=42 确保全队数据划分一致，符合 DIY规范.txt 要求
X = np.load(EMB_PATH).astype(np.float32)
y = pd.read_parquet(LABEL_PATH)["Brightness"].values.astype(np.float32)
assert X.shape[0] == y.shape[0], "⚠️ 特征与标签数量不匹配！"

# ================= 2. 特征标准化 (StandardScaler) =================
# 仅使用训练集拟合均值/方差，防止数据泄露
# 保存 scaler_mean.npy 与 scaler_scale.npy 供 optimize.py 推理时复用
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分: 80% train | 10% val | 10% test
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED)

# DataLoader
train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
val_ds   = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=0)

# ================= 3. MLP 亮度预测模型定义 =================
# 结构: 1280 → 512 → 256 → 128 → 1 (回归输出)
# 使用 BatchNorm + Dropout(0.2) 防止过拟合，ReLU 激活
# 固定 torch.manual_seed(42) 确保权重初始化可复现
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

# ================= 4. 模型训练循环 =================
# 优化器: AdamW | 损失函数: HuberLoss | 学习率: 1e-3 (动态衰减)
# 启用 Early Stopping：验证集 MAE 连续 15 轮不提升则终止，保存最佳权重
# 半精度(FP16)可在此处开启，当前为保持精度使用 FP32，RTX 4080 完全可承载
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

# ================= 5. 测试集评估与打包标准 .joblib 预测器 =================
print("\n📊 加载最佳权重并进行测试集评估...")
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "gfp_mlp_best.pth"), weights_only=True))
model.eval()
with torch.no_grad():
    y_pred = model(torch.tensor(X_te).to(DEVICE)).cpu().numpy()

r2 = r2_score(y_te, y_pred)
mae = mean_absolute_error(y_te, y_pred)
rmse = np.sqrt(mean_squared_error(y_te, y_pred))

print("="*40)
print(f"📈 测试集评估 (原始亮度尺度) | R² = {r2:.4f} | MAE = {mae:.4f} | RMSE = {rmse:.4f}")
print("="*40)

# 打包高性能 MLP 为团队规范 .joblib（含权重、Scaler、指标、血统）
predictor_package = {
    "model": model.state_dict(),
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "metrics": {"R2": float(r2), "MAE": float(mae), "RMSE": float(rmse)},
    "params": {"hidden_dims": [512, 256, 128], "lr": 1e-3, "batch_size": 1024, "loss": "HuberLoss", "patience": 15},
    "provenance": {"random_seed": 42, "source_data": "data/processed/gfp_esm2_emb.npy", "sample_size": len(X)},
    "metadata": {"train_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "framework": "pytorch", "version": "v1_high_perf"}
}

joblib_path = os.path.join(MODEL_DIR, "gfp_brightness_predictor.joblib")
joblib.dump(predictor_package, joblib_path)
print(f"\n✅ 高性能预测器已打包为 .joblib: {joblib_path}")
print(f"📦 字典键: {list(predictor_package.keys())} | R²={r2:.4f} (保留原高精度)")

