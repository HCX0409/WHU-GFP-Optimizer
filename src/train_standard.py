# src/train_standard.py
"""
功能：加载标准化 .joblib 特征，训练 RF 或 MLP 亮度预测模型，输出合规 .joblib 模型文件
规范对齐：固定种子42 / 统一字典键名 / 记录血统与元数据 / 统一命名
作者：何承炫 | 日期：2024-05-20
"""
import os, joblib, datetime, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FEAT_DIR = ROOT / "data/Feats"
MODEL_DIR = ROOT / "data/Models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

AUTHOR = "HCX"
DATE_STR = datetime.datetime.now().strftime("%y%m%d")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 加载最新特征文件
feat_files = sorted(FEAT_DIR.glob("GFP_Feat_*.joblib"))
if not feat_files: raise FileNotFoundError("❌ 未找到特征文件，请先运行 extract_features.py")
feat_path = feat_files[-1]
data = joblib.load(feat_path)
X, Y = data["X"], data["Y"]
print(f"📂 加载特征: {feat_path.name} | X:{X.shape}, Y:{Y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED)

# ================= 选择模型 =================
USE_RF = True  # True=随机森林, False=PyTorch MLP
if USE_RF:
    model_name = "RF"
    params = {"n_estimators": 300, "max_depth": 15, "random_state": SEED}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
else:
    model_name = "MLP"
    params = {"hidden_dims": [512, 256, 128], "lr": 1e-3, "epochs": 50, "batch_size": 64}
    class MLP(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)
            )
        def forward(self, x): return self.net(x).squeeze(-1)
    model = MLP(X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dl = DataLoader(ds, batch_size=params["batch_size"], shuffle=True)
    model.train()
    for ep in range(params["epochs"]):
        for xb, yb in dl:
            opt.zero_grad()
            loss = nn.MSELoss()(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()

# 测评 (核心任务3)
if USE_RF:
    y_pred = model.predict(X_test)
else:
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test)).numpy()

metrics = {
    "R2": r2_score(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
}
print(f"📊 测评结果: R²={metrics['R2']:.4f} | MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f}")

# 打包合规字典
emb_model = feat_path.stem.split("_")[2]  # 自动提取嵌入模型名
out_name = f"GFP_{model_name}_{emb_model}_{AUTHOR}_{DATE_STR}_v1.joblib"
out_path = MODEL_DIR / out_name

package = {
    "model": model,
    "metrics": metrics,
    "params": params,
    "provenance": {
        "random_seed": SEED,
        "feat_dim": X.shape[1],
        "source_dataset": feat_path.name,
        "sample_size": len(X)
    },
    "metadata": {
        "train_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": 0,  # 可后续补充
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "python_libs": {"sklearn": "1.3+", "torch": "2.1+"}
    }
}
joblib.dump(package, out_path)
print(f"✅ 模型已保存: {out_path}")
