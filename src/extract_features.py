# src/extract_features.py
"""
功能：使用 ESM2-150M 提取 full_sequences.csv 的序列特征，输出标准化 .joblib 数据集
规范对齐：固定采样5000条(random_state=42) / 原始亮度不取对数 / int8序列编码 / 统一命名
作者：何承炫 | 日期：2024-05-20
"""
import os, joblib, datetime
import numpy as np
import pandas as pd
import torch
from transformers import EsmTokenizer, EsmModel
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data/full_sequences.csv"
OUT_DIR  = ROOT / "data/Feats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 固定参数
SAMPLE_N = 5000
SEED = 42
AUTHOR = "HCX"
DATE_STR = datetime.datetime.now().strftime("%y%m%d")
MODEL_NAME = "esm2_t30_150M_UR50D"
MODEL_PATH = ROOT / "esm2_150M_model"  # 需提前下载该模型权重

# 氨基酸整数编码映射（严格按规范）
AA_MAP = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,
          'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}

def encode_seqs(seqs):
    """序列转 int8 矩阵，短序列末尾补 0"""
    max_len = max(len(s) for s in seqs)
    mat = np.zeros((len(seqs), max_len), dtype=np.int8)
    for i, s in enumerate(seqs):
        mat[i, :len(s)] = [AA_MAP.get(aa, 0) for aa in s]
    return mat

print("📂 加载数据并固定采样...")
df_all = pd.read_csv(CSV_PATH)
df = df_all.sample(n=SAMPLE_N, random_state=SEED).reset_index(drop=True)
print(f"✅ 采样完成: {len(df)} 条 | 首条序列预览: {df['full_sequence'].iloc[0][:10]}...")

print("⬇️ 加载 ESM2-150M...")
tokenizer = EsmTokenizer.from_pretrained(MODEL_PATH)
model = EsmModel.from_pretrained(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu").eval()

print("⏳ 提取特征向量 (X)...")
X_list = []
with torch.no_grad():
    for i in range(0, len(df), 32):
        batch = df["full_sequence"].iloc[i:i+32].tolist()
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=260, return_tensors="pt").to(model.device)
        out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        X_list.append(pooled.cpu().float().numpy())
X = np.vstack(X_list).astype(np.float32)

Y = df["brightness"].values.astype(np.float32)  # 原始亮度，不取对数
seqs = encode_seqs(df["full_sequence"].tolist())

metadata = {
    "extract_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_version": MODEL_NAME,
    "esm_layers": 30,
    "sample_size": SAMPLE_N,
    "random_seed": SEED
}

out_name = f"GFP_Feat_{MODEL_NAME}_{AUTHOR}_{DATE_STR}_v1.joblib"
out_path = OUT_DIR / out_name
joblib.dump({"X": X, "Y": Y, "seqs": seqs, "metadata": metadata}, out_path)
print(f"✅ 特征已保存: {out_path} | X.shape={X.shape}, Y.shape={Y.shape}, seqs.shape={seqs.shape}")
