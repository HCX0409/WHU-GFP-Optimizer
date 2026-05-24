# gfp_preprocess.py (已针对 RTX 4080 优化)
import pandas as pd
import re
import torch
from torch.utils.data import DataLoader
from transformers import EsmTokenizer, EsmModel
from tqdm import tqdm
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ================= 配置区 =================
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH     = ROOT / "data/raw/GFP_data.xlsx"
OUT_DIR       = ROOT / "data/processed"
ESM_MODEL_DIR = ROOT / "esm2_model"  # 原 MODEL_NAME 已更名
BATCH_SIZE = 64          # RTX 4080(16G) 推荐 64，吞吐量最大
POS_OFFSET = 1           # 数据集编号偏移校正
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================

os.makedirs(OUT_DIR, exist_ok=True)
print(f"🚀 使用设备: {DEVICE} | 输出目录: {OUT_DIR}")
assert len(WT_SEQ) == 238, "WT序列长度必须为238"

def mut_to_seq(mut_str):
    if pd.isna(mut_str) or str(mut_str).strip().upper() in ["WT", ""]:
        return WT_SEQ
    seq_list = list(WT_SEQ)
    for m in str(mut_str).split(":"):
        match = re.match(r"^([A-Z])(\d+)([A-Z])$", m.strip())
        if not match: continue
        orig, pos_str, new = match.groups()
        idx = int(pos_str) + POS_OFFSET - 1
        if 0 <= idx < 238:
            seq_list[idx] = new
    return "".join(seq_list)

print("📖 正在读取 Excel...")
df = pd.read_excel(XLSX_PATH, engine="openpyxl")
print(f"📊 原始行数: {len(df)}")

print("🧬 还原完整序列并去重...")
df["sequence"] = df["aaMutations"].apply(mut_to_seq)
df = df[df["sequence"].str.len() == 238].drop_duplicates(subset=["sequence"]).reset_index(drop=True)
print(f"✅ 去重后有效序列数: {len(df)}")

df[["Brightness"]].to_parquet(os.path.join(OUT_DIR, "gfp_labels.parquet"), index=False)

print("⬇️ 加载 ESM-2 模型（首次下载约1.2GB）...")
tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

sequences = df["sequence"].tolist()
def collate_fn(batch_seqs):
    return tokenizer(batch_seqs, padding=True, truncation=True, max_length=256, return_tensors="pt")

dataloader = DataLoader(sequences, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False, num_workers=0)

all_emb = []
print("⏳ 开始批量提取表征 (RTX 4080 预计 10~15 分钟)...")
# 开启半精度加速，大幅降低显存占用并提升吞吐
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    for batch in tqdm(dataloader, desc="ESM-2 Batches", unit="batch"):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        hidden = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_emb.append(pooled.cpu().float().numpy())

embeddings = np.vstack(all_emb)
np.save(os.path.join(OUT_DIR, "gfp_esm2_emb.npy"), embeddings)
print(f"✅ 完成！特征维度: {embeddings.shape}")
print(f"💾 已保存至: {OUT_DIR}")
