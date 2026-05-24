# src/rank.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import EsmTokenizer, EsmModel
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_CSV = ROOT / "data/GFP_Optimized_Candidates.csv"
ESM_MODEL_DIR  = ROOT / "esm2_model"
OUT_CSV        = ROOT / "data/GFP_Competition_Submission.csv"
WT_SEQ         = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not CANDIDATES_CSV.exists():
    raise FileNotFoundError(f"❌ 找不到候选文件: {CANDIDATES_CSV}\n请先运行 optimize 步骤。")

print("📂 加载候选序列与 ESM-2...")
df = pd.read_csv(CANDIDATES_CSV)
tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_DIR)
esm = EsmModel.from_pretrained(ESM_MODEL_DIR).to(DEVICE).eval()

THERMO_BONUS_MUTS = {"F64L", "S65T", "V163A", "S205T", "T203Y", "Q80R", "E111V", "N146I", "M153T", "S175G", "T203V", "A206V"}
def thermo_bonus(mut_str):
    if pd.isna(mut_str) or mut_str == "WT": return 0
    return len(set(str(mut_str).split(":")) & THERMO_BONUS_MUTS) * 0.15

def esm_stability_score(seq):
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(DEVICE)
    with torch.no_grad():
        out = esm(**inputs)
        log_probs = torch.log_softmax(out.last_hidden_state @ esm.embeddings.word_embeddings.weight.T, dim=-1)
        token_ids = inputs["input_ids"][0]
        return log_probs[0, torch.arange(len(token_ids)), token_ids].mean().item()

print("⏳ 计算稳定性得分与综合排序...")
wt_logp = esm_stability_score(WT_SEQ)
scores = []
for _, row in df.iterrows():
    seq = row["Sequence"]
    muts = row.get("Mutations", "WT")
    stab = esm_stability_score(seq) - wt_logp
    bonus = thermo_bonus(muts)
    final_score = 0.6 * row["Pred_Log10"] + 0.3 * stab + 0.1 * bonus
    scores.append({
        "Mutations": muts, "Pred_Log10": row["Pred_Log10"], "Pred_Linear": row["Pred_Linear"],
        "ESM_Stability_dLogP": stab, "Thermo_Bonus": bonus, "Competition_Score": final_score, "Sequence": seq
    })

res = pd.DataFrame(scores).sort_values("Competition_Score", ascending=False).reset_index(drop=True)
res.to_csv(OUT_CSV, index=False)
print(f"✅ 比赛提交排序完成！已保存: {OUT_CSV}")
print("\n📊 Top 5 提交建议:")
print(res[["Mutations","Pred_Log10","ESM_Stability_dLogP","Competition_Score"]].head().to_string(index=False))
