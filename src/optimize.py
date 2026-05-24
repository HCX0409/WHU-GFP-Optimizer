# src/optimize.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os, re, random
from pathlib import Path
from transformers import EsmTokenizer, EsmModel, AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
# warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR     = ROOT / "models"
ESM_MODEL_DIR = ROOT / "esm2_model"
SAPROT_LOCAL  = ROOT / "saprot_model"
OUT_CSV       = ROOT / "data/GFP_Optimized_Candidates.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WT_SEQ = "MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNIVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
CHAMPION_SEQS = {
    "sfGFP": WT_SEQ,
    "mBaoJin": "MVSKGEEENMASTPFKFQLKGTINGKSFTVEGEGEGNSHEGSHKGKYVCTSGKLPMSWAALGTTFGYGMKYYTKYPSGLKNWFREVMPGGFTYDRHIQYKGDGSIHAKHQHFMKNGTYHNIVEFTGQDFKENSPVLTGDMNVSLPNEVPQIPRDDGVECPVTLLYPLLSDSKYVEAHQYTICKPLHNQPAPDVPYHWIRKQYTQSKDDAEERDHICQSETLEAHLKGMDLYK",
    "TGP": "MVSKGEEENMASTPFKFQLKGTINGKSFTVEGEGEGNSHEGSHKGKYVCTSGKLPMSWAALGTTFGYGMKYYTKYPSGLKNWFREVMPGGFTYDRHIQYKGDGSIHAKHQHFMKNGTYHNIVEFTGQDFKENSPVLTGDMNVSLPNEVPQIPRDDGVECPVTLLYPLLSDSKYVEAHQYTICKPLHNQPAPDVPYHWIRKQYTQSKDDAEERDHICQSETLEAHLKGMDLYK"
}
CHAMPION_WEIGHTS = {"sfGFP": 0.3, "mBaoJin": 0.4, "TGP": 0.3}

POP_SIZE, ELITE_SIZE, GENERATIONS = 500, 50, 5
MUT_RATE, MAX_MUTS = 0.6, 3
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
PROTECTED = set([64,65,66, 10,22,34,45,58,78,90,102,115,128,140,152,165,178,190,202,227])
MUTABLE_POS = [i for i in range(len(WT_SEQ)) if i not in PROTECTED]

print("⬇️ 加载 ESM-2 (亮度) + SaProt 650M (热稳定代理)...")
esm_tok = EsmTokenizer.from_pretrained(ESM_MODEL_DIR)
esm_model = EsmModel.from_pretrained(ESM_MODEL_DIR).to(DEVICE).eval()

saprot_tok = AutoTokenizer.from_pretrained(SAPROT_LOCAL, trust_remote_code=True)
saprot_model = AutoModel.from_pretrained(SAPROT_LOCAL, trust_remote_code=True).to(DEVICE).eval()

def get_anchor_embs():
    embs = {}
    for name, seq in CHAMPION_SEQS.items():
        inputs = saprot_tok(seq, return_tensors="pt", add_special_tokens=True).to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = saprot_model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(1e-9)
            embs[name] = emb.cpu().float().numpy().flatten()
    return embs
ANCHOR_EMB = get_anchor_embs()

class BrightnessMLP(nn.Module):
    def __init__(self, in_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

mlp_model = BrightnessMLP().to(DEVICE)
mlp_model.load_state_dict(torch.load(MODEL_DIR / "gfp_mlp_best.pth", weights_only=True))
mlp_model.eval()
scaler = StandardScaler()
scaler.mean_ = np.load(MODEL_DIR / "scaler_mean.npy")
scaler.scale_ = np.load(MODEL_DIR / "scaler_scale.npy")

def predict_brightness(seqs):
    all_emb = []
    for i in range(0, len(seqs), 32):
        batch = seqs[i:i+32]
        inputs = esm_tok(batch, padding=True, truncation=True, max_length=260, return_tensors="pt").to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = esm_model(**inputs)
            hidden = out.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_emb.append(pooled.cpu().float().numpy())
    X = np.vstack(all_emb)
    X_scaled = scaler.transform(X)
    with torch.no_grad():
        pred = mlp_model(torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    return pred

def get_thermo_proxies(seqs):
    all_embs, all_folds = [], []
    for i in range(0, len(seqs), 32):
        batch = seqs[i:i+32]
        inputs = saprot_tok(batch, padding=True, truncation=True, max_length=260, return_tensors="pt").to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            out = saprot_model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = out.last_hidden_state
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(1e-9)
            all_embs.append(pooled.cpu().float().numpy())
            all_folds.append(hidden.std(dim=1).mean(dim=1).cpu().float().numpy())
    embs, folds = np.vstack(all_embs), np.concatenate(all_folds)
    bonuses = []
    for emb, f_val in zip(embs, folds):
        sim_tgp = cosine_similarity([emb], [ANCHOR_EMB["TGP"]])[0][0]
        sim_bj  = cosine_similarity([emb], [ANCHOR_EMB["mBaoJin"]])[0][0]
        sim_sf  = cosine_similarity([emb], [ANCHOR_EMB["sfGFP"]])[0][0]
        anchor_bonus = (max(0, (sim_tgp-0.78)*0.4)*CHAMPION_WEIGHTS["TGP"] +
                        max(0, (sim_bj-0.80)*0.3)*CHAMPION_WEIGHTS["mBaoJin"] +
                        max(0, (sim_sf-0.82)*0.2)*CHAMPION_WEIGHTS["sfGFP"])
        fold_bonus = max(0, (1.5 - f_val) * 0.15)
        bonuses.append(min(anchor_bonus + fold_bonus, 0.5))
    return np.array(bonuses)

def seq_to_mutations(seq):
    muts = [f"{WT_SEQ[i]}{i+1}{seq[i]}" for i in range(len(seq)) if i < len(WT_SEQ) and seq[i] != WT_SEQ[i]]
    return ":".join(muts) if muts else "WT"

def mutate_sequence(parent_seq):
    seq = list(parent_seq)
    n_muts = random.randint(1, MAX_MUTS)
    pos = random.sample(MUTABLE_POS, min(n_muts, len(MUTABLE_POS)))
    for p in pos:
        if random.random() < MUT_RATE:
            seq[p] = random.choice([a for a in AA_LIST if a != seq[p]])
    return "".join(seq)

print("🌱 初始化种群...")
population = [WT_SEQ] + [mutate_sequence(WT_SEQ) for _ in range(POP_SIZE - 1)]
history = []

for gen in range(1, GENERATIONS + 1):
    print(f"\n🧬 Generation {gen}/{GENERATIONS} | 种群大小: {len(population)}")
    print("⏳ 正在计算 ESM-2 亮度预测...")
    preds = predict_brightness(population)
    
    print("⏳ 正在计算 SaProt 热稳定代理...")
    thermo_proxies = get_thermo_proxies(population)
    
    history_gen = []
    for idx, (seq, score) in enumerate(zip(population, preds)):
        length = len(seq)
        if not (220 <= length <= 250) or score < 3.35:
            fit = -999.0
        else:
            thermo = thermo_proxies[idx]
            charge = seq.count('K')+seq.count('R')-seq.count('D')-seq.count('E')
            charge_bonus = 0.25 if charge <= -7 else (0.12 if charge <= -5 else 0.0)
            loop_seq = seq[144:150] + seq[189:200]
            loop_rigid = sum(1 for aa in loop_seq if aa in "PG") / max(len(loop_seq), 1)
            loop_bonus = loop_rigid * 0.15
            
            retention = thermo + charge_bonus + loop_bonus
            init_brightness = 10 ** score
            fit = init_brightness * retention
            
        history_gen.append({
            "Generation": gen, "Sequence": seq, "Length": len(seq),
            "Mutations": seq_to_mutations(seq), "Pred_Log10": score, "Fitness": fit
        })
    
    history.extend(history_gen)
    
    elite_idx = np.argsort([h["Fitness"] for h in history_gen])[-ELITE_SIZE:]
    elites = [population[i] for i in elite_idx]
    next_gen = elites.copy()
    while len(next_gen) < POP_SIZE:
        next_gen.append(mutate_sequence(random.choice(elites)))
    population = next_gen

df = pd.DataFrame(history).drop_duplicates(subset=["Sequence"])
df = df.sort_values("Fitness", ascending=False).reset_index(drop=True)
wt_pred = predict_brightness([WT_SEQ])[0]
df = df[df["Pred_Log10"] > wt_pred].head(50)

df.to_csv(OUT_CSV, index=False)
print(f"\n✅ 优化完成！Top 50 候选已保存至: {OUT_CSV}")
print(f"📈 WT预测亮度(log10): {wt_pred:.3f} | 最佳候选: {df.iloc[0]['Pred_Log10']:.3f} ({df.iloc[0]['Mutations']})")
