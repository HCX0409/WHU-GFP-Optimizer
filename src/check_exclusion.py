# check_exclusion.py
import pandas as pd
import os
import re

# ================= 配置区 =================
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
EXCLUSION_CSV = ROOT / "data/external/Exclusion_List.csv"
CANDIDATE_CSV = ROOT / "data/GFP_Competition_Submission.csv"
# 方式2：手动粘贴单条序列（取消下方注释并替换内容即可）
# MANUAL_SEQ = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
# ==========================================

def clean_seq(s):
    """严格清洗：去空格/换行/转大写/仅保留氨基酸字母"""
    if pd.isna(s): return ""
    return re.sub(r'[^A-Z]', '', str(s).upper())

def load_exclusion_list(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 找不到排除列表: {path}")
    
    df = pd.read_csv(path)
    # 自动识别序列列
    seq_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ['seq', 'aa', 'protein', 'sequence', '突变']):
            seq_col = col
            break
    if seq_col is None:
        seq_col = df.columns[0]
        print(f"⚠️ 未识别到序列列名，默认使用第一列: '{seq_col}'")
        print(f"📋 可用列名: {list(df.columns)}")
    
    # 清洗并转为集合（O(1) 查找）
    excl_set = set(df[seq_col].dropna().apply(clean_seq))
    excl_set = {s for s in excl_set if len(s) > 50}  # 过滤非序列脏数据
    print(f"📂 已加载排除列表: {len(excl_set)} 条有效序列")
    return excl_set

def check_candidates(excl_set, candidates, source_name="手动输入"):
    print(f"\n🔍 开始检查 ({source_name})...")
    print("-" * 50)
    safe_seqs = []
    for i, seq in enumerate(candidates, 1):
        c_seq = clean_seq(seq)
        if not c_seq: continue
        if c_seq in excl_set:
            print(f"🚫 候选 {i} 命中排除列表！(完全匹配)")
        else:
            print(f"✅ 候选 {i} 安全，可提交。")
            safe_seqs.append(seq)
    print("-" * 50)
    return safe_seqs

if __name__ == "__main__":
    excl_set = load_exclusion_list(EXCLUSION_CSV)
    
    # 优先读取比赛提交CSV
    if os.path.exists(CANDIDATE_CSV):
        df_cand = pd.read_csv(CANDIDATE_CSV)
        seq_col_cand = "Sequence" if "Sequence" in df_cand.columns else df_cand.columns[0]
        candidates = df_cand[seq_col_cand].dropna().tolist()
        safe = check_candidates(excl_set, candidates, source_name=CANDIDATE_CSV)
    else:
        print(f"⚠️ 未找到 {CANDIDATE_CSV}，请手动在脚本顶部设置 MANUAL_SEQ")
        
    # 若需检查手动序列，取消下方注释：
    # if 'MANUAL_SEQ' in dir():
    #     check_candidates(excl_set, [MANUAL_SEQ], source_name="手动输入")
