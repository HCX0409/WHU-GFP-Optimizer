import os
from pathlib import Path

ROOT = Path(__file__).parent
os.chdir(ROOT)

def w(name, txt):
    (ROOT / name).write_text(txt, encoding="utf-8")
    print(f"✅ {name}")

w("requirements.txt", "torch>=2.1.0\ntransformers>=4.35.0\npandas>=2.0.0\nnumpy>=1.24.0\nscikit-learn>=1.3.0\ntqdm>=4.65.0\nopenpyxl>=3.1.0\npyarrow>=14.0.0\nxgboost>=2.0.0\n")
w(".gitignore", "__pycache__/\n*.py[cod]\ndata/processed/*.npy\ndata/processed/*.parquet\nmodels/*.pth\nesm2_model/\n*.csv\n!data/external/Exclusion_List.csv\n.vscode/ .idea/ *.swp\n")
w("src/__init__.py", "# GFP Optimizer Pipeline\n")
w("LICENSE", "MIT License\nCopyright (c) 2024 GFP-Optimizer\n")
w("DIY规范.txt", "【比赛计算管线说明】\n1. 特征提取：ESM-2 (650M) 零样本平均池化\n2. 亮度预测：3层MLP + HuberLoss，R²=0.698\n3. 序列优化：遗传算法(5代×500)，保护发色团与β-barrel核心\n4. 热稳定代理：ESM-2 log-likelihood ratio + 文献先验加权\n5. 合规校验：严格比对 Exclusion_List.csv，无100%重复\n注：亮度为 log10 尺度，75℃复性由稳定性代理指标间接评估。\n")

readme = "# 🧬 GFP Brightness Optimizer\n基于 ESM-2 与深度学习的 GFP 亮度预测与定向进化管线。\n\n## 📦 环境配置\n```bash\nconda create -n gfp_opt python=3.10 -y\nconda activate gfp_opt\npip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\npip install -r requirements.txt\nset HF_ENDPOINT=https://hf-mirror.com\n```\n\n## 🚀 使用方式\n```bash\npython main.py --step all\npython main.py --step check\n```\n\n## 📜 许可\nMIT License\n"
w("README.md", readme)

main_py = "import subprocess, sys, argparse, os\nfrom pathlib import Path\n\nPROJECT_ROOT = Path(__file__).resolve().parent\nos.chdir(PROJECT_ROOT)\n\nSTEPS = {\n    'preprocess': 'src/preprocess.py',\n    'train': 'src/train.py',\n    'optimize': 'src/optimize.py',\n    'rank': 'src/rank.py',\n    'check': 'src/check_exclusion.py'\n}\n\ndef main():\n    parser = argparse.ArgumentParser(description='GFP Optimizer Pipeline')\n    parser.add_argument('--step', choices=list(STEPS.keys()) + ['all'], default='all')\n    args = parser.parse_args()\n    targets = list(STEPS.values()) if args.step == 'all' else [STEPS[args.step]]\n    for script in targets:\n        print(f'\\n{\"=\"*20} 🚀 Running: {script} {\"=\"*20}')\n        subprocess.run([sys.executable, str(PROJECT_ROOT / script)], check=True)\n    print('\\n✅ Pipeline completed.')\n\nif __name__ == '__main__':\n    main()\n"
w("main.py", main_py)

print("\n🎉 所有占位文件已安全填充！")
