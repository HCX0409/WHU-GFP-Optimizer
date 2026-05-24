# 🧬 GFP Brightness Optimizer
基于 ESM-2 与深度学习的 GFP 亮度预测与定向进化管线。

## 📦 环境配置
```bash
conda create -n gfp_opt python=3.10 -y
conda activate gfp_opt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
set HF_ENDPOINT=https://hf-mirror.com
```

## 🚀 使用方式
```bash
python main.py --step all
python main.py --step check
```

## 📜 许可
MIT License
