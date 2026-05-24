# 🧬 WHU-GFP-Optimizer
基于 ESM-2 + SaProt 双编码器的 GFP 亮度预测与热稳定性定向进化管线。  
**对齐赛事**：SynBio Challenges GFP 赛道（全序列开放 / 72℃热复性双能乘积评分）  
**团队规范**：严格遵循 `DIY规范.txt`，固定随机种子(42)，统一特征/模型命名与数据血统记录。

## 📦 环境配置
```bash
conda create -n gfp_esm python=3.10 -y
conda activate gfp_esm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# (可选) 加速 HuggingFace 下载
set HF_ENDPOINT=https://hf-mirror.com

📥 模型权重下载（首次运行必需）
为保持仓库轻量合规，大模型权重已通过 .gitignore 排除。请手动下载以下文件至对应目录：

模型	存放目录	必需文件	下载链接
ESM-2 650M	esm2_model/	config.json, pytorch_model.bin, vocab.txt, tokenizer_config.json, special_tokens_map.json	HuggingFace
SaProt 650M	saprot_model/	同上	HuggingFace

🚀 使用方式
管线已模块化，建议按顺序执行（SaProt 已内嵌，无需额外命令）：

python main.py --step preprocess   # 1. 数据清洗 + ESM-2 特征提取
python main.py --step train        # 2. 训练亮度预测 MLP (输出 .pth)
python main.py --step optimize     # 3. 🆕 遗传算法优化 (自动加载 SaProt 热稳定代理，双能乘积适应度)
python main.py --step rank         # 4. 多目标排序 + 比赛硬约束过滤 (生成 Top 6 CSV)
python main.py --step check        # 5. 校验 Exclusion_List，确保 100% 不重复
