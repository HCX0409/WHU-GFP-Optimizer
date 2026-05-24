# 竞赛推理权重目录
本目录存放竞赛管线专用模型与标准化参数，供 `optimize.py` 高速加载。
- `gfp_mlp_best.pth`: PyTorch MLP 亮度预测模型（Early Stopping 最佳权重）
- `scaler_mean.npy` / `scaler_scale.npy`: StandardScaler 参数（推理时特征对齐必需）
- 注：团队统一 `.joblib` 格式模型由 `src/train_standard.py` 生成至 `data/Models/`，两者职责隔离。
