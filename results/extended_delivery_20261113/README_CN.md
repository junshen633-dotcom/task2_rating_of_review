# 扩展实验提交文件

首选文件：`recommended.csv`，对应 **E34_four_model**。

推荐完全依据 cal 上的噪声修正潜在 MSE，候选权重预先固定为等权；没有使用 eval 或榜单分数选择。
修正 MSE 是依赖噪声矩阵正确性的条件估计，不是已认证的干净标签误差；没有将它开平方冒充 RMSE。

| 候选 | cal 观察 RMSE | 条件修正潜在 MSE | 文件 |
|---|---:|---:|---|
| E4_seed43 | 1.14166058 | 0.30346514 | `baseline_e4_seed43_latent.csv` |
| E4_two_seed | 1.13892731 | 0.29231017 | `e4_two_seed_latent.csv` |
| E34_four_model | 1.13857930 | 0.29089230 | `e34_four_model_latent.csv` |
| E4_long8_single | 1.14083647 | 0.30010171 | `e4_long8_single_latent.csv` |
| E4_long8_plus_seed42 | 1.13881073 | 0.29182949 | `e4_long8_plus_seed42_latent.csv` |

长训练完成 8 轮；观察 RMSE 最佳在第 4 轮，修正潜在 MSE 最佳在第 4 轮。
前三轮相对原 E4_seed43 的 cal RMSE 最大绝对差：0。

每份预测包含 10,000 条记录，已按 sampleSubmission 的 ID 对齐，并验证唯一性与评分 [1,5] 范围。
`selection.json` 包含选择规则、全部比较、配对误差区间与训练记录；`analyses/` 保存各候选原始诊断。
配对区间是固定噪声矩阵下按独立行近似的结果，未覆盖噪声矩阵估计、分组相关及模型选择带来的不确定性。
已有基线文件保留为 `baseline_e4_seed43_latent.csv`；新候选不保证在 Kaggle 上优于基线。
`SHA256SUMS` 可用于核对所有交付文件。

冻结候选的既有 eval 基准复核（不是新盲测，不用于再选模型）：

| 模型 | 观察 RMSE | 条件修正潜在 MSE |
|---|---:|---:|
| 原 E4_seed43 | 1.14379404 | 0.31090560 |
| E34_four_model | 1.14072880 | 0.29837256 |
