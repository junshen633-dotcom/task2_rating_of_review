# COMP5434 Task 2 — Review Rating Prediction

Apache Spark 负责清洗、分组划分和 OOF 统计特征，自研 PyTorch HashTextCNN 预测 1–5 星评分。全量数据预处理、E1–E4 对照、seed 43 复验及独立评估已完成。首轮最终模型为 **E4_seed43**，观察评分校准版独立评估 **RMSE 1.143588 / MAE 0.862771**。

用户提供的榜单截图（2026-11-13）显示：同一 E4_seed43 的 `latent_hypothesis.csv` 为 **0.54546**，`observed_calibrated.csv` 为 **0.70392**。这是用户提交的结果，自动实验脚本未执行 Kaggle 提交。模型使用 80% 训练部分，未进行 100% 标签重训练。旧实验采用不同划分，不能直接比较本地 RMSE。

## 2026-11-13 后续实验已完成

推荐新提交：[recommended.csv](results/extended_delivery_20261113/recommended.csv)，即 E3/E4 × seed42/43 四模型的潜在评分等权平均。完整对比见 [交付说明](results/extended_delivery_20261113/README_CN.md)。这批新文件尚无 Kaggle 成绩。

| 候选 | cal 观察 RMSE | 条件修正潜在 MSE |
|---|---:|---:|
| 原 E4_seed43 | 1.141661 | 0.303465 |
| E4 两 seed 集成 | 1.138927 | 0.292310 |
| **E3/E4 四模型集成** | **1.138579** | **0.290892** |
| 八轮实验的最佳单模型（第4轮） | 1.140836 | 0.300102 |
| 最佳长训练模型 + 旧 E4 seed42 | 1.138811 | 0.291829 |

重新训练8轮，前三轮的训练损失和cal RMSE完全复现旧实验；第4轮最佳，第5–8轮训练损失下降但校准误差回升。新增计算预算的收益小于固定集成；增加轮数不是当前主要突破口。

选择冻结后，四模型集成在已有 eval 基准上的观察 RMSE=1.140729、条件修正潜在 MSE=0.298373；原 E4_seed43 为1.143794和0.310906。该 eval 以前已使用，不称新盲测；没有据此再选择候选。修正 MSE 依赖噪声矩阵假设，不是已测得的干净标签误差。

HPC 作业22291（55m10s）和22293（3m26s）均 COMPLETED / exit0，错误日志为空。测试38通过、1跳过；每份交付CSV均验证10000个唯一ID、模板顺序和评分范围。逐轮checkpoint保留在HPC，本地交付包括预测、分析和训练记录。

在完成首轮实验、已有四个模型和smoke_prepared后，从仓库根目录运行：

```bash
# 沿用上节/环境说明中的 TASK2_CODE、TASK2_RAW、TASK2_WORK、TASK2_ENV
jid=$(sbatch --parsable hpc/extended.sh)
sbatch --dependency=afterok:$jid hpc/finalize_extended.sh
# 新输出目录不得包含旧结果；重跑时请归档或使用新的工作目录。
```

## 实验结果

| 实验 | 目标函数 | OOF 元数据 | Seed | 校准集 RMSE |
|---|---|---|---|---|
| E1 | MSE | 无 | 42 | 1.151117 |
| E2 | 普通 CE | 无 | 42 | 1.147978 |
| E3 | 前向噪声修正 CE | 无 | 42 | 1.143010 |
| E4 | 前向噪声修正 CE | 有 | 42 | 1.141971 |
| E3_seed43 | 前向噪声修正 CE | 无 | 43 | 1.143799 |
| E4_seed43 | 前向噪声修正 CE | 有 | 43 | 1.141661 |

每组最多 3 epochs，最佳均为第 3 轮；仅依据 cal 选择模型及拟合收缩。最终冻结模型后评估 eval。

| 预测文件 | 含义 | 独立 eval RMSE | MAE |
|---|---|---|---|
| `observed_raw.csv` | 噪声转移后的观察评分期望 | 1.143794 | 0.856041 |
| `observed_calibrated.csv` | raw 向训练均值轻微收缩，系数由 cal 拟合 | **1.143588** | 0.862771 |
| `latent_hypothesis.csv` | 假设噪声转移前的潜在评分期望 | 1.227962 | **0.666336** |

三份文件均为 10,000 行，按 sampleSubmission 的 id 对齐。本地观察标签 RMSE 偏好 `observed_calibrated.csv`，而用户实际榜单支持 `latent_hypothesis.csv`。这一排序反转支持目标标签机制存在差异，但不证明全部测试标签干净或 Private LB 一定同样改善。标题星级也是输入，标题代理准确率不能充当独立干净标签准确率。后续固定集成与八轮实验协议见 [后续实验](docs/EXTENDED_EXPERIMENTS_CN.md)。

## 仓库内容

- `prepare.py`：严格 CSV 解析、指纹分组划分、5 折 OOF、Parquet 和训练集噪声矩阵估计。
- `model.py` / `train.py`：自研文本编码、融合、损失、流式数据加载和训练循环。
- `postprocess.py`：cal-only 收缩、独立评估和按 id 导出。
- `audit_data.py` / `compare_calibration.py`：数据诊断和校准集比较。
- `hpc/`：共用环境配置及 prepare / smoke / train / final 作业。
- `tests/`：模型、流式读取、交叉拟合与后处理测试。
- `results/`：统计证据、最终模型权重、元数据、指标及三份预测。
- [实验设计](docs/EXPERIMENT_DESIGN_CN.md)、[旧实现审计](docs/AUDIT_CN.md)、[运行记录](docs/RUNS_CN.md)。

原始数据、抽样评论、Parquet 中间产物、环境、认证信息及缓存不纳入 Git。最终 `best.pt` 约 34 MB，随本次仓库交付；校验和见 `results/SHA256SUMS`。

## HPC 环境

以下命令在 PolyU HPC 上执行。Python 3.11、Java 17、Spark 3.5.7、PyTorch 2.5.1 CUDA 12.1 是已运行组合；不依赖外部预训练权重。

```bash
module load anaconda3
source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda create -n task2_v2 -c conda-forge python=3.11 openjdk=17 pip -y
conda activate task2_v2
export PYTHONNOUSERSITE=1
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

将原始 `train.csv`、`test.csv`、`sampleSubmission.csv` 放在个人 scratch 数据目录。每次 sbatch 从仓库根目录提交；脚本通过 SLURM_SUBMIT_DIR 定位代码。

```bash
export TASK2_CODE="$PWD"
export TASK2_RAW="/scratch/$USER/data"
export TASK2_WORK="/scratch/$USER/task2_v2"
export TASK2_ENV=task2_v2
mkdir -p "$TASK2_WORK"
module load slurm
sbatch hpc/smoke.sh
# squeue -u "$USER" 查看状态；确认日志 TASK2_SMOKE_ALL_PASSED 后继续。
sbatch hpc/prepare.sh
```

确认预处理 COMPLETED / exit 0 后，依次运行以下训练；每个完成再提交下一个，避免资源配额冲突。重跑请使用新的 TASK2_WORK 或新实验名。

```bash
sbatch hpc/train.sh mse E1
sbatch hpc/train.sh ce E2
sbatch hpc/train.sh forward E3
# 本次 E1–E3 的 cal 胜出为 forward；新数据需重新比较。
sbatch hpc/train.sh forward E4 --metadata
sbatch hpc/train.sh forward E3_seed43 --seed 43
sbatch hpc/train.sh forward E4_seed43 --metadata --seed 43
# 所有候选完成、按 cal 冻结配置后才运行最终独立评估。
sbatch hpc/final.sh E4_seed43
```

结果在 `$TASK2_WORK/submission_E4_seed43/`。`final.sh` 读取本次训练输出的 checkpoint；若使用仓库附带权重，必须使用完全匹配的 prepared/metadata.json，训练程序会校验其 SHA256，不能与新生成的元数据混用。

`hpc/env.sh` 隔离用户 site-packages，清除遗留 Spark/Python 环境变量并设置 conda Java，修复此前 JAVA_GATEWAY_EXITED 环境冲突。若仍报错，查看 SLURM 日志中 gateway 异常前的 Java stderr。脚本不覆盖 SLURM 的 CUDA_VISIBLE_DEVICES。资源为单节点 Spark local[8] 和单 A800 MIG 20GB；不是多机训练。平台说明：[HPC how-to](https://hpc.comp.polyu.edu.hk/howto/#make-gpus-visible-in-the-slurm-job)。

## 验证与局限

原 HPC 执行中 26 项测试通过（含 Spark 集成）；修复 CUDA 无索引设备选择后 smoke 训练、推理和导出通过。整理后的验证记录见 [VALIDATION.txt](VALIDATION.txt)。GPU 实验没有因本次仓库整理重新训练。

划分 train=6,150,112 / cal=766,812 / eval=767,721，按不含标签的真实评论指纹分组；test=10,000。统计先验也排除本折。噪声矩阵来自训练集标题代理，噪声机制仍是假设。最终测试推理约 0.767 秒仅包含已准备 Parquet 的读取、分词、模型及写出，不包含原始 CSV 的 Spark 预处理。
