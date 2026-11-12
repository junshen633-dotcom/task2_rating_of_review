# 现有 Task 2 实现审计

审计日期：2026-11-11。对象是本机 `D:/Program Files/COMP5434/task2_rating_of_each_review/code` 下源码；没有登录 HPC，也没有核对 HPC 当前文件版本。原文件未修改。这里确认的是代码行为，不将其直接等同于全部已报告实验的真实运行版本。

## 首要结论

先修复训练特征中的目标自泄漏，再判断模型表征能力。当前代码的 holdout 标签没有直接加入统计表，但训练行使用了含自身标签的统计特征。两者不是同一件事，“先划分再聚合”不足以解决训练行的目标编码泄漏。

## 1. 已确认：训练行的统计特征包含自身标签

- `src/features_stat.py:120` 在整个训练折计算实体均值、标准差、平滑均值。
- `src/features_stat.py:124` 将同一统计表回填到训练折。
- `src/nn_prepare.py:109` 和 `:112` 对神经网络重复相同做法。
- `nn_prepare.py` 文件开头“no row leaks its own rating”的注释与实现不符。

若一个用户在训练折中只有一条评论，其 `user_mean=y`；alpha=10 时 `user_smooth=(y+10*mu)/11`。后者虽然数值被压缩，却仍可精确反解原始标签。验证行没有这种关系。这可导致模型依赖训练时的捷径，在 holdout 上失效；它是 LR 训练分数远优于验证分数的一项直接可检验解释。

修复：训练特征使用五折 cross-fitting，某折所有有标签聚合（均值、方差、频数门控及父商品回退）只来自其余四折。验证/校准/最终评估仅使用 fit 部分的统计。保留一组完全去除目标统计的文本模型，测量修复的实际贡献。不能只改为平滑系数更大。

## 2. 已确认：收缩分数在拟合收缩系数的同一数据上计算

- `src/nn_train.py:51–57` 同时用 labels 拟合 s 并计算其 RMSE。
- `src/nn_train.py:143` 又用同一 holdout 选择 epoch，`:157` 再用它拟合收缩。
- `src/shrinkage_test.py` 的 `scan_anchor` 同样复用了评分数据。

这属于开发集调参，可以报告开发集结果；不能将收缩后的结果称为独立、无偏的泛化评估。它与训练特征自泄漏的影响不同，不能混为一谈。

修复：固定 fit/dev/calibration/audit 划分。dev 选 epoch/模型，calibration 拟合后处理，audit 只在方案冻结后评价一次。若数据已经被多次查看，应如实承认 audit 并非真正未触碰的数据，并锁定此后的实验方案。

`s=0.3395` 不代表“66% 个性化部分是噪声”。s 是给定预测与锚点的回归投影系数，受偏差、尺度、相关性共同影响；`s=1` 也可能仅仅命中了 [0,1] 约束上界。分母为零时现有代码还需要显式处理。

## 3. 已确认：经典文本基线只保留 20 个词

- `conf/config.yaml:64` 为 `text_tfidf_k:20`。
- `src/features_text.py:121` 用它设置 CountVectorizer 的 `vocabSize`。
- 配置中的 `hashing_features:65536` 并未用于这条训练路径。

因此当前 GBT/LR 并没有获得通常意义上的高维词/词组 TF-IDF，不能据此判断经典文本模型已无提升空间。保留全部稀疏词组特征无需做 PCA 或将整矩阵转为 dense。可做高维稀疏文本线性基线或轻量哈希袋模型，作为神经网络结果的独立检查。

## 4. 已确认：神经网络没有编码标题全文

- `src/nn_prepare.py:61`、`:84` 只将 `comment_clean` 编成词序列。
- `conf/config.yaml:121` 将正文截断为前 50 个 token。
- `src/nn_prepare.py:45` 的 dense 列对标题仅保留正面/负面词典命中数。

“Five Stars”等强标题信号不在这两个词典内，因此正文模型基本无法使用它。新实验首先对比 comment-only 与 `[title] + [comment]`，再讨论增加 Transformer 层数。星级标题应作为输入，不能未经证据将原标签改成标题星级。

## 5. 已复现：空序列的 TextCNN 返回 -1e9 特征

- `src/nn_model.py:101–102` 对 padding 位置填 -1e9，再取最大值。
- 全部为 padding 时，所有 CNN 通道都等于 -1e9，之后送入 MLP/BatchNorm。
- 本机用原始 TextEncoder 构造全零 token 输入，返回 `[0,0,0,0,-1e9,-1e9,-1e9,-1e9,-1e9,-1e9]`，已复现。

ETL 会把空正文回退到标题（`etl.py:83`），所以“原始正文为空 366 条”不等于“NN 输入有 366 条全空”。清洗仅含 HTML/URL/空白的文本仍可能产生空输入，实际影响必须统计清洗后的全空序列数。

修复：为全空文本提供 EMPTY token；并在编码器中显式将全空池化结果置零。卷积前还应将 padding 位置的词向量与位置向量一起置零，避免 padding 的位置向量影响有效边界位置的卷积。

## 6. 已验证：位置嵌入加均值池化并不能恢复词序

`nn_model.py:86–92` 在关闭 CNN 时输出 mean(token_embedding + position_embedding)。对同长度序列，仅重排词仍得到同一个向量，位置部分只是长度相关项。本机将 `[2,3,4,0,0]` 改成 `[4,2,3,0,0]`，两输出最大浮点差为约 1.19e-7。

因此 `base → position` 的收益不能解释成“模型获得词序”；可能是长度特征等作用。TextCNN 才引入内容与局部位置的交互。报告应改用准确的机制解释。

## 7. 推理与复现风险，尚未确认发生错误

- `nn_dataset.py:26` 读 Parquet 特征后丢弃 id。
- `nn_infer.py:69–70` 独立用 pandas 再读 id，并与预测数组按位置拼接（`:84`）。同版本读同一目录通常可能保持相同顺序，但代码没有保证该契约，也没有检查 id 集合、唯一性和对齐。
- `nn_train.py:225` 丢弃返回的 shrink 指标，checkpoint 不保存收缩系数；推理由用户再次手输 `--shrink`，默认 1。
- checkpoint 保存 metadata，但 `nn_infer.py:33` 优先加载磁盘上的 metadata。词表映射没有作为 checkpoint 资产保存。

修复：Dataset 在同一次读取中保留 id，预测随 batch 输出 id，再按 sampleSubmission 的 id join/reindex；断言 10,000 个 id 唯一、无缺失、预测有限。checkpoint 同时保存词表/分词配置/特征列/数据指纹/后处理参数，推理使用 checkpoint 所属元数据。

## 8. 固定 seed 不足以证明所有实验用了相同 holdout

`src/validation.py:40` 每次对读取的 DataFrame 调用 randomSplit。若 Parquet 被重写、源文件或分区结构变化，仅 seed 相同不足以构成可审计的相同划分证据。

修复：持久化 `id, split` manifest 并记录 SHA256。逐个模型保存 holdout id/label/prediction，再用 id join 比较。无现有预测文件时，不能确认 1.366 与 1.386 对应完全相同数据、预处理和提交路径。

## 9. 计时与实验结论需要补证据

- Spark 的读取、变换是惰性执行。`pipeline.py` 对部分只构造 DataFrame 的代码块计时，实际工作可能被归入后面的 evaluate；不能直接当作独立阶段耗时。
- `nn_train.py` 没有逐 epoch 壁钟时间记录；它的 train RMSE 是 batch loss 的均值再开方，不是冻结 checkpoint 在完整训练集上的严格 RMSE。
- MAE 更低而 RMSE 更高不能单独证明“少数极端误差是树模型失败根因”。还应输出预测方差、残差分位数、按评分与实体样本数分组的 MSE、最大 1% 残差的 SSE 贡献。
- 不同数据集的论文 RMSE 不可直接与本竞赛比较；不能由 0.45 的榜首分数证明“必然存在近重复检索”。用户提供的榜单数字没有在本次审计中重新核验。
- “相同文本标签标准差为零”必须说明分组定义、只统计 n>=2 的组，并给出有冲突组比例；单例组 std=0 没有诊断意义。

## 建议修复顺序

1. 锁定解析校验、id 对齐与 split manifest。
2. 修复训练行目标自泄漏；添加无目标统计对照。
3. 编码标题全文，修复空序列与 padding。
4. 独立校准与最终评估，保存逐行预测。
5. 再比较普通 MSE 与噪声感知目标；重复检索作为可否证诊断。
6. 只有前述流程可信且收益明确时，继续更大模型或序数多任务头。

本机未找到用户总结中 Stage 3 DistilBERT、duplicate_diagnostic.py 的源文件，因此没有对其具体实现作断言。
