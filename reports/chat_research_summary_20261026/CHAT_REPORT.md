# Green AI：模型初选与不确定性实验聊天交接报告

报告日期：2026-10-26（日本时间）。本报告可独立阅读，也可将本文件与 images 文件夹交给新的 ChatGPT 对话。

## 给接手 ChatGPT 的简要上下文

用户希望在完整微调前，使用七个预训练语言模型的 embedding 相似度分布统计，预测模型微调后的 Macro F1 排名，以减少训练和环境成本。要求保留五个指标共同实验、尽量少改原代码、使用 RTX 5080 加速、在 VS Code Notebook 内运行并展示结果。七个数据集都已完成十轮新文档采样实验。尚未基于这些十轮数据重新运行 grid search。

关键结论：当前五指标组合有部分排序/筛选能力，但不能稳定跨领域选出唯一最佳模型。必须区分固定共享公式与留出数据集选出的公式；重复样本不等于独立数据集。

## 1. 聊天中完成的工作与范围

聊天前段涉及 Ohsumed、AG News、arXiv、Rheology 的训练/预测代码适配、标签编码、GPU 使用及 grid search。早期步骤的执行记录并不全部在本报告中重新验证；本报告的定量结论以目前磁盘保存的七数据集十轮预测结果为准。

完成了：Financial 的十轮实验副本和 GPU 加速；随后 EnvData、ECtHR、ACGNews、Ohsumed、Rheology、arXiv 的十轮副本；指标汇总、逐轮 Spearman、Top1/Top3、模型排名和分布图；AG News 标签配对和打乱标签诊断。原始预测 Notebook 保留。

未完成：新十轮数据的联合 grid search、独立新数据集验证、真实 F1 的训练不确定性评估、所有数据预处理的一致化、论文实验的严格复现。

## 2. 模型、数据与标签

七个候选模型：BERT、SciBERT、LegalBERT、FinancialBERT、PharmBERT、AgricultureBERT、ChemicalBERT。目标是既有训练记录中的 Macro F1，不是本次重新训练得到的 F1。

| 数据集 | 本次预测池/标签处理 |
|---|---|
| Financial | 原 Financial 文本清理、字符长度分组；三类单标签 |
| EnvData | 2,647 条，二类；沿用字符长度分组 |
| ECtHR | 仅 train 9,000 篇；29 个实际目标标签，加无违反条款分析标记；多标签 |
| ACGNews（AG News） | train 120,000 篇，四类单标签；test 未用于预测抽样 |
| Ohsumed | 清理后 34,386 篇、23 类；同文档及重复规范文本的标签合并；多标签 |
| Rheology | train 1,612 + dev 403 = 2,015 篇，五类单标签 |
| arXiv | 51,774 篇；只保留九个高频标签；过滤后本数据没有整篇被删；多标签 |

预测配对使用类别名称/名称列表，而不是把 one-hot 或 multi-hot 的 0/1 当成类别集合。训练用向量编码与配对用标签集合是不同用途。

## 3. 每个实验具体做了什么

每数据集重复十次，每次重新按长度分组抽取约 200 篇（本次结果均为 200）；一次样本供全部七模型共用。种子和实际样本均保存，可追溯。文本经原 checkpoint tokenizer，截断/填充最大 128 tokens，batch size 64；模型 eval + inference mode，最后隐藏层做 attention-mask mean pooling，未在此次实验中微调。

200 篇共有 200×199/2=19,900 个无序非自身文档对。单标签时同类/异类；多标签时有标签交集为同标签，标签集合完全不相交为异标签。每个文档对仅进入一个总分布一次。

五个组合输入：同标签均值 μ_same、异标签均值 μ_diff、同标签 |η|、异标签 |η|、两分布的 χ² distance。η 为原代码的留一 Gaussian KDE 与严格经验 CDF 的负相关估计，不是 Fisher–Pearson skewness。

### 两层重复要分开

外层十轮：重新抽文档，测量文档样本变化导致的指标波动。

内层每模型 1,000 次：同标签分数固定，从异标签分数按“完整标签集合配对类型”分组、有放回抽取，抽出总数等于同标签对数。每次计算异标签指标与 χ² 等，然后汇总。它不是重新训练，也不是重新抽文档。

配额规则：先每组取 floor(同标签对数/组数)，余数分给迭代顺序中前面的组。组数多时，该余数规则及分组顺序可能产生影响，目前未专项验证，不能忽略。

### 重要统计限制

文档对共享文档，分数不是独立样本。内部的分层分数重采样不能直接作为完整文档抽样不确定性的可靠置信区间。十轮跨样本 SD 不是 95% CI；Notebook 中 t 区间仅是条件于当前数据和过程的重复抽样均值近似区间，不是未来一轮范围、F1 区间或覆盖率保证。七十组模型结果不是七十个独立数据集。

## 4. GPU 修改与运行情况

原代码 embedding 用 GPU，但 cosine、NumPy/SciPy η/直方图主要在 CPU。改写后 cosine、均值、χ² 的主要运算和 float64 分块 η 进入 CUDA；经验 CDF 使用排序查找保留严格小于和重复值语义。TF32 近似关闭。保留原 NumPy 随机抽样过程。

数值检查包括合成连续分数、重复值、常量、NaN 情形、相同种子的短 bootstrap 和 cosine 对照；合成 η 最大绝对差约 3.33e-16。单次 7,000 分数 η 测试 CPU 约 0.90 秒、GPU 约 0.019 秒。该约 48 倍是单函数测试，不能泛化为整实验加速倍数；尚不能声称对所有历史 CPU 实验逐项精确复核。

Ohsumed 曾出现约 12% GPU、约 97% CPU：一个样本有 2,120 种完整标签集合配对类型，Python 分组抽样形成瓶颈。低 GPU 利用率不等于未用 GPU；显存占用也不等于 GPU 在工作。旧 Notebook 内核会保留显存。

Financial 首次后台使用 Agg，图片只存 PNG；随后已嵌入输出。后续按用户选择由 VS Code 点击“运行全部”，使用 inline 图片输出。

## 5. 两种组合必须分开

固定最终组合（full_fit_exploratory），七数据集共用：

S = -0.3 Z(μ_same) + 0.2 Z(μ_diff) + 0.1 Z(|η_same|) - 0.3 Z(|η_diff|) + 0.1 Z(χ²)。

分数越大预测越好。它是排名分数，不是预测 F1。每轮在七候选模型之间 z-score，标准差 ddof=0。该公式已经用过全部七数据集，表现属于探索性拟合结果。

留出流程（held_out_*）：用其他六个数据集的旧记录选公式；当前数据集不参与该次选择。其权重和标准化可因留出任务而不同，但在某数据集十轮内固定不变，没有逐轮调参。标准化规则逐轮应用，未使用 F1 来标准化。

| 留出数据集 | 标准化 | 权重顺序：[μ_same, μ_diff, |η_same|, |η_diff|, χ²] |
|---|---|---|
| FinancialData | robust | [-0.2, 0.1, 0.1, -0.2, 0.4] |
| EnvData | rank | [-0.1, -0.2, -0.1, -0.2, 0.4] |
| ECtHR | zscore | [-0.3, 0.2, 0.1, -0.3, 0.1] |
| ACGNews | zscore | [-0.1, -0.1, -0.1, -0.5, 0.2] |
| Ohsumed | robust | [-0.3, 0.3, 0.1, -0.2, 0.1] |
| Rheology | robust | [-0.2, 0.1, 0.1, -0.5, 0.1] |
| arXiv | robust | [0.2, -0.2, 0.1, -0.3, 0.2] |

旧 grid search：五项强制非零，步长 0.1，绝对权重和为 1，最小绝对权重 0.1，共 4,032 候选；内层按数据集验证选择标准化/权重，外层留出一个数据集。全部七任务此前已经被查看，不能称作完全未触碰的最终测试。

Financial 最初分析用旧样本固定标准化参数；本报告按其余任务相同的逐轮标准化重新计算，其数字替代之前口头汇总，避免混用。

## 6. 七数据集结果：统一重新计算

Spearman 评价预测分数与真实 Macro F1 的排序关系，不是准确率。± 是十轮之间 SD。Top1 为实际最佳模型被选第一的次数；Top3 为它在预测前三中的次数。F1 regret 是实际最佳 F1 减预测第一名 F1，单位百分点；越小越好。

### 固定共享最终组合

| 数据集 | Spearman均值 ± SD | 最低–最高 | 负相关轮数 | Top1命中 | Top3含最佳 | F1损失(百分点) |
|---|---:|---:|---:|---:|---:|---:|
| FinancialData | 0.221 ± 0.294 | -0.286–0.714 | 2/10 | 0/10 | 4/10 | 1.48 |
| EnvData | 0.546 ± 0.172 | 0.286–0.857 | 0/10 | 2/10 | 9/10 | 1.74 |
| ECtHR | 0.043 ± 0.269 | -0.286–0.500 | 3/10 | 0/10 | 5/10 | 3.84 |
| ACGNews | 0.511 ± 0.149 | 0.214–0.750 | 0/10 | 9/10 | 9/10 | 0.05 |
| Ohsumed | 0.861 ± 0.066 | 0.714–0.929 | 0/10 | 1/10 | 10/10 | 1.95 |
| Rheology | 0.493 ± 0.277 | -0.107–0.821 | 1/10 | 2/10 | 7/10 | 4.64 |
| arXiv | 0.639 ± 0.277 | 0.250–1.000 | 0/10 | 3/10 | 7/10 | 2.24 |

### 留出数据集的泛化流程

| 数据集 | Spearman均值 ± SD | 最低–最高 | 负相关轮数 | Top1命中 | Top3含最佳 | F1损失(百分点) |
|---|---:|---:|---:|---:|---:|---:|
| FinancialData | -0.100 ± 0.467 | -0.679–0.750 | 6/10 | 0/10 | 1/10 | 1.70 |
| EnvData | 0.411 ± 0.221 | 0.000–0.685 | 0/10 | 0/10 | 8/10 | 2.19 |
| ECtHR | 0.043 ± 0.269 | -0.286–0.500 | 3/10 | 0/10 | 5/10 | 3.84 |
| ACGNews | 0.479 ± 0.177 | 0.179–0.750 | 0/10 | 9/10 | 10/10 | 0.05 |
| Ohsumed | 0.696 ± 0.311 | -0.107–0.929 | 1/10 | 3/10 | 9/10 | 2.21 |
| Rheology | 0.450 ± 0.304 | 0.000–0.857 | 0/10 | 2/10 | 8/10 | 4.64 |
| arXiv | 0.446 ± 0.284 | 0.143–0.893 | 0/10 | 0/10 | 6/10 | 3.19 |

等权平均：固定组合 Spearman **0.473**；留出流程 **0.346**。固定组合 Top1 **17/70**、Top3 **51/70**，平均 F1 损失 **2.28** 个百分点；留出流程分别为 **14/70**、**47/70**、**2.54** 个百分点。频率只是本次重复实验描述，不是独立成功概率或新任务覆盖率。

![整体结果](images/01_overall_results.png)

![逐轮 Spearman](images/02_spearman_all_rounds.png)

## 7. 如何解读每个任务

- Financial：固定组合平均相关性弱，留出流程平均为负；不是单纯随机噪声，现有方程的可迁移性不足。
- EnvData：有正向排序能力，但反复偏爱 BERT，未稳定选中真实最佳 AgricultureBERT。系统性偏差和采样波动同时存在。
- ECtHR：平均相关性接近零，常把实际最弱 BERT 排到前面。两流程结果一致是因为选出的权重与标准化恰好一致，不是代码复制错误。
- AG News：十轮中九轮选中最佳 BERT，选择损失很小；中间模型顺序依然不可靠，所以平均 Spearman 不算最高。
- Ohsumed：固定公式完整排序最稳定，十轮都找出真实前三名集合，但经常在 AgricultureBERT、SciBERT、PharmBERT 内互换第一；候选筛选优于唯一最优选择。
- Rheology：正向排序存在但波动明显；BERT 被多次高估，选择损失较大。SciBERT 与 AgricultureBERT 真实 F1 很近，和错选 BERT 的代价不同。
- arXiv：固定公式平均相关性较好但波动大；最高一轮为 1.0 不能代表稳定。留出组合从未选中最佳 SciBERT，不过经常选到非常接近的 AgricultureBERT，需同时看 F1 损失。

多个任务的异标签 |η| 加权贡献波动较大；这只是诊断线索。总分方差为 wᵀΣw，必须考虑相关性，不能单看各列 SD 就决定删除指标。

## 8. AG News 的“分布太清晰”检查

四类编码与配对检查通过；真实 embedding 的七十组分数都与直接类别相等/不等掩码一致。原 train 有 79 条规范化重复文本，但十轮样本均未抽到重复文本。

原图只画五模型并合并三次样本，会隐藏部分轮间波动。不同模型整体相似度基线也不同：模型曲线分开不等于标签区分能力强。

实际平均同/异标签间距：BERT 约 0.0488，PharmBERT 0.0267，AgricultureBERT 0.0186，ChemicalBERT 0.0062。打乱标签后的均值间距均接近零。支持存在标签相关结构，但未控制长度/主题等混杂，不能证明因果，也不是完整各向异性检验。

![AG News 标签对照](images/03_agnews_label_control.png)

## 9. 与 Riyadi 论文的关系

聊天中核对的论文为 Riyadi et al. (2025), Benchmarking with a Language Model Initial Selection for Text Classification Tasks。原 PDF 位于用户 Zotero 路径。

论文 Section 5 称使用单张 RTX 4090 做计算，但没有逐项说明 CPU/GPU 分工；不能据此断言每一统计运算都在 CUDA。Figure 5 报告同样本量十次采样的均值/偏度及 95% 区间。论文使用均值与 Fisher–Pearson 偏度；当前研究增加了同/异标签分布、η、χ² 与内部 1,000 次分数重采样，计算方法不是论文的简单重复。

论文写 embedding 最大长度 512；当前预测配置是 128。arXiv 论文描述九大类+Other 的十类任务，当前是九类过滤任务；ECtHR 当前标签体系等也需独立核对。因此不能直接声称复现论文设置或已经超越论文结果。

## 10. 尚未解决的问题及下一步

1. 先预先确定主要研究目标：完整排序（Spearman）还是减少训练候选（Top-k / F1 regret）。不能看完结果才随意选择最有利目标。
2. 统一七任务分析规则；保留原始数据，不将 Financial 旧固定基准分析混入逐轮标准化结果。
3. 在新 grid search 中，每数据集十轮作为一组，外层一起留出；训练目标先对轮次平均、再对数据集等权平均。不要随机拆开 490 行。
4. 五指标全部保留；若加入稳定性惩罚，其强度只能在内层数据集验证选择。不要根据外层结果调权重。
5. 和固定选某模型、旧方程、单指标诊断基线比较。单指标作为基线不等于替代用户要求的五指标主实验。
6. 审核多标签完整集合分层、余数配额顺序以及标签组均衡的研究含义；若改变定义，应创建明确的新实验版本，不覆盖旧结果。
7. 检查长度度量不一致（Financial/Env 字符，其余主要词数）、128 token 截断、预训练 embedding 与完整微调 F1 之间的匹配、F1 来源精度和折数差异。
8. 当前真实 F1 作为固定观测值；未验证近似 F1 模型的排序差异是否超出训练/验证噪声。更完整研究需加入训练种子/折间不确定性。
9. 需要额外未用于调参的新数据集。七任务已反复查看，新的七任务表现仍属于探索性证据。

## 11. 文件、原始证据与接手说明

新对话建议先阅读本报告和 final_results.csv，再查看两张总览图；如需分布细节，再查看 images 下按数据集命名的图片。

所有数字由保存结果重新计算，没有重跑模型、重新训练或更新 GridSearch/result1.csv。final_results.csv 含完整逐轮 Spearman/排名、准确权重和原始目录清单。新报告不会更改实验文件。

Financial 原实验仅有合并后的两张分布图，本包保留其原图；其余六任务提供十轮同/异标签分布。Financial 若要十轮小图，可从已保存的七十个结果 pickle 重新绘制，不必重算模型。

本报告中的文件位置相对于项目根目录 /home/amyliu/Desktop/GR：

- FinancialData: `Financial/results/uncertainty10_gpu_20261025_153340`，master_seed=20261025。
- EnvData: `EnvData/results/uncertainty10_gpu_20261025_163632`，master_seed=20261026。
- ECtHR: `ECtHR/results/uncertainty10_gpu_20261025_170358`，master_seed=20261027。
- ACGNews: `ACGNews/results/uncertainty10_gpu_20261025_174025`，master_seed=20261028。
- Ohsumed: `Ohsumed/results/uncertainty10_gpu_20261025_175601`，master_seed=20261029。
- Rheology: `Rheology/results/uncertainty10_gpu_20261025_182742`，master_seed=20261030。
- arXiv: `arXiv/results/uncertainty10_gpu_20261025_185210`，master_seed=20261031。

Notebook 副本：Financial/rankPredict_uncertainty10.ipynb、EnvData/rankPredict_uncertainty10.ipynb、ECtHR/rankPredict_uncertainty10.ipynb、ACGNews/rankPredict_uncertainty10.ipynb、Ohsumed/RankPredict_uncertainty10.ipynb、Rheology/rankPredict_uncertainty10.ipynb、arXiv/rankPredict_uncertainty10.ipynb。

### 可直接复制到新 ChatGPT 的提示

“请阅读这份 Green AI 报告和 final_results.csv。我们已经对七个数据集分别做了十轮文档采样，每轮七模型、每模型一千次内部异标签分数重采样。下一步拟用五个指标共同做按数据集分组的嵌套 grid search。请严格区分固定共享组合与留出组合，不把重复样本当独立任务，不用外层数据调参，也不要把 bootstrap SD 当成文档级 95% CI。保持原实验不变，先核对统计定义和评价目标，再实现新实验。”


## 12. 新增：论文式相关性表

从已有十轮结果及保存的 Micro/Macro F1 提取，未重跑模型。先按模型平均十轮指标，再跨七模型计算有符号 Spearman/Pearson；不是组合分数的相关性，也不是逐轮相关性的平均。

- [对应论文版式：均值与 Fisher–Pearson 偏度](images/table19_mean_skewness.png)
- [实际五指标：Macro F1](images/table_five_metrics_macro.png)
- [实际五指标：Micro F1](images/table_five_metrics_micro.png)
- 下方“相关性表计算说明”

每图另有可缩放 SVG；全精度数值和逐轮相关性波动已统一放入 final_results.csv。偏度由保存的 same_scores 计算，不等于 η，不能混用。


## 13. 清理后的唯一数值汇总

[final_results.csv](final_results.csv) 是本报告唯一数值汇总。采用长表，`value` 为数值，`record_type` 标识结果类型；按 dataset/model/repetition/procedure/target/method/metric 筛选，避免将不同类型的行当成独立样本合并计算。

| record_type | 内容 |
|---|---|
| model_input_run / model_input_mean | 每模型每轮输入及十轮均值，含真实 Micro/Macro F1 |
| correlation_of_means | 论文式表格：指标先平均，再跨七模型相关 |
| correlation_per_run | 每轮单独相关 |
| correlation_sampling_mean/std/min/max | 十轮相关系数的波动 |
| combination_summary | 两套组合的均值、SD、Top-k、regret（regret_pp 为百分点） |
| combination_run | 各轮 Spearman、Top-k、regret（f1_regret 为原始 0–1 F1 差） |
| combination_model_rank | 各轮各模型预测排名 |
| formula_weight | 各组合固定权重及标准化规则 |
| experiment_config | 主种子及原始实验目录 |
| agnews_diagnostic | AG News 标签置乱等诊断均值 |

删除的只是报告中的重复导出和过程 JSON。原始实验目录中的样本、逐模型结果、种子配置及原 Notebook 保留；模型/tokenizer 配置不属于可删除的过程 JSON。


## 14. 相关性表计算说明

主图：`table19_mean_skewness.png`（另有 SVG，全精度数据统一保存于 final_results.csv）。版式参考截图，数据全部为当前项目结果，不是论文原表。

- 每个数据集只有七个模型观测点；先对十轮指标按模型求平均，再与已有平均 F1 算 Spearman/Pearson。不是把七十行当独立样本，也不是十轮相关系数的平均。
- 主表 Mean 是同标签 cosine 均值；Skewness 从已保存 `same_scores` 计算 `scipy.stats.skew(bias=False)`，即调整 Fisher–Pearson 偏度。它不同于原代码的 η。未重新运行模型或训练。
- 表格保留相关系数正负号，没有按结果选择绝对值、反转指标或调权重。原截图没有完整说明其符号处理；不能仅按表面数值宣称超过论文。
- 两张五指标表分别对应 Micro/Macro F1，使用实际实验指标 SameMean、DffMean、SameAbsEta、DiffAbsEta、Chi2Distance。它们不是线性组合的相关性。
- Average 是七个数据集等权算术平均的描述量，不是总体合并相关性，也没有 Fisher-z 元分析含义。
- `final_results.csv` 的 correlation_per_run 与 correlation_sampling_* 记录 为补充：固定 F1，对每轮单独算相关系数及其采样波动；与主表不同。
- Micro/Macro 均有真实保存来源，记录见 本节下方来源列表；Financial/EnvData 从 Notebook 打印的六位小数均值提取，其余任务来自折/种子 CSV。Macro 与 result1.csv 按模型逐一核对通过。
- ECtHR 为三种子均值，其余为五折均值；数据、标签、128-token 截断与论文不完全一致。这是版式对齐的当前实验表，不是严格复现。
- 输入数值、完整有符号相关系数与来源均保存以便审计。只有 7 个模型，相关估计不稳定，不据此做未校正的显著性声明。


### F1 来源

- FinancialData：`AgusDatasetBasedCompare/financial_3epochAmy.ipynb`；5-fold printed mean; six decimal places。
- EnvData：`fullFIneTuningResults/envData_3epochAmy.ipynb`；5-fold printed mean; six decimal places。
- ACGNews：`ACGNews/results/ag_news_5fold/ag_news_5fold_results.csv`；5 folds。
- Rheology：`Rheology/results/rheology_5fold/rheology_5fold_results.csv`；5 folds。
- arXiv：`arXiv/results/arxiv_9class_5fold/arxiv_9class_5fold_results.csv`；5 folds。
- ECtHR：`ECtHR/real_ranking_raw_results.csv`；3 seeds。
- Ohsumed：`Ohsumed/results/ohsumed_5fold/ohsumed_5fold_results.csv`；5 folds。
