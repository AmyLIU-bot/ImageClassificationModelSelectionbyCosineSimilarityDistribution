# Uncertainty test：七数据集误差与选择稳定性分析

日期：2026-10-26。仅从已有结果抽取，没有重新训练语言模型、重新计算 embedding 或修改原实验。主目标是解释采样敏感性和系统性错误；系数重选为单独的探索实验，不替换原公式。

## 实验定义

7 数据集 × 10 文档样本 × 7 模型。五个指标全部保留。真实 Macro F1 固定来自 GridSearch/result1.csv。逐轮标准化严格沿用原 zscore/robust/rank 规则；固定共享公式与历史留出公式分别分析。原始指标有符号相关性保留，不逐轮取绝对值。

唯一数值文件 uncertainty_results.csv 使用长表：record_type、dataset、model、model_b、repetition、procedure、selection、method、metric、statistic、value、source。不同 record_type 不能当独立样本直接混合求均值。Notebook uncertainty_analysis.ipynb 是必要的可复现分析代码，不是临时 agent 操作脚本。

## 1. 单指标：350 个逐轮 Spearman

这里的相关系数回答关联方向；十轮间波动才描述采样敏感性。方向翻转指十轮内既有正也有负，零单独计数。

| dataset       | metric       |    mean |     sd |   minimum |   maximum |   positive |   negative |   zero | has_both_signs   |
|:--------------|:-------------|--------:|-------:|----------:|----------:|-----------:|-----------:|-------:|:-----------------|
| ACGNews       | Chi2Distance |  0.7107 | 0.1653 |    0.4643 |    0.9286 |         10 |          0 |      0 | False            |
| ACGNews       | DffMean      | -0.6429 | 0      |   -0.6429 |   -0.6429 |          0 |         10 |      0 | False            |
| ACGNews       | DiffAbsEta   | -0.35   | 0.2142 |   -0.6429 |   -0.0714 |          0 |         10 |      0 | False            |
| ACGNews       | SameAbsEta   | -0.4607 | 0.167  |   -0.6429 |   -0.1786 |          0 |         10 |      0 | False            |
| ACGNews       | SameMean     | -0.6357 | 0.0151 |   -0.6429 |   -0.6071 |          0 |         10 |      0 | False            |
| ECtHR         | Chi2Distance |  0.0464 | 0.4629 |   -0.4643 |    0.6786 |          4 |          6 |      0 | True             |
| ECtHR         | DffMean      |  0.2964 | 0.0339 |    0.2857 |    0.3929 |         10 |          0 |      0 | False            |
| ECtHR         | DiffAbsEta   | -0.3179 | 0.3822 |   -0.8929 |    0.2857 |          3 |          7 |      0 | True             |
| ECtHR         | SameAbsEta   | -0.1679 | 0.3199 |   -0.7143 |    0.2143 |          4 |          6 |      0 | True             |
| ECtHR         | SameMean     |  0.3179 | 0.0518 |    0.2857 |    0.3929 |         10 |          0 |      0 | False            |
| EnvData       | Chi2Distance |  0.15   | 0.4511 |   -0.4643 |    0.8571 |          6 |          3 |      1 | True             |
| EnvData       | DffMean      |  0.0179 | 0.0911 |   -0.1429 |    0.0714 |          7 |          3 |      0 | True             |
| EnvData       | DiffAbsEta   | -0.625  | 0.1914 |   -0.9286 |   -0.3214 |          0 |         10 |      0 | False            |
| EnvData       | SameAbsEta   | -0.4321 | 0.2779 |   -0.7143 |    0.1429 |          1 |          9 |      0 | True             |
| EnvData       | SameMean     | -0.1    | 0.0749 |   -0.1429 |    0.0714 |          1 |          9 |      0 | True             |
| FinancialData | Chi2Distance | -0.0179 | 0.4394 |   -0.6429 |    0.7143 |          4 |          6 |      0 | True             |
| FinancialData | DffMean      | -0.2429 | 0.1075 |   -0.3571 |   -0.0714 |          0 |         10 |      0 | False            |
| FinancialData | DiffAbsEta   | -0.3714 | 0.2266 |   -0.7857 |   -0.0357 |          0 |         10 |      0 | False            |
| FinancialData | SameAbsEta   | -0.625  | 0.1545 |   -0.7857 |   -0.3214 |          0 |         10 |      0 | False            |
| FinancialData | SameMean     | -0.1571 | 0.1107 |   -0.2857 |   -0.0714 |          0 |         10 |      0 | False            |
| Ohsumed       | Chi2Distance |  0.6893 | 0.1105 |    0.5714 |    0.9286 |         10 |          0 |      0 | False            |
| Ohsumed       | DffMean      | -0.6786 | 0      |   -0.6786 |   -0.6786 |          0 |         10 |      0 | False            |
| Ohsumed       | DiffAbsEta   | -0.8071 | 0.0845 |   -0.9643 |   -0.6429 |          0 |         10 |      0 | False            |
| Ohsumed       | SameAbsEta   | -0.775  | 0.0907 |   -0.8929 |   -0.6429 |          0 |         10 |      0 | False            |
| Ohsumed       | SameMean     | -0.6786 | 0      |   -0.6786 |   -0.6786 |          0 |         10 |      0 | False            |
| Rheology      | Chi2Distance |  0.4964 | 0.2342 |    0.1429 |    0.7857 |         10 |          0 |      0 | False            |
| Rheology      | DffMean      |  0.0286 | 0.0151 |    0      |    0.0357 |          8 |          0 |      2 | False            |
| Rheology      | DiffAbsEta   | -0.4571 | 0.2997 |   -0.8214 |    0      |          0 |          8 |      2 | False            |
| Rheology      | SameAbsEta   | -0.2893 | 0.4379 |   -0.8929 |    0.3929 |          2 |          6 |      2 | True             |
| Rheology      | SameMean     |  0.025  | 0.0173 |    0      |    0.0357 |          7 |          0 |      3 | False            |
| arXiv         | Chi2Distance |  0.6786 | 0.2259 |    0.1429 |    0.8929 |         10 |          0 |      0 | False            |
| arXiv         | DffMean      | -0.7    | 0.0845 |   -0.7857 |   -0.5714 |          0 |         10 |      0 | False            |
| arXiv         | DiffAbsEta   | -0.4643 | 0.295  |   -1      |   -0.1071 |          0 |         10 |      0 | False            |
| arXiv         | SameAbsEta   | -0.4821 | 0.1719 |   -0.75   |   -0.2143 |          0 |         10 |      0 | False            |
| arXiv         | SameMean     | -0.6786 | 0      |   -0.6786 |   -0.6786 |          0 |         10 |      0 | False            |

共有 **9/35 个数据集—指标组合**出现正负方向翻转。单纯负相关若方向稳定，可能适合负权重；不能直接称为指标无效。

![单指标](images/01_single_metric_stability.png)

## 2. 指标波动如何传入总分

CSV 保存原始值、标准化值、加权贡献的十轮均值/SD，每模型每轮分数和排名，以及完整 5×5 加权协方差矩阵。验证全部 98 个数据集—模型—公式的 Var(S)=wᵀΣw 恒等式通过。

Cov(C_j,S) 作为可加的方差诊断，和为总方差，但可能为负；不是因果重要性。对角线之和与总方差的差就是交叉协方差项。十轮协方差仅初步估计。

贡献箱线图合并了模型与轮次，仅用于总览；个体解释应在 CSV 筛选具体模型查看，不能把跨模型差异当抽样不确定性。

![贡献](images/02_weighted_contributions.png)
![方差](images/03_score_variance.png)

## 3. 关键模型对

A 总是已有平均真实 F1 更高的模型；正 ΔS 表示预测方向正确。frequent_wrong 对象由这次固定公式的十轮结果识别，是事后诊断，不是预注册检验。所有 21 对也保存，避免只选择有利案例。下面使用固定共享公式。

| dataset       | comparison             | model           | model_b         |   f1_gap_pp |    mean |     sd |   minimum |   maximum |   wins |   losses |   ties |
|:--------------|:-----------------------|:----------------|:----------------|------------:|--------:|-------:|----------:|----------:|-------:|---------:|-------:|
| FinancialData | best_vs_second         | AgricultureBERT | BERT            |      0.7208 | -0.1415 | 0.3031 |   -0.6703 |    0.2562 |      5 |        5 |      0 |
| FinancialData | best_vs_frequent_wrong | AgricultureBERT | FinancialBERT   |      1.7773 | -0.2819 | 0.3846 |   -0.8888 |    0.5945 |      1 |        9 |      0 |
| FinancialData | best_vs_worst          | AgricultureBERT | ChemicalBERT    |      8.8482 |  0.2768 | 0.2963 |   -0.4104 |    0.5919 |      8 |        2 |      0 |
| EnvData       | best_vs_second         | AgricultureBERT | SciBERT         |      1.7505 |  0.1694 | 0.4205 |   -0.7861 |    0.6731 |      8 |        2 |      0 |
| EnvData       | best_vs_frequent_wrong | AgricultureBERT | BERT            |      2.0836 | -0.1586 | 0.1868 |   -0.5314 |    0.0874 |      3 |        7 |      0 |
| EnvData       | best_vs_worst          | AgricultureBERT | ChemicalBERT    |      7.5074 |  0.5794 | 0.2462 |    0.2947 |    1.0137 |     10 |        0 |      0 |
| ECtHR         | best_vs_second         | SciBERT         | AgricultureBERT |      0.6013 | -0.0286 | 0.2348 |   -0.3848 |    0.2216 |      6 |        4 |      0 |
| ECtHR         | best_vs_frequent_wrong | SciBERT         | BERT            |      5.6638 | -0.2282 | 0.351  |   -0.8283 |    0.2964 |      2 |        8 |      0 |
| ECtHR         | best_vs_worst          | SciBERT         | BERT            |      5.6638 | -0.2282 | 0.351  |   -0.8283 |    0.2964 |      2 |        8 |      0 |
| ACGNews       | best_vs_second         | BERT            | PharmBERT       |      0.1978 |  0.6339 | 0.2894 |   -0.0674 |    0.8893 |      9 |        1 |      0 |
| ACGNews       | best_vs_frequent_wrong | BERT            | SciBERT         |      0.4735 |  0.6767 | 0.331  |   -0.1975 |    0.9356 |      9 |        1 |      0 |
| ACGNews       | best_vs_worst          | BERT            | FinancialBERT   |      1.9668 |  0.6541 | 0.1919 |    0.1609 |    0.8753 |     10 |        0 |      0 |
| Ohsumed       | best_vs_second         | AgricultureBERT | SciBERT         |      0.1626 | -0.0069 | 0.1269 |   -0.1683 |    0.22   |      4 |        6 |      0 |
| Ohsumed       | best_vs_frequent_wrong | AgricultureBERT | SciBERT         |      0.1626 | -0.0069 | 0.1269 |   -0.1683 |    0.22   |      4 |        6 |      0 |
| Ohsumed       | best_vs_worst          | AgricultureBERT | LegalBERT       |     15.214  |  0.7308 | 0.129  |    0.5625 |    0.9554 |     10 |        0 |      0 |
| Rheology      | best_vs_second         | AgricultureBERT | SciBERT         |      0.1491 |  0.1535 | 0.2959 |   -0.4272 |    0.5121 |      7 |        3 |      0 |
| Rheology      | best_vs_frequent_wrong | AgricultureBERT | BERT            |     10.2521 | -0.009  | 0.3284 |   -0.3718 |    0.6857 |      4 |        6 |      0 |
| Rheology      | best_vs_worst          | AgricultureBERT | LegalBERT       |     14.0174 |  0.4782 | 0.3164 |   -0.077  |    0.9874 |      9 |        1 |      0 |
| arXiv         | best_vs_second         | SciBERT         | AgricultureBERT |      0.3428 | -0.0273 | 0.2725 |   -0.4619 |    0.3485 |      5 |        5 |      0 |
| arXiv         | best_vs_frequent_wrong | SciBERT         | PharmBERT       |      3.8784 |  0.0076 | 0.4206 |   -0.698  |    0.5995 |      4 |        6 |      0 |
| arXiv         | best_vs_worst          | SciBERT         | ChemicalBERT    |      8.0305 |  0.6284 | 0.3324 |    0.1915 |    1.0845 |     10 |        0 |      0 |

分数差跨零表示顺序采样敏感；十轮始终排反表示观测到稳定错误，不能据十次推断未来必然排反。8/10 是频率，不是“80% 概率 A 更好”。F1 非常接近时应同时看实际 regret。

![模型对](images/04_key_model_pairs.png)

## 4. 重新选择权重：100 次/留出任务，共 700 次

保留原 4,032 个候选（五项非零、0.1 步长、绝对权重和为 1）。每次仅在其余六数据集内部，对每数据集的十个已有轮次有放回抽十次；先对轮次平均候选 Spearman，再对任务等权平均。内层留出训练数据集，选择 zscore/robust/rank；然后用全部六任务选权重。并列沿用原确定性规则。

外层留出的十轮仅在配置确定后评价；改写外层目标的隔离检查对七任务通过。未用外层表现选惩罚强度或筛选权重。选择 seed=20261026。

这只反映已观察十轮内的系数选择敏感性；不是回归系数 CI，未覆盖所有可能数据扰动，也没有解决七任务已被反复查看的问题。

近优集合预先定义为所选标准化下训练平均 Spearman 与最佳差不超过 0.01。记录近优权重数、距所选权重的最大 L1 距离、各权重范围；最多20个按候选索引均匀取样的近优解，在外层十轮上比较其排名与所选解的 Spearman。后者仅用于描述，不能用来反选模型。

| dataset       |   mean_test_rho |   selection_sd_test_rho |   mean_near_candidates |   mean_near_max_l1 |   mean_near_prediction_agreement |
|:--------------|----------------:|------------------------:|-----------------------:|-------------------:|---------------------------------:|
| ACGNews       |          0.4812 |                  0.0735 |                   4.64 |              0.658 |                           0.9807 |
| ECtHR         |          0.0461 |                  0.0682 |                   6.57 |              0.772 |                           0.9688 |
| EnvData       |          0.4554 |                  0.0563 |                   5.85 |              0.72  |                           0.9416 |
| FinancialData |          0.3041 |                  0.1053 |                   5.28 |              0.72  |                           0.9459 |
| Ohsumed       |          0.8193 |                  0.0994 |                   3.82 |              0.624 |                           0.9524 |
| Rheology      |          0.4536 |                  0.0606 |                   5.16 |              0.646 |                           0.9625 |
| arXiv         |          0.5669 |                  0.0915 |                   5.55 |              0.774 |                           0.9579 |

标准化选择次数（每行100）：

| dataset       |   rank |   robust |   zscore |
|:--------------|-------:|---------:|---------:|
| FinancialData |     40 |       50 |       10 |
| EnvData       |     36 |       38 |       26 |
| ECtHR         |     40 |       44 |       16 |
| ACGNews       |     27 |       18 |       55 |
| Ohsumed       |     38 |       30 |       32 |
| Rheology      |     26 |       55 |       19 |
| arXiv         |     17 |       20 |       63 |

![权重](images/05_coefficient_selection.png)

## 5. 参照：不使用五指标的固定模型偏好基线

只用外层训练任务的平均真实排名确定固定模型优先顺序，在目标任务上评价。它是比较基线，不是取代五指标主实验。

| dataset       |    rho |   top1 |   top3 |   regret |
|:--------------|-------:|-------:|-------:|---------:|
| FinancialData | 0.6847 |      1 |      1 |   0      |
| EnvData       | 0.75   |      1 |      1 |   0      |
| ECtHR         | 0.4643 |      0 |      1 |   0.006  |
| ACGNews       | 0.5714 |      0 |      1 |   0.0049 |
| Ohsumed       | 0.75   |      1 |      1 |   0      |
| Rheology      | 0.6071 |      1 |      1 |   0      |
| arXiv         | 0.9286 |      0 |      1 |   0.0034 |

## 6. 真实 F1 自身波动

直接提取已有每折或 seed 的 Macro F1。Financial/Env 只有可靠打印均值与 SD，保留六位精度，不虚构配对折值；其他任务额外保存实际第一与第二模型按同 fold/seed 配对的差值均值、SD、胜出次数。

| dataset       | model           | unit                 |   n |     mean |       sd |
|:--------------|:----------------|:---------------------|----:|---------:|---------:|
| Ohsumed       | PharmBERT       | fold                 |   5 | 0.682587 | 0.004248 |
| Ohsumed       | FinancialBERT   | fold                 |   5 | 0.608793 | 0.006484 |
| Ohsumed       | SciBERT         | fold                 |   5 | 0.727792 | 0.005204 |
| Ohsumed       | BERT            | fold                 |   5 | 0.597272 | 0.006811 |
| Ohsumed       | LegalBERT       | fold                 |   5 | 0.577278 | 0.003937 |
| Ohsumed       | AgricultureBERT | fold                 |   5 | 0.729418 | 0.003852 |
| Ohsumed       | ChemicalBERT    | fold                 |   5 | 0.582626 | 0.005372 |
| ACGNews       | PharmBERT       | fold                 |   5 | 0.945411 | 0.000931 |
| ACGNews       | FinancialBERT   | fold                 |   5 | 0.92772  | 0.001455 |
| ACGNews       | SciBERT         | fold                 |   5 | 0.942653 | 0.002073 |
| ACGNews       | BERT            | fold                 |   5 | 0.947388 | 0.001008 |
| ACGNews       | LegalBERT       | fold                 |   5 | 0.941838 | 0.001302 |
| ACGNews       | AgricultureBERT | fold                 |   5 | 0.942519 | 0.001233 |
| ACGNews       | ChemicalBERT    | fold                 |   5 | 0.93211  | 0.000744 |
| arXiv         | PharmBERT       | fold                 |   5 | 0.524315 | 0.008297 |
| arXiv         | FinancialBERT   | fold                 |   5 | 0.506256 | 0.008733 |
| arXiv         | SciBERT         | fold                 |   5 | 0.563098 | 0.006205 |
| arXiv         | BERT            | fold                 |   5 | 0.518951 | 0.008129 |
| arXiv         | LegalBERT       | fold                 |   5 | 0.502278 | 0.004159 |
| arXiv         | AgricultureBERT | fold                 |   5 | 0.55967  | 0.0124   |
| arXiv         | ChemicalBERT    | fold                 |   5 | 0.482794 | 0.008898 |
| Rheology      | PharmBERT       | fold                 |   5 | 0.723304 | 0.016389 |
| Rheology      | FinancialBERT   | fold                 |   5 | 0.659315 | 0.028792 |
| Rheology      | SciBERT         | fold                 |   5 | 0.779579 | 0.008061 |
| Rheology      | BERT            | fold                 |   5 | 0.678549 | 0.052768 |
| Rheology      | LegalBERT       | fold                 |   5 | 0.640896 | 0.056043 |
| Rheology      | AgricultureBERT | fold                 |   5 | 0.78107  | 0.025743 |
| Rheology      | ChemicalBERT    | fold                 |   5 | 0.729972 | 0.012436 |
| ECtHR         | PharmBERT       | seed                 |   3 | 0.069325 | 0.006241 |
| ECtHR         | FinancialBERT   | seed                 |   3 | 0.088351 | 0.001163 |
| ECtHR         | SciBERT         | seed                 |   3 | 0.107745 | 0.013359 |
| ECtHR         | BERT            | seed                 |   3 | 0.051107 | 0.002603 |
| ECtHR         | LegalBERT       | seed                 |   3 | 0.095641 | 0.003999 |
| ECtHR         | AgricultureBERT | seed                 |   3 | 0.101732 | 0.011482 |
| ECtHR         | ChemicalBERT    | seed                 |   3 | 0.068955 | 0.01014  |
| FinancialData | AgricultureBERT | printed_fold_summary |   5 | 0.850527 | 0.008596 |
| FinancialData | BERT            | printed_fold_summary |   5 | 0.843319 | 0.01398  |
| FinancialData | SciBERT         | printed_fold_summary |   5 | 0.840296 | 0.012817 |
| FinancialData | FinancialBERT   | printed_fold_summary |   5 | 0.832754 | 0.008038 |
| FinancialData | LegalBERT       | printed_fold_summary |   5 | 0.832125 | 0.005234 |
| FinancialData | PharmBERT       | printed_fold_summary |   5 | 0.831389 | 0.009239 |
| FinancialData | ChemicalBERT    | printed_fold_summary |   5 | 0.762045 | 0.016234 |
| EnvData       | AgricultureBERT | printed_fold_summary |   5 | 0.898844 | 0.011027 |
| EnvData       | SciBERT         | printed_fold_summary |   5 | 0.881339 | 0.021754 |
| EnvData       | BERT            | printed_fold_summary |   5 | 0.878008 | 0.01245  |
| EnvData       | LegalBERT       | printed_fold_summary |   5 | 0.876135 | 0.014626 |
| EnvData       | FinancialBERT   | printed_fold_summary |   5 | 0.868348 | 0.013063 |
| EnvData       | PharmBERT       | printed_fold_summary |   5 | 0.867752 | 0.015798 |
| EnvData       | ChemicalBERT    | printed_fold_summary |   5 | 0.82377  | 0.017622 |

折/seed 数很少，交叉验证折也并非独立重复训练任务；配对差值只是描述，不能直接称作模型真实优劣的95%置信区间。F1 的这些波动尚未传播进组合选择或这次 Spearman。

## 已知限制与下一步

样本共享文档对，内部 bootstrap 不是文档级独立置信区间；七十重复不是独立任务。数据预处理长度分组规则未被统一，本分析保持原实验。不同模型的训练 F1 来源及折/seed 数不同。原多标签按完整标签集配对分组的重采样及余数分配规则未改变。

下一步可以预先确定主要目标（Spearman、Top-k 或 regret），用当前分析定位失败机制，再设计新版本方法；不得据这次外层结果反复优化并称其为独立测试。需要新任务外部验证。

## 输入来源

- FinancialData: `Financial/results/uncertainty10_gpu_20261025_153340`
- EnvData: `EnvData/results/uncertainty10_gpu_20261025_163632`
- ECtHR: `ECtHR/results/uncertainty10_gpu_20261025_170358`
- ACGNews: `ACGNews/results/uncertainty10_gpu_20261025_174025`
- Ohsumed: `Ohsumed/results/uncertainty10_gpu_20261025_175601`
- Rheology: `Rheology/results/uncertainty10_gpu_20261025_182742`
- arXiv: `arXiv/results/uncertainty10_gpu_20261025_185210`

真实 F1 与权重来源：GridSearch/result1.csv；GridSearch/results_nested_five_required/。

## 结果解释与优先关注

- 35个数据集—指标组合中，9个出现正负方向翻转，集中在 ECtHR、EnvData、Financial 的 χ² 以及部分 η 指标；完整明细见第一节。
- 原固定公式存在十轮始终排反的模型对：

| dataset   | model         | model_b       |   f1_gap |    mean |     sd |   wins |   losses |
|:----------|:--------------|:--------------|---------:|--------:|-------:|-------:|---------:|
| ECtHR     | ChemicalBERT  | BERT          |   0.0178 | -0.9158 | 0.1989 |      0 |       10 |
| ECtHR     | FinancialBERT | BERT          |   0.0372 | -0.4926 | 0.3535 |      0 |       10 |
| ECtHR     | FinancialBERT | PharmBERT     |   0.019  | -0.3956 | 0.2918 |      0 |       10 |
| ACGNews   | ChemicalBERT  | FinancialBERT |   0.0044 | -0.5741 | 0.1244 |      0 |       10 |

这些是系统性偏差的观测证据，增加相同机制下的采样次数不保证能修正。

- 训练任务平均排名的固定偏好基线在七任务的平均 Spearman 为 0.679，Top1 命中 4/7，Top3 含最佳 7/7。当前复杂组合必须认真与该基线比较，不能只展示自身正相关。此基线仍来自相同七个已探索任务，不是外部最终测试。
- 部分近优权重的 L1 差异较大，但外层排名与所选解的平均相关性仍较高；这支持“系数不唯一不一定等于预测完全不稳定”，不支持将权重大小解释为因果重要性。
- 权重图混合了不同标准化选择，数值尺度未必可直接比较；CSV 另存 method 分层的均值、SD、符号次数。选择波动的SD与文档采样波动SD是两层不同统计量。
- 新权重重选的外层均值不能直接宣称比旧方法有显著进步：训练输入与程序已改变，且外层任务已被反复查看，需预设后续实验和新数据验证。
