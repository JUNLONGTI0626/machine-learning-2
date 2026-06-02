# 10年版 DY 高风险状态分类模型对照实验（75/80/85 标签）

## 实验设定
- 主线：true TVP-VAR-DY features -> future high-DSV20-state classification
- 固定特征：tci_total, net_wti, net_henry_hub, net_brent, net_rbob, net_pjm_west, net_cels, tci_total_lag1, tci_total_roll5_std
- 固定时间切分：train 2016-01-12~2022-09-12；validation 2022-10-21~2024-03-04；test 2024-04-15~2025-12-31
- 面板实际最后日期：2025-11-21。
- 按要求仍使用既有10年版固定边界。

## 标签口径
- high_dsv20_state_75: DSV20 > rolling 75% threshold
- high_dsv20_state_80: DSV20 > rolling 80% threshold
- high_dsv20_state_85: DSV20 > rolling 85% threshold
- rolling threshold 仅基于 t 时点前历史 DSV20（shift(1)+expanding quantile）。

## 各标签最佳模型（综合规则）
- 75% 标签：**random_forest**（AP=0.6971, F1=0.7161, Recall=1.0000, BalAcc=0.5000, ROC AUC=0.6627）
- 80% 标签：**random_forest**（AP=0.6208, F1=0.6222, Recall=0.6806, BalAcc=0.6107, ROC AUC=0.6622）
- 85% 标签：**random_forest**（AP=0.6281, F1=0.6187, Recall=0.9587, BalAcc=0.6002, ROC AUC=0.6882）

## 跨标签稳定性
- 最稳定模型：**random_forest**（在 3 组标签中最优）。
- 家族层面AP均值领先：75%: kernel; 80%: tree; 85%: tree。

## 问题逐条回答
1. 75% 标签综合最好模型：random_forest。
2. 80% 标签综合最好模型：random_forest。
3. 85% 标签综合最好模型：random_forest。
4. 存在相对稳定模型：random_forest。
5. 结合模型家族比较，树模型通常更适合该任务，其次是线性与核方法；KNN与朴素贝叶斯更依赖样本局部结构与分布假设。
6. 最佳模型在80%标签上更偏向 recall。
7. 75% 标签阳性覆盖更广，通常更适合“风险升温预警”识别。
8. 80% 标签仍可作为主结果，能在预警覆盖与误报控制之间取得平衡。
9. 85% 标签可作为更严格的稳健性标签，用于检验模型对极端高风险状态的识别能力。
10. 若不同标签最优模型不同，论文主模型建议优先80%标签最优，并要求其在75/85方向一致。
11. 若当前最优模型在三标签上均保持可接受AP与F1，值得推进到下一轮特征组、winsorized与DY参数稳健性。

## 跳过/失败模型记录
- label 75, model xgboost_classifier: unavailable: No module named 'xgboost'
- label 75, model lightgbm_classifier: unavailable: No module named 'lightgbm'
- label 80, model xgboost_classifier: unavailable: No module named 'xgboost'
- label 80, model lightgbm_classifier: unavailable: No module named 'lightgbm'
- label 85, model xgboost_classifier: unavailable: No module named 'xgboost'
- label 85, model lightgbm_classifier: unavailable: No module named 'lightgbm'
