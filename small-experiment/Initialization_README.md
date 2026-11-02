# I 初始化：CCP 小范围实验

这是当时 I 定向初始化小范围实验的精简包，不包含 Transformer、PPO、Teacher 训练、HV gate 或任何 test 文件。

## 实验条件

- 10 个实例 × 3 个算法种子；
- 200 个冻结不确定性情景；
- 成本、排放和完成时间分别取 q90；
- 每次运行 400 次真实目标评价；
- 种群 20；
- 对比 Base、Balanced、Diverse 和 Reliability 初始化。

正式选中的小范围策略是 `I_DIRECTED_BALANCED`：保留 Base 的 12 个固定初始个体，只定向控制最后 8 个随机个体的路径，并综合成本、排放、时间、延误和准时概率进行选择。

## 直接结果

`result/ORIGINAL_SERVER_OUTPUT.txt` 是当时服务器终端输出的原始文本，没有重新计算、没有制作图表。原始结果中 `I_DIRECTED_BALANCED` 为：

- 30 次配对运行；
- 平均 HV 提升 20.6716%；
- 中位数提升 12.0277%；
- 27 胜、0 平、3 负；
- 最差结果 -8.1473%。

## 程序文件

- `program/run_I_small_screen.sh`：无测试步骤的实验入口。
- `program/evaluate_i_directed_initialization.py`：运行 Base 和各 I 策略并生成原始结果。
- `program/transformer_ppo/initialization_rule.py`：Directed Balanced 初始化的核心实现。
- `program/transformer_ppo/initialization_rule_features.py`：初始化路径的成本、排放、时间和可靠性特征。
- `program/run_learning_components_deterministic.py`：单组件运行入口。
- `program/transformer_ppo/episode.py`：把 I 控制器接入一次 NSGA-II/CCP 运行。
- `program/baseline_uncertainty.py`：不确定性情景、CCP-q90 和 NSGA-II 主程序。

`transformer_ppo` 是项目原有共享目录名；本实验执行的是固定 I 规则，没有运行 Transformer。

这些是主要修改文件，不是可脱离完整仓库独立运行的完整项目。复现时应把它们放回服务器完整 `expanda` 目录，并提供实例、归一化文件和已有 Base 结果目录。
