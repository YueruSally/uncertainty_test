# M 变异：CCP 小范围实验

这是当时 M 定向变异小范围实验的精简包，不包含 Transformer、PPO、Teacher 训练或任何 test 文件。

## 实验条件

- 10 个实例 × 3 个算法种子；
- 200 个冻结不确定性情景；
- 成本、排放和完成时间分别取 q90；
- 每次运行 400 次真实目标评价；
- 种群 20；
- 对比 Base、随机 Replace 和三种定向 Replace。

正式选中的小范围策略是 `REPLACE_G10_JOINT_BALANCED`：从第 10 代开始，根据成本、排放、时间、延误和准时概率联合选择“批次 × 替换路径”。

## 直接结果

`result/ORIGINAL_SERVER_OUTPUT.txt` 是当时服务器终端输出的原始文本，没有重新计算、没有制作图表。原始结果中 `REPLACE_G10_JOINT_BALANCED` 为：

- 30 次配对运行；
- 平均 HV 提升 14.1947%；
- 中位数提升 14.0707%；
- 28 胜、1 平、1 负；
- 最差结果 -0.6095%。

## 程序文件

- `program/run_M_small_screen.sh`：无测试步骤的实验入口。
- `program/evaluate_m_late_replace_frequency.py`：运行 Base 和各 M 策略并生成原始结果。
- `program/transformer_ppo/controller.py`：`joint_balanced` M 定向 Replace 的核心实现。目录名是项目遗留名称，本实验没有运行 Transformer。
- `program/run_learning_components_deterministic.py`：单组件运行入口。
- `program/transformer_ppo/episode.py`：把 M 控制器接入一次 NSGA-II/CCP 运行。
- `program/baseline_uncertainty.py`：不确定性情景、CCP-q90 和 NSGA-II 主程序。

这些是主要修改文件，不是可脱离完整仓库独立运行的完整项目。复现时应把它们放回服务器完整 `expanda` 目录，并提供实例、归一化文件和已有 Base 结果目录。
