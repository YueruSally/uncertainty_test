# 不确定性 + CCP + Transformer/PPO

本仓库用于比较初始化（I）、交叉（C）和变异（M），并在三个组件中加入 Transformer/PPO。成本、排放和完成时间均在冻结的不确定性情景中计算 q90，所有方法使用相同实例、随机种子、评价预算、归一化范围和 HV 参考点。

## 文件

```text
expanda/
├── baseline_uncertainty.py              # NSGA-II、不确定性和 CCP-q90
├── transformer_ppo/                     # I/C/M 控制器、Transformer、PPO、HV 奖励
├── evaluate_i_directed_initialization.py
├── evaluate_c_directed_crossover.py
├── evaluate_m_late_replace_frequency.py
├── collect_m_teacher_rollouts.py         # 收集 M 训练数据
├── train_m_balanced_teacher_transformer.py
├── evaluate_m_balanced_teacher_transformer.py
├── run_*                                 # 四个实验入口
└── data/                                 # 运输网络和边境数据
results/                                  # 服务器直接输出的原始结果
requirements.txt                          # Python 依赖
```

`transformer_ppo/` 内的文件互相调用，属于核心模块，不能继续单独删除。

## 当前结果

400 次评价的小范围实验：

| 方法 | 平均 HV 提升 | 平均时间 |
|---|---:|---:|
| Base | 0.00% | 10.00 秒 |
| 变异 M | 9.42% | 10.70 秒 |
| 初始化 I | 16.35% | 8.51 秒 |

M Transformer 小试验：

| 方法 | 平均 HV 提升 | 平均时间 |
|---|---:|---:|
| Balanced 参考方法 | 8.96% | 8.94 秒 |
| Transformer BC | 8.16% | 29.21 秒 |

Transformer 已经可以训练和运行，但目前低于参考方法，因此还不能认为学习成功。

## 使用

```bash
python3 -m pip install -r requirements.txt
cd expanda
```

准备好实例目录、归一化文件和初始 checkpoint 后，选择一个入口运行：

```bash
bash run_i_directed_initialization_screen_v1.sh
bash run_c_joint_single_early04_screen_v4.sh
bash run_m_directed_balanced_validation_v1.sh
bash run_m_balanced_teacher_transformer_pilot_v1.sh
```

运行前通过 `INSTANCES`、`NORMALIZATION`、`CHECKPOINT` 或 `BOOTSTRAP_CHECKPOINT` 指定实际路径。实例和归一化文件可分别由 `generate_transformer_ppo_instances.py` 与 `freeze_transformer_ppo_normalization.py` 生成。

代码来源版本：`4c5709a`。模型 checkpoint 和运行日志属于生成文件，不放入源码仓库。
