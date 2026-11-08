  src/                         7 个重点代码文件
  results/ALL_METHODS_VS_BASE.csv  以 BASE 为基准的总表

重点代码
  initialization.py   初始化：构造 K8 候选、Transformer 残差选择
  crossover.py        交叉：构造候选子代对、Transformer 残差选择
  mutation.py         变异：构造候选、Transformer 排序、执行所选变异
  combined.py         I+C+M 三组件同时启用时的协调
  network.py          Transformer 网络结构
  experiment.py       相同实例、种子、预算下的八组比较
  baseline_uncertainty.py  原始 NSGA-II、编码与目标评价
  support/            上述代码实际依赖的公共运行模块，不需要从这里开始阅读

目录说明
  assets/   九个冻结权重、实例、归一化参数、基础网络数据
  results/  四组分表、总表、240 次逐次结果与完整结果 JSON
  run.py    唯一运行入口；自动处理服务器旧路径
  environment.yml  原服务器兼容环境配置（Linux CPU / Python 3.10）

在当前文件夹执行：
  conda env create -f environment.yml
  conda activate icm-components
  python run.py --verify        # 核验模型并重算已有结果，不运行搜索
  python run.py --run           # 重新运行八组实验，结果保存到 new_runs/


验证：9 个模型加载和240次原结果统计重算通过；8 种方法在一个共同实例/种子上，
