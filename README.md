BetterCode/
  ├── data/                          # 数据文件夹
  │       ├── raw/                   # 原始数据
  │       ├── cleaned/               # 清洗后的数据
  │       ├── results/               # 实验结果
  │       └── split_data/            # 分组实验数据
  │
  ├── modules/                       # 核心功能模块
  │       ├── data_cleaning/         # 数据清洗模块
  │       │       ├── __init__.py
  │       │       ├── base_cleaner.py          # 清洗模块基类
  │       │       ├── dependency_cleaner.py    # 句子信息分析器
  │       │       ├── train_cleaner.py         # 训练数据生成
  │       │       └── test_cleaner.py          # 测试数据生成
  │       ├── experiments/           # 实验模块
  │       │       ├── __init__.py
  │       │       ├── base_experiment.py       # 实验基类
  │       │       ├── experiment_registry.py   # 实验注册表
  │       │       └── runners/                 # 插件化实验逻辑
  │       │              └── experiment_1     # 实验代码集合
  │       │                        ├── experiment_1.py  #实验1
  │       │                        ├── experiment_2.py  #实验2
  │       │                        ├── experiment_3.py  #实验3
  │       │                        ├── experiment_4.py  #实验4
  │       │                        ├── experiment_5.py  #实验5
  │       │                        └── experiment_6.py  #实验6
  │       └── utils/                        # 通用工具库
  │               ├── __init__.py
  │               ├── hash.py              # 数据映射
  │               ├── InvertedIndex.py     # 倒排索引
  │               ├── Itemset.py           # 动态配置加载工具
  │               ├── log_with_time.py     # 记录带有时间戳的日志消息
  │               ├── MSFAR.py             # 关联规则挖掘算法实现
  │               └── PrefixTree.py        # 前缀树算法实现
  │               
  ├── resources/             
  │       └── stanza_resources/          # stanza模型文件  
│ │
  ├── README.md                          # 项目说明文档
  └── requirements.txt                   # 依赖文件
        