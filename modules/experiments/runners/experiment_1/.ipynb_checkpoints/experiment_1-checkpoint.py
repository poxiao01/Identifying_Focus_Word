import logging
from typing import Any, List, FrozenSet
from typing import Tuple, Dict

import pandas as pd

from modules.experiments.BaseExperiment import BaseExperiment
from modules.utils.InvertedIndex import InvertedIndex
from modules.utils.MSFAR import MSFAR
from modules.utils.hash import Hasher

# 类型别名定义
TransactionData = List[List[Tuple[str, str]]]  # 交易数据类型
Rule = Dict[str, FrozenSet[str]]  # 规则类型
RuleSet = List[Rule]  # 规则集合类型


def print_first_element_type(container, level=0):
    # 获取容器的类型
    container_type = type(container)

    # 缩进控制，增加层级使得输出结构更清晰
    indent = "  " * level

    # 如果容器是集合、列表或元组
    if isinstance(container, (set, list, tuple)):
        print(f"{indent}{container_type}(")
        if len(container) == 0:
            print(f"{indent}  )")
        else:
            first_item = next(iter(container))  # 获取第一个元素
            if isinstance(first_item, (set, list, tuple, dict)):
                print_first_element_type(first_item, level + 1)  # 递归打印第一个元素的类型
            else:
                print(f"{indent}  {type(first_item)}")
        print(f"{indent})")

    # 如果容器是字典，打印出键和值的类型
    elif isinstance(container, dict):
        print(f"{indent}{container_type}(")
        if len(container) == 0:
            print(f"{indent}  )")
        else:
            first_key, first_value = next(iter(container.items()))  # 获取第一个键值对
            print(f"{indent}  Key: {type(first_key)}, Value: {type(first_value)}")
            if isinstance(first_key, (set, list, tuple, dict)):
                print_first_element_type(first_key, level + 1)  # 递归打印字典键的类型
            if isinstance(first_value, (set, list, tuple, dict)):
                print_first_element_type(first_value, level + 1)  # 递归打印字典值的类型
        print(f"{indent})")

    else:
        print(f"{indent}{container_type}")


def rules_to_set(rules: List[dict]) -> set:
    """
    将规则列表转换为集合，方便比较。
    :param rules: 规则列表，每条规则包含 antecedents, consequences, 和 confidence。
    :return: 转换后的规则集合。
    """
    return {
        (frozenset(rule['antecedents']), frozenset(rule['consequences']), str(rule['confidence']))
        for rule in rules
    }

def read_excel_data(file_path: str, nrows: int = None) -> pd.DataFrame:
    """读取Excel文件数据

    Args:
        file_path: Excel文件路径
        nrows: 限制读取的行数，默认为None表示读取所有行
    Returns:
        DataFrame: 读取的数据框
    """
    try:
        logging.info(f"Reading Excel file from: {file_path} (nrows={nrows})")
        return pd.read_excel(file_path, nrows=nrows)
    except Exception as e:
        logging.error(f"Error reading Excel file: {e}")
        raise


def convert_to_transactions(df: pd.DataFrame) -> List[List[Tuple]]:
    """将DataFrame转换为交易记录格式

    Args:
        df: 输入数据框

    Returns:
        List[List[Tuple]]: 交易记录列表
    """
    transaction_data = []
    for _, row in df.iterrows():
        # 使用列表推导式优化
        row_data = [(col_name, str(row[col_name]))
                    for col_name in df.columns
                    if pd.notna(row[col_name])]
        transaction_data.append(row_data)
    return transaction_data

class RuleSearcher:
    """
    规则搜索器类

    实现了多种规则搜索算法。
    """

    @staticmethod
    def sequential_search(rules: RuleSet,
                          sentence_info: List[str],
                          limit: int = 1) -> List[float]:
        """
        顺序搜索匹配规则

        参数:
            rules: 规则集合
            sentence_info: 处理后的句子信息
            limit: 返回规则数量限制

        返回:
            匹配规则的置信度列表
        """
        max_confidence = 0.0

        for antecedent, consequent, confidence in rules:
            if (all(item in sentence_info for item in antecedent) and
                    all(item in sentence_info for item in consequent)):
                max_confidence = max(max_confidence, float(confidence))

        return [max_confidence]

    @staticmethod
    def inverted_index_search(rules: RuleSet,
                              sentence_info: List[tuple[set, dict]],
                              limit: int = 1) -> List[List]:
        """
        使用倒排索引进行规则搜索

        参数:
            rules: 规则集合
            sentence_info: 处理后的句子信息
            limit: 返回规则数量限制

        返回:
            匹配规则的置信度列表
        """
        index = InvertedIndex(rules)
        return index.query_rules_by_sentences_with_limit(sentence_info, limit)

class Experiment_1(BaseExperiment):
    """
    模块名称: Experiment_1

    功能描述: 
    本模块旨在获取倒排索引和顺序查找两种方法在查找规则时的运行时间，并进行对比分析。
    通过随机选择20条问句，计算每条问句在不同支持度阈值和置信度条件下的查找时间，
    以评估两种方法的效率。

    设计目的: 
    在问句预测疑问词的过程中，查找规则是一个关键步骤。为了优化这一过程，
    本实验对比了倒排索引和顺序查找两种方法在不同支持度阈值和置信度条件下的性能。
    通过实验，我们希望能够确定哪种方法在特定条件下能够更快地找到最高置信度的规则，
    从而为系统优化提供数据支持。

    实验设置:
    - 支持度阈值范围: 20-70，间隔为5
    - 置信度范围: 0.2-0.85，间隔为0.05
    - 问句数量: 随机选择20条问句
    - 对比指标: 每条问句在两种查找方法下的运行时间

    预期输出: 
    实验将生成一个包含每条问句在不同支持度阈值和置信度条件下，
    倒排索引和顺序查找的运行时间对比的报告。该报告将帮助开发者理解两种方法
    在不同条件下的性能差异，并为后续的系统优化提供依据。
    """

    def __init__(self, config):
        super().__init__(config)
        self.train_data_file_path = None
        self.test_data_file_path = None
        self.min_confidences = None
        self.support_values = None

    def setup(self):
        """设置实验组，初始化必要的变量和资源。"""
        print("设置实验组...")
        self.support_values = list(range(20, 71, 5))  # 示例支持度阈值范围
        self.min_confidences = [i / 100 for i in range(20, 86, 5)]  # 示例置信度范围 0.20-0.85，步长 0.05
        self.train_data_file_path = '../../../../data/cleaned/Train-data-cleand.xlsx'
        self.test_data_file_path = '../../../../data/cleaned/Test-data-cleaned.json'



    def run_analysis(self, file_path: str,
                     nrows: int = None,
                     support_value: float = 10,
                     min_confidence: float = 0.8) -> List[Dict]:
        """运行关联规则分析，并返回符合条件的规则

        Args:
            file_path: Excel 文件路径
            nrows: 限制读取的行数，默认为 None 表示读取所有行
            support_value: 支持度阈值，默认为 10
            min_confidence: 最小置信度，默认为 0.8

        Returns:
            List[Dict]: 符合条件的规则列表
        """

        df = read_excel_data(file_path, nrows=nrows)
        transaction_data = convert_to_transactions(df)

        hasher = Hasher(3)

        for i in range(len(transaction_data)):
            for j in range(len(transaction_data[i])):
                transaction_data[i][j] = hasher.hash((str(transaction_data[i][j][0]), str(transaction_data[i][j][1])))

        msfar = MSFAR(min_support=support_value, min_confidence=min_confidence)
        msfar.initialize_prefix_trees(transaction_data)

        msfar.prune_trees()
        rules = msfar.generate_rules()
        valid_rules = []  # 用于存储符合条件的规则
        for rule in rules:
            antecedents = frozenset([str(hasher.decode_hashed_item(x)) for x in rule['condition']])
            consequences = frozenset([str(hasher.decode_hashed_item(rule['decision']))])
            confidence = f"{rule['confidence']:.10f}"
            valid_rules.append({
                'antecedents': antecedents,
                'consequences': consequences,
                'confidence': confidence
            })

        return valid_rules

    def analyze_rules(self, file_path_xlsx: str, support_value: int, min_confidence: float, nrows: Any = None) -> set:
        """
        分析规则并生成规则集合。
        :param file_path_xlsx: 输入 Excel 文件路径。
        :param support_value: 支持度阈值。
        :param min_confidence: 置信度阈值。
        :param nrows: 读取的行数，None 表示全部。
        :return: 规则集合。
        """
        valid_rules = self.run_analysis(
            file_path_xlsx, nrows=nrows, support_value=support_value, min_confidence=min_confidence
        )
        return rules_to_set(valid_rules)
    def run(self):
        """运行实验组，执行实验逻辑并记录。"""
        for support_value in self.support_values:
            for min_confidence in self.min_confidences:
                experiment_rule_set = self.analyze_rules(
                    file_path_xlsx=self.train_data_file_path,
                    support_value = support_value,
                    min_confidence = min_confidence
                )
                print(type(experiment_rule_set))
                print(experiment_rule_set)
                # print(print_first_element_type(experiment_rule_set))
                exit(1)
                # # 1.使用倒排索引
                # # 比较搜索方法
                # searcher = RuleSearcher()
                #
                # # 2.顺序查找
    def teardown(self):
        """清理实验组，释放资源。"""
        print("清理实验组...")


Experiment = Experiment_1(None)
Experiment.setup()
Experiment.run()

