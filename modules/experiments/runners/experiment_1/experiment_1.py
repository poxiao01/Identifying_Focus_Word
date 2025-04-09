import logging
import random
from typing import Any, List, FrozenSet
from typing import Tuple, Dict

import pandas as pd
import time
from modules.experiments.BaseExperiment import BaseExperiment
from modules.utils.InvertedIndex import InvertedIndex
from modules.utils.MSFAR import MSFAR
from modules.utils.hash import Hasher
from modules.utils.merge_train_test import prepare_validation_sentences

# 类型别名定义
TransactionData = List[List[Tuple[str, str]]]  # 测试集数据类型
Rule = (FrozenSet[str], FrozenSet[str], str)  # 规则类型
RuleSet = set[Rule]  # 规则集合类型


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

    提供多种规则搜索算法实现，用于根据给定的规则集合和句子信息，搜索符合条件的规则并返回匹配规则的置信度列表。
    """

    @staticmethod
    def sequential_search(rules: 'RuleSet', sentence_info: List[Tuple[set, dict]], limit: int = 1) -> (
            list)[list[int | str | float]]:
        """
        顺序搜索匹配规则并返回匹配规则的置信度列表。

        该方法通过遍历所有规则，逐一匹配每个句子的前件集合与规则前件进行匹配，若规则匹配成功且置信度较高，则记录该规则的置信度。

        参数:
            rules (RuleSet): 规则集合，包含每条规则的前件、后件和置信度。
            sentence_info (List[Tuple[set, dict]]): 句子信息，每个句子包含前件集合和后件字典。
            limit (int, optional): 返回规则数量的限制，默认为 1。

        返回:
            List[float]: 每个句子匹配到的规则的置信度列表。
        """
        results = []  # 存储每个句子匹配到的规则的置信度

        # 遍历每个句子
        for idx, (antecedent_set, consequent_dict) in enumerate(sentence_info):
            start_time = time.time()
            max_confidence = 0.0  # 初始化最大置信度
            rule = None  # 初始化规则（此处用于存储匹配的规则）

            # 遍历所有规则
            for antecedent, consequent, confidence in rules:
                consequent = next(iter(consequent))  # 获取后件
                # 检查前件是否全部匹配且后件是否在后件字典中，并且置信度是否更高
                if all(item in antecedent_set for item in antecedent) and consequent in consequent_dict:
                    confidence = float(confidence)
                    if confidence > max_confidence:  # 更新最大置信度和规则
                        max_confidence = confidence
                        rule = (antecedent, consequent, confidence)
            # 将结果添加到列表中
            results.append([idx, 'sequential_search', time.time() - start_time])

        return results

    @staticmethod
    def inverted_index_search(rules: 'RuleSet', sentence_info: List[Tuple[set, dict]], limit: int = 1) \
            -> List[List[str]]:
        """
        使用倒排索引进行规则搜索。

        该方法使用倒排索引对句子信息进行规则查询，快速找到符合条件的规则，并限制返回的规则数量。

        参数:
            rules (RuleSet): 规则集合，包含每条规则的前件、后件和置信度。
            sentence_info (List[Tuple[set, dict]]): 句子信息，每个句子包含前件集合和后件字典。
            limit (int, optional): 返回规则数量的限制，默认为 1。

        返回:
            List[List[str]]: 每个句子匹配到的符合条件的后件词列表。
        """
        results = []  # 存储每个句子的匹配结果

        # 初始化倒排索引
        start_time = time.time()
        index = InvertedIndex(rules)  # 构建倒排索引
        results.append({"倒排索引初始化时间": time.time() - start_time})

        # 遍历每个句子
        for idx, (antecedent_set, consequent_dict) in enumerate(sentence_info):
            start_time = time.time()
            # 使用倒排索引查询单个句子，并返回符合条件的后件
            query_result = index.query_single_sentence_with_limit(antecedent_set, consequent_dict, limit)
            results.append([idx, 'inverted_index_search', time.time() - start_time])

        return results


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
        self.results = []
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
        decimal_places = 10  # 指定保留小数点后的位数

        for support_value in self.support_values:
            for min_confidence in self.min_confidences:
                # 分析规则并获取规则集合
                experiment_rule_set = self.analyze_rules(
                    file_path_xlsx=self.train_data_file_path,
                    support_value=support_value,
                    min_confidence=min_confidence,
                    nrows=5380 - 100  # 选前5280条数据作为训练集
                )
                # 加载测试数据
                validation_sentences = prepare_validation_sentences(self.test_data_file_path)
                validation_sentences = validation_sentences[5281:]  # 选后100条作为测试集

                # 从测试集中随机选择20条数据
                selected_validation_sentences = random.sample(validation_sentences, 20)

                # 进行两种搜索方法的对比
                searcher = RuleSearcher()

                # 获取顺序搜索的结果
                sequential_search_results = searcher.sequential_search(
                    rules=experiment_rule_set, sentence_info=selected_validation_sentences, limit=1
                )
                # 获取倒排索引搜索的结果
                inverted_index_search_results = searcher.inverted_index_search(
                    rules=experiment_rule_set, sentence_info=selected_validation_sentences, limit=1
                )

                # 获取倒排索引初始化时间
                inverted_index_init_time = inverted_index_search_results[0].get("倒排索引初始化时间", 0)
                inverted_index_search_results = inverted_index_search_results[1:]
                # 将倒排索引初始化时间作为第一行存储
                self.results.append({
                    'Support Value': support_value,
                    'Min Confidence': min_confidence,
                    'Sentence Index': 'N/A',  # 倒排索引初始化时间不与特定句子关联
                    'Sequential Search Time': 'N/A',
                    'Inverted Index Search Time': f'倒排索初始化用时{round(inverted_index_init_time, decimal_places)}'
                })

                # 存储其他每个句子的实验结果
                for idx, (sequential_time, inverted_time) in enumerate(
                        zip(sequential_search_results, inverted_index_search_results)):
                    # 提取时长
                    sequential_time_duration = sequential_time[2]  # 提取顺序搜索的时长
                    inverted_time_duration = inverted_time[2]

                    # 存储每个句子的搜索时间
                    self.results.append({
                        'Support Value': support_value,
                        'Min Confidence': min_confidence,
                        'Sentence Index': idx,
                        'Sequential Search Time': round(sequential_time_duration, decimal_places),  # 存储顺序搜索时长
                        'Inverted Index Search Time': round(inverted_time_duration, decimal_places)  # 存储倒排索引时长
                    })
        # 保存实验结果到 Excel
        self.save_results_to_excel()

    def save_results_to_excel(self):
        """保存实验结果到 Excel 文件"""
        df = pd.DataFrame(self.results)

        # 保存为 Excel 文件
        df.to_excel('experiment_1_results.xlsx', index=False)
        print("实验结果已保存到 'experiment_1_results.xlsx' 文件。")

    def teardown(self):
        """清理实验组，释放资源。"""
        print("清理实验组...")


Experiment = Experiment_1(None)
Experiment.setup()
Experiment.run()
