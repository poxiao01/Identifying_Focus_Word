import logging
import os
import json
import pandas as pd
from typing import Any, List, Dict, Tuple, FrozenSet
import time
from modules.experiments.BaseExperiment import BaseExperiment
from modules.utils.InvertedIndex import InvertedIndex
from modules.utils.MSFAR import MSFAR
from modules.utils.hash import Hasher
from modules.utils.log_with_time import log_with_time
from modules.utils.merge_train_test import prepare_validation_sentences, load_json_data

# 类型别名定义
TransactionData = List[List[Tuple[str, str]]]  # 交易数据类型
Rule = tuple[FrozenSet[str], FrozenSet[str], str, str]  # 规则类型
RuleSet = set[Rule]  # 规则集合类型


def convert_rules_to_set(rules: List[dict]) -> RuleSet:
    """
    将规则列表转换为集合，以便后续比较和分析。
    :param rules: 规则列表，每条规则包含 antecedents, consequences, 和 confidence, support。
    :return: 转换后的规则集合。
    """
    return {
        (frozenset(rule['antecedents']), frozenset(rule['consequences']), str(rule['confidence']), str(rule['support']))
        for rule in rules
    }


def read_excel(file_path: str, nrows: int = None) -> pd.DataFrame:
    """读取Excel文件数据并返回DataFrame。

    Args:
        file_path: Excel文件路径
        nrows: 限制读取的行数，默认为None表示读取所有行

    Returns:
        DataFrame: 读取的数据框
    """
    try:
        logging.info(f"读取Excel文件：{file_path} (nrows={nrows})")
        return pd.read_excel(file_path, nrows=nrows)
    except Exception as e:
        logging.error(f"读取Excel文件失败：{e}")
        raise


def convert_dataframe_to_transactions(df: pd.DataFrame) -> List[List[Tuple]]:
    """将DataFrame转换为交易记录格式。

    Args:
        df: 输入数据框

    Returns:
        List[List[Tuple]]: 交易记录列表
    """
    transaction_data = [
        [(col_name, str(row[col_name])) for col_name in df.columns if pd.notna(row[col_name])]
        for _, row in df.iterrows()
    ]
    return transaction_data


class Experiment(BaseExperiment):
    """实验类，用于执行关联规则分析实验。"""

    def __init__(self, config: Any):
        super().__init__(config)
        self.results = []
        # 直接从配置中获取所有参数
        self.train_data_file_path = config.get('train_data_file_path')
        self.test_data_file_path = config.get('test_data_file_path')
        self.test_file_path_json = config.get('test_file_path_json')
        self.save_file_path = config.get('save_file_path')
        self.min_confidences = config.get('min_confidences')
        self.support_values = config.get('support_values')

    def setup(self):
        """初始化实验配置，直接使用外部传入的配置，不进行特殊处理。"""
        logging.info("设置实验配置...")
        logging.info(f"支持度范围: {self.support_values}")
        logging.info(f"置信度范围: {self.min_confidences}")
        logging.info(f"训练数据文件: {self.train_data_file_path}")
        logging.info(f"测试数据文件: {self.test_data_file_path}")
        logging.info(f"测试JSON文件: {self.test_file_path_json}")
        logging.info(f"结果保存路径: {self.save_file_path}")

    def run_analysis(self, file_path: str, nrows: int = None, support_value: float = 10,
                     min_confidence: float = 0.8) -> List[Dict]:
        """运行规则生成分析，并返回符合条件的规则。

        Args:
            file_path: 输入Excel文件路径
            nrows: 限制读取的行数，默认为None表示读取所有行
            support_value: 支持度阈值，默认为10
            min_confidence: 最小置信度，默认为0.8

        Returns:
            List[Dict]: 符合条件的规则列表
        """
        df = read_excel(file_path, nrows)
        transaction_data = convert_dataframe_to_transactions(df)

        hasher = Hasher(3)
        transaction_data = self._hash_transactions(transaction_data, hasher)

        msfar = MSFAR(min_support=support_value, min_confidence=min_confidence)
        msfar.initialize_prefix_trees(transaction_data)
        msfar.prune_trees()

        rules = msfar.generate_rules()
        return self._extract_valid_rules(rules, hasher)

    def _hash_transactions(self, transaction_data: List[List[Tuple]], hasher: Hasher) -> List[List[Tuple]]:
        """对交易数据进行哈希处理。

        Args:
            transaction_data: 原始交易数据
            hasher: 哈希处理工具

        Returns:
            List[List[Tuple]]: 哈希后的交易数据
        """
        for i in range(len(transaction_data)):
            for j in range(len(transaction_data[i])):
                transaction_data[i][j] = hasher.hash((str(transaction_data[i][j][0]), str(transaction_data[i][j][1])))
        return transaction_data

    def _extract_valid_rules(self, rules: List[dict], hasher: Hasher) -> List[Dict]:
        """从生成的规则中提取符合条件的规则，并进行解码。

        Args:
            rules: 原始生成的规则列表
            hasher: 哈希处理工具

        Returns:
            List[Dict]: 解码后的有效规则列表
        """
        valid_rules = []
        for rule in rules:
            antecedents = frozenset([str(hasher.decode_hashed_item(x)) for x in rule['condition']])
            consequences = frozenset([str(hasher.decode_hashed_item(rule['decision']))])
            confidence = f"{rule['confidence']:.10f}"
            support = f"{rule['support']:.10f}"
            valid_rules.append({
                'antecedents': antecedents,
                'consequences': consequences,
                'confidence': confidence,
                'support': support
            })
        return valid_rules

    def compute_accuracy(self, result: List[List[str]], question_words: List[str], max_used_rules: int) -> dict:
        """计算预测结果的准确率。

        Args:
            result: 预测结果
            question_words: 正确答案
            max_used_rules: 最大使用规则数

        Returns:
            Dict: 各规则准确率统计
        """
        accuracy_stats = {i: {'total': 0, 'correct': 0} for i in range(1, max_used_rules + 1)}

        # 补充短行数据
        for i, row in enumerate(result):
            if len(row) < max_used_rules:
                result[i].extend([""] * (max_used_rules - len(row)))

        for i, predicted_words_list in enumerate(result):
            correct = 0
            for j in range(max_used_rules):
                if predicted_words_list[j] == question_words[i]:
                    correct = 1
                accuracy_stats[j + 1]['total'] += 1
                if correct:
                    accuracy_stats[j + 1]['correct'] += 1

        for idx in accuracy_stats:
            accuracy_stats[idx]['accuracy'] = accuracy_stats[idx]['correct'] / accuracy_stats[idx]['total']

        return accuracy_stats

    def run(self):
        """执行实验并记录结果。"""
        max_used_rules = 1

        for support_value in self.support_values:
            for min_confidence in self.min_confidences:
                experiment_rule_set = convert_rules_to_set(self.run_analysis(file_path=self.train_data_file_path,
                                                                             support_value=support_value,
                                                                             min_confidence=min_confidence, nrows=None))
                validation_sentences = prepare_validation_sentences(self.test_data_file_path)
                test_data = load_json_data(self.test_file_path_json)
                question_words = [item['question_word'] for item in test_data]

                inverted_index = InvertedIndex(experiment_rule_set)
                result_limit = inverted_index.query_rules_by_sentences_with_limit(validation_sentences, max_used_rules)
                accuracy_stats = self.compute_accuracy(result_limit, question_words, max_used_rules)
                self.results.append({
                    'support_value': support_value,
                    'min_confidence': min_confidence,
                    'accuracy_stats': accuracy_stats
                })
        self.save_results_to_excel()

    def save_results_to_excel(self):
        """将实验结果保存到Excel文件中。"""
        formatted_results = []
        experiment_id = 1

        for result in self.results:
            support_value = result['support_value']
            min_confidence = result['min_confidence']
            accuracy_stats = result['accuracy_stats']

            for i in range(1, len(accuracy_stats) + 1):
                stats = accuracy_stats[i]
                formatted_results.append({
                    '实验 ID (experiment_id)': experiment_id,
                    '支持度 (support_value)': support_value,
                    '置信度 (min_confidence)': min_confidence,
                    '使用前 i 条后件不同的规则': i,
                    'total': stats['total'],
                    'correct': stats['correct'],
                    'accuracy': round(stats['accuracy'], 10)
                })
            experiment_id += 1

        df = pd.DataFrame(formatted_results)
        df.to_excel(self.save_file_path, index=False)
        logging.info(f"实验结果已保存到 {self.save_file_path} 文件。")


def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    # 实验配置
    config = {
        'train_data_file_path': '../../../../data/cleaned/train-data-cleand.xlsx',
        'test_data_file_path': '../../../../data/cleaned/test-data-cleaned.json',
        'test_file_path_json': '../../../../data/raw/test-data.json',
        'save_file_path': '../../../../data/results/experiment_results.xlsx',
        'min_confidences': [i / 100 for i in range(5, 51, 5)],  # 0.20 - 0.85
        'support_values': list(range(5, 51, 5))  # 20 - 70，每隔5一个
    }

    # 初始化实验类
    experiment = Experiment(config)

    # 设置实验配置
    experiment.setup()

    # 运行实验
    experiment.run()

    logging.info("实验已成功完成！")


# 自己的测试实验
# 两个数据集分别实验、数据集1-10的实验，两个实验，你重新跑一下我们的前缀树算法，
# 支持度从5-30（间隔为5），置信度从0.05-0.3（间隔为0.05）
if __name__ == "__main__":
    main()
