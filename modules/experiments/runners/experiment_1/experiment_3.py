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
Rule = Tuple[FrozenSet[str], FrozenSet[str], str]  # 规则类型
RuleSet = set[Rule]  # 规则集合类型


def convert_rules_to_set(rules: List[dict]) -> RuleSet:
    """
    将规则列表转换为集合，以便后续比较和分析。
    :param rules: 规则列表，每条规则包含 antecedents, consequences, 和 confidence。
    :return: 转换后的规则集合。
    """
    return {
        (frozenset(rule['antecedents']), frozenset(rule['consequences']), str(rule['confidence']))
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
            valid_rules.append({
                'antecedents': antecedents,
                'consequences': consequences,
                'confidence': confidence
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
        max_used_rules = 5

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

    # 定义训练数据所在的目录和结果保存目录
    base_train_dir = '../../../../data/split_data/'
    base_save_dir = '../../../../data/results/'

    # 列出指定目录下所有符合条件的训练数据文件
    # 此处排除了 "train-data-cleand.xlsx"，只选择 train-data-数字.xlsx 文件
    train_files = [
        filename for filename in os.listdir(base_train_dir)
        if filename.startswith('train-data-') and filename.endswith('.xlsx') and filename != 'train-data-cleand.xlsx'
    ]
    # 对文件名进行排序，确保顺序一致
    train_files = sorted(train_files, key=lambda x: int(x.split('-')[-1].split('.')[0]))
    # 遍历每个训练数据文件，分别运行实验
    for train_file in train_files:
        # 构造训练数据文件的完整路径
        train_file_path = os.path.join(base_train_dir, train_file)
        # 构造对应的保存结果文件路径，例如：train-data-1_results.xlsx
        save_file_name = train_file.replace('.xlsx', '_results.xlsx')
        save_file_path = os.path.join(base_save_dir, save_file_name)

        # 构建配置字典（其他配置保持不变）
        config = {
            'train_data_file_path': train_file_path,
            'test_data_file_path': '../../../../data/cleaned/test-data-cleaned.json',
            'test_file_path_json': '../../../../data/raw/test-data.json',
            'save_file_path': save_file_path,
            'min_confidences': [i / 100 for i in range(20, 86, 5)],  # 0.20 - 0.85
            'support_values': list(range(20, 71, 5))  # 20 - 70，每隔5一个
        }

        logging.info(f"运行实验，训练数据文件：{train_file_path}")
        experiment = Experiment(config)
        experiment.setup()
        experiment.run()
        logging.info(f"实验 {train_file} 完成，结果保存到：{save_file_path}")

    logging.info("所有实验已成功完成！")


# 随机去掉5280中的280条问句，剩下5000条，然后随机均等分成10份（每份500条）。最后随机选一份作为数据集1号（500条），
# 然后从剩余9份中不放回地选择一份数据集加入之前的数据集，作为数据集2号（1000条），
# 以此类推，得到数据集3，4，5，6，7，8，9，10号数据集，问句条数依次从500到5000（间隔为500），
# 再最后分别用10个数据集生成规则并在100条问句的测试集上实验，得到10份结果。
if __name__ == "__main__":
    main()
