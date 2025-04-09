import logging
from typing import Any, List, Set
from typing import Tuple, Dict
import json
import pandas as pd


class JSONDataReader:
    @staticmethod
    def read_data(file_path: str) -> List[Tuple[Set[str], Dict[str, List[str]]]]:
        """
        从JSON文件中读取多个字典的`data`字段，并提取`set`和`dict`内容。

        :param file_path: JSON文件的路径。
        :return: 一个列表，每个元素是一个元组，包含一个set和一个dict。
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)

            # 验证 JSON 数据是否是列表
            if not isinstance(json_data, list):
                raise ValueError("JSON格式错误：数据不是列表。")

            result = []
            for item in json_data:
                # 确保每个元素是字典且包含 `data`
                if not isinstance(item, dict) or "data" not in item:
                    print(f"跳过无效项：{item}")
                    continue

                data = item["data"]
                if not isinstance(data, dict) or "set" not in data or "dict" not in data:
                    print(f"跳过无效data项：{data}")
                    continue

                # 提取 `set` 和 `dict`
                set_data = set(data["set"]) if isinstance(data["set"], list) else set()
                dict_data = data["dict"] if isinstance(data["dict"], dict) else {}

                # 添加到结果
                result.append((set_data, dict_data))

            return result

        except (json.JSONDecodeError, ValueError) as e:
            print(f"读取JSON文件时出错：{e}")
            return []


def prepare_validation_sentences(file_path_processed: str) -> List[Tuple[set, dict]]:
    """
    从处理后的 JSON 文件中读取验证句子。
    :param file_path_processed: 处理后的 JSON 文件路径。
    :return: 验证句子列表。
    """
    reader = JSONDataReader()
    return reader.read_data(file_path_processed)


def load_json_data(file_path: str) -> List[dict]:
    """
    从 JSON 文件中加载数据。
    :param file_path: JSON 文件路径。
    :return: 加载后的数据列表。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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


def merge_train_test(train_data_xlsx_path, test_data_json_path, sentences_json_path):
    """
    合并训练数据（train-data）和测试数据（test-data）的函数。

    """
    df = read_excel_data(train_data_xlsx_path, nrows=None)
    transaction_data = convert_to_transactions(df)

    test_data = load_json_data(sentences_json_path)
    question_words = [item['question_word'] for item in test_data]
    validation_sentences = prepare_validation_sentences(test_data_json_path)
    # # 创建一个字典，将训练数据和测试数据以键值对形式存储
    # combined_data = {
    #     "train": train_data,  # 将训练数据赋值给键 "train"
    #     "test": test_data     # 将测试数据赋值给键 "test"
    # }
    # return combined_data  # 返回合并后的字典


if __name__ == "__main__":
    # 配置路径
    train_data_xlsx_path = "../../data/cleaned/Train-data-cleand.xlsx"
    test_data_json_path = "../../data/cleaned/Test-data-cleaned.json"
    merge_train_test(train_data_xlsx_path=train_data_xlsx_path, test_data_json_path=test_data_json_path)
