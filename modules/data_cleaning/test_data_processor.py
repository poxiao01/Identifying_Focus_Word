import os
from typing import List, Dict, Any, Tuple, Set
import json
import string
from collections import defaultdict
from typing import List, Tuple

from modules.data_cleaning.dependency_analyzer import find_structure_word, map_word_positions_to_relations, \
    ShortestPathFinder

"""
模块名称: test_data_processor
功能描述: 用于生成测试数据，包含从JSON读取数据、调用依赖分析、处理并保存数据。
"""


import stanza

class DependencyAnalyzerUseWhenVerifying:
    def __init__(self, model_dir: str, _sentences_list: List[str]):
        """
        初始化 DependencyAnalyzer 类。

        参数:
        - model_dir (str): Stanza NLP 模型的自定义目录位置。
        - sentences_list (List[str]): 句子列表。
        - question_word_list (List[str]): 对应每个句子的疑问词列表。
        """
        self.model_dir = model_dir  # 目录
        self.nlp = None
        self.all_words_dependencies_list = []  # 所有句子的单词的依赖结构
        self.all_words_pos_list = []  # 所有句子的单词及其词性
        self.structure_words_and_pos_list = []  # 所有句子的句型词及其词性
        self.sentences_list = _sentences_list  # 数据集，包含全部的句子

        self.initialize()  # 初始化操作

    def initialize(self):
        """
        初始化 Stanza 的 NLP Pipeline。

        在使用其他方法之前，必须调用此方法初始化 NLP Pipeline。
        """
        try:
            self.nlp = stanza.Pipeline('en',
                                       processors='tokenize,pos,lemma,depparse', use_gpu=True,
                                       dir='../../resources/stanza_resources', download_method=None)
        except Exception as e:
            print(f"初始化 NLP Pipeline 失败: {e}")
            raise

        self._process_sentences()

    def _process_sentences(self):
        """
        处理所有句子，提取依赖关系、词性和句型词。
        """
        for index_, sentence_ in enumerate(self.sentences_list):
            # 去除末尾的标点符号（仅在末尾是标点符号时进行去除）
            sentence_ = sentence_.strip()
            if sentence_ and sentence_[-1] in string.punctuation:
                sentence_ = sentence_[:-1]
            doc = self.nlp(sentence_)
            sentence_dependencies = []
            sentence_words_pos = []

            counts_dict = dict()
            for item in doc.sentences:
                for word in item.words:
                    head_word = item.words[word.head - 1].text if word.head > 0 else word.text
                    if word.deprel not in counts_dict:
                        counts_dict[word.deprel] = 0
                    else:
                        counts_dict[word.deprel] += 1
                    sentence_dependencies.append((f'{word.deprel}_{counts_dict[word.deprel]}', [head_word, word.text]))
                    sentence_words_pos.append((word.text, word.xpos, word.upos))
            self.all_words_dependencies_list.append(sentence_dependencies)
            self.all_words_pos_list.append(sentence_words_pos)

            structure_word_and_pos = find_structure_word(doc.sentences)
            self.structure_words_and_pos_list.append(structure_word_and_pos)

    def get_all_words_dependencies(self) -> List[List[Tuple[str, List[str]]]]:
        """
        获取所有句子的单词依赖结构。

        返回:
        List[List[Tuple[str, List[str]]]]: 所有句子的依赖关系列表。
        """
        return self.all_words_dependencies_list

    def get_all_words_pos(self) -> List[List[Tuple[str, str, str]]]:
        """
        获取所有句子的单词及其词性。

        返回:
        List[List[Tuple[str, str, str]]]: 所有单词及其词性信息。
        """
        return self.all_words_pos_list

    def get_structure_words_and_pos(self, by_index: int = None) -> List[Tuple[str, str, str]]:
        """
        获取句型词及其词性。

        Args:
        - by_index (int, optional): 要获取的结构词的索引位置。如果为None，则返回所有句型词及其词性信息。默认为None。

        Returns:
        List[Tuple[str, str, str]]: 句型词及其词性信息。
        """
        structure_words_and_pos_list = []

        if by_index is not None:
            word, pos = self.structure_words_and_pos_list[by_index]
            if pos == 'VB':
                structure_words_and_pos_list.append(('VB', pos, word))  # 替换为 'VB'
            else:
                structure_words_and_pos_list.append((word, pos, word))  # 保留原有词和词性
        else:
            for word, pos in self.structure_words_and_pos_list:
                if pos == 'VB':
                    structure_words_and_pos_list.append(('VB', pos, word))  # 替换为 'VB'
                else:
                    structure_words_and_pos_list.append((word, pos, word))  # 保留原有词和词性

        return structure_words_and_pos_list

    def find_same_dependency(self, idx, question_word):
        """
        查找句型词与问题词之间的直接依存关系。

        Args:
        - idx (int): 要检查的索引位置

        Returns:
        - list of tuples or None: 如果找到依存关系，则返回包含依存关系的元组列表；
          如果未找到任何依存关系，则返回None。
          每个元组的格式为 ('SAME_DEPENDENCY', rel_type)，其中rel_type表示依存关系的类型。
        """
        dependency_list = []
        # 遍历依存关系列表，检查句型词与问题词之间是否存在直接依存关系
        for rel, (head_word, word) in self.all_words_dependencies_list[idx]:
            rel = rel.upper().replace(':', '_')
            # 检查两种情况：问题词是否依赖于句型词，或者句型词是否依赖于问题词
            if question_word == head_word and self.structure_words_and_pos_list[idx][0] == word:
                dependency_list.append(('SAME_DEPENDENCY', rel + '_1'))
            if question_word == word and self.structure_words_and_pos_list[idx][0] == head_word:
                dependency_list.append(('SAME_DEPENDENCY', rel + '_2'))

        if not dependency_list:
            return None
        return dependency_list

    def get_structure_words_in_dependencies_position(self) -> List[List[Tuple[str, str]]]:
        """
        获取句型词在依赖结构的位置。

        返回:
        List[List[Tuple[str, int]]]: 句型词在依赖树中的位置标记列表。
        """
        structure_words_in_dependencies_position = []
        for __index, dependencies in enumerate(self.all_words_dependencies_list):
            structure_words_in_dependencies_position.append(map_word_positions_to_relations(
                self.structure_words_and_pos_list[__index][0], dependencies, 'SENTENCE_'))
        return structure_words_in_dependencies_position

    def retrieve_all_information(self):
        result_list = []
        for __index, sentence in enumerate(self.sentences_list):
            #  前件集合
            antecedents_set = set()
            #  后件字典
            consequences_dict = defaultdict(set)

            # 句子类型(疑问句 or 陈述句)
            __str = '疑问句' if self.sentences_list[__index][-1] == '?' else '陈述句'
            antecedents_set.add(('SENTENCE_PATTERN', __str))

            # # 句型词
            # antecedents_set.add(('SENTENCE_STRUCTURE_WORD', self.structure_words_and_pos_list[__index][0]))
            # # 特殊处理疑问句的句型词
            # auxiliary_set = {'who', 'what', 'when', 'which', 'how', 'where', 'whose', 'how many'}
            # if (self.structure_words_and_pos_list[__index][0].lower() in auxiliary_set
            #         and __str == '疑问句'):
            #     antecedents_set.add(('SENTENCE_STRUCTURE_WORD', self.structure_words_and_pos_list[__index][0].lower()))
            # else:
            #     antecedents_set.add(('SENTENCE_STRUCTURE_WORD', self.structure_words_and_pos_list[__index][0]))

            # 特殊处理陈述句的句型词（VB）
            auxiliary_set = {'who', 'what', 'when', 'which', 'how', 'where', 'whose', 'how many'}
            if ('V' in self.structure_words_and_pos_list[__index][1]
                    and self.structure_words_and_pos_list[__index][
                        0].lower() not in auxiliary_set and __str == '陈述句'):
                antecedents_set.add(('SENTENCE_STRUCTURE_WORD', 'VB'))
            else:
                antecedents_set.add(('SENTENCE_STRUCTURE_WORD', self.structure_words_and_pos_list[__index][0]))

            # 句型词词性
            antecedents_set.add(('SENTENCE_STRUCTURE_WORD_POS', self.structure_words_and_pos_list[__index][1]))

            # 句型词在依赖关系中的位置
            __structure_word = self.structure_words_and_pos_list[__index][0]
            if __structure_word == 'How many':
                __structure_word = 'How'
            for x in map_word_positions_to_relations(__structure_word,
                                                     self.all_words_dependencies_list[__index], 'SENTENCE_'):
                antecedents_set.add(x)

            for question_word, question_xpos, __ in self.all_words_pos_list[__index]:
                # 疑问词与句型词相同
                if question_word == self.structure_words_and_pos_list[__index][0]:
                    consequences_dict[('SAME_QS_WORD', 'True')].add(question_word)

                # 是否同依赖
                _ = self.find_same_dependency(__index, question_word)
                if _ is not None:
                    consequences_dict[tuple(_)].add(question_word)

                # 依赖路径
                dependency_resolver = ShortestPathFinder(
                    question_word=question_word,  # 当前句子的问题词
                    structure_word=self.structure_words_and_pos_list[__index][0],  # 当前句子的结构词
                    dependency_relations=self.all_words_dependencies_list[__index],  # 当前句子的依赖关系列表
                    _sentence=self.sentences_list[__index]  # 当前句子文本
                )
                sentence_dependency_paths = dependency_resolver.get_dependency_paths(must_find=False)
                if sentence_dependency_paths:
                    for sentence_dependency_path in sentence_dependency_paths:
                        consequences_dict[('DEPENDENCY_PATH', sentence_dependency_path)].add(question_word)

                # 疑问词
                consequences_dict[('QUESTION_WORD', question_word)].add(question_word)

                # 疑问词词性
                consequences_dict[('QUESTION_WORD_POS', question_xpos)].add(question_word)

                for x in map_word_positions_to_relations(question_word, self.all_words_dependencies_list[__index],
                                                         'QUESTION_'):
                    consequences_dict[x].add(question_word)
            antecedents_set = {str(item) for item in antecedents_set}
            consequences_dict = {str(key): value for key, value in consequences_dict.items()}
            result_list.append((antecedents_set, consequences_dict))
        return result_list


def get_full_sentences_information(sentences: List, model_dir: str) -> list[tuple[set, dict]]:
    test = DependencyAnalyzerUseWhenVerifying(model_dir=model_dir, _sentences_list=sentences)
    data = test.retrieve_all_information()
    return data


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


class DataProcessor:
    """
    工具类：用于处理 JSON 数据文件，提取句子并转换为可序列化的格式。
    """

    def __init__(self, input_file_path: str, model_dir: str, output_file_path = None):
        """
        初始化数据处理器
        :param input_file_path: 输入 JSON 文件路径
        :param model_dir: Stanza 模型路径
        :param output_file_path: 输出 JSON 文件路径
        """
        self.input_file_path = input_file_path
        self.model_dir = model_dir
        if output_file_path is None:
            self.output_file_path = self._generate_output_path(input_file_path)
        else:
            self.output_file_path = output_file_path

    @staticmethod
    def _generate_output_path(input_path: str) -> str:
        """
        根据输入文件路径生成输出文件路径，自动添加 `_processed` 后缀。
        :param input_path: 输入文件路径
        :return: 自动生成的输出文件路径
        """
        base, ext = os.path.splitext(input_path)
        return f"{base}_processed{ext}"

    @staticmethod
    def read_json_file(file_path: str) -> List[Dict[str, Any]]:
        """
        从 JSON 文件中读取数据
        :param file_path: JSON 文件路径
        :return: 数据列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"文件未找到: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"无法解析 JSON 文件: {file_path}")

    @staticmethod
    def write_json_file(file_path: str, data: List[Dict[str, Any]]) -> None:
        """
        将数据写入到 JSON 文件
        :param file_path: JSON 文件路径
        :param data: 数据列表
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except IOError as e:
            raise IOError(f"写入文件失败: {file_path}. 原因: {str(e)}")

    @staticmethod
    def make_json_serializable(data: Any) -> Any:
        """
        递归将数据中的不可序列化类型转换为 JSON 支持的类型。
        :param data: 输入数据
        :return: 可序列化的数据
        """
        if isinstance(data, frozenset):
            return list(data)  # 将 frozenset 转为 list
        elif isinstance(data, set):
            return list(data)  # 将 set 转为 list
        elif isinstance(data, dict):
            # 将字典的键和值递归处理
            return {str(key): DataProcessor.make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, tuple):
            return tuple(DataProcessor.make_json_serializable(item) for item in data)  # 递归处理 tuple
        elif isinstance(data, list):
            return [DataProcessor.make_json_serializable(item) for item in data]  # 递归处理 list
        else:
            return data  # 对于其他类型，直接返回

    def process_sentences(self, sentences: List[str]) -> List[Tuple[set, dict]]:
        """
        调用外部函数处理句子
        :param sentences: 输入句子列表
        :return: 处理后的句子信息
        """
        # 使用外部函数 get_full_sentences_information
        processed_sentences = get_full_sentences_information(sentences, self.model_dir)
        return processed_sentences

    def add_ids_to_processed_data(self, processed_sentences: List[Tuple[set, dict]]) -> List[Dict[str, Any]]:
        """
        给处理后的数据添加唯一 ID
        :param processed_sentences: 处理后的句子信息
        :return: 添加了 ID 的数据列表
        """
        return [
            {
                "id": idx + 1,
                "data": {
                    "set": list(sentence_data[0]),  # 确保 set 已转为 list
                    "dict": self.make_json_serializable(sentence_data[1])  # 确保 dict 中的 frozenset 被转换
                }
            }
            for idx, sentence_data in enumerate(processed_sentences)
        ]

    def process_and_save(self) -> None:
        """
        主流程：读取数据 -> 处理数据 -> 添加 ID -> 保存结果
        """
        # 1. 读取输入文件中的数据
        print(f"正在读取文件: {self.input_file_path}")
        data = self.read_json_file(self.input_file_path)

        # 2. 提取句子
        sentences = [item['sentence'] for item in data]

        # 3. 处理句子
        print("正在处理句子...")
        processed_sentences = self.process_sentences(sentences)

        # 4. 给数据添加唯一 ID
        print("正在为数据添加 ID...")
        processed_data = self.add_ids_to_processed_data(processed_sentences)

        # 5. 保存到输出文件
        print(f"正在将结果保存到文件: {self.output_file_path}")
        self.write_json_file(self.output_file_path, processed_data)
        print("数据处理完成！")


# # 示例调用
#
# if __name__ == "__main__":
#     # 输入文件路径
#     input_file = "../../data/raw/All-data.json"
#     model_dir = "../../resources/stanza_resources"
#     # 创建数据处理器实例
#     processor = DataProcessor(input_file, model_dir)
#
#     # 执行数据处理和保存
#     processor.process_and_save()
