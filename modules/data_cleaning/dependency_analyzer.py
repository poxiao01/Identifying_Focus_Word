import string
from typing import List, Tuple, Dict, Set, Optional
import re
import json
import stanza
from stanza.models.common.doc import Sentence
import pandas as pd

"""
模块名称: dependency_analyzer
功能描述: 提取句子相关信息，作为实验的输入数据。
设计目的: 提供统一的接口，确保不同类型的清洗器可以动态调用。
"""


class ShortestPathFinder:
    """
    该类用于解析句子中的依存关系，寻找从句型词到问题词的最短依赖路径。
    """

    def __init__(self, question_word: str, structure_word: str, dependency_relations: List[Tuple[str, Tuple[str, str]]],
                 _sentence: str):
        """
        初始化依赖关系解析器。

        :param question_word: 问题词
        :param structure_word: 句型词
        :param dependency_relations: 句子的依存关系列表
        :param _sentence: 完整的句子文本
        """
        if structure_word == 'How many' or structure_word == 'how many':
            structure_word = 'How'
            if structure_word not in _sentence:
                if 'how' in _sentence:
                    structure_word = 'how'
                else:
                    print(f'{_sentence}\n{question_word} {structure_word} {dependency_relations}')
                    raise f'错误！句型词未找到！{question_word} {structure_word} {dependency_relations}'

        self.question_word = question_word  # 疑问词
        self.structure_word = structure_word  # 句型词
        self.dependency_relations = dependency_relations  # 依赖关系
        self.sentence = _sentence  # 句子
        self.edge_dict = self._construct_directed_graph()  # 构建有向图
        self.dependency_paths_list = []  # 存储找到的依赖路径

    def _construct_directed_graph(self) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        根据依存关系构建有向图。

        :return: 有向图字典，键为节点（词语），值为（目标节点，关系，方向）的列表
        """
        edge_dict: Dict[str, List[Tuple[str, str, str]]] = {}
        for relation, (head_word, word) in self.dependency_relations:
            edge_dict.setdefault(head_word, []).append((word, relation, '-->'))
            edge_dict.setdefault(word, []).append((head_word, relation, '<--'))
        return edge_dict

    def _find_shortest_dependency_paths(self, must_find=True) -> None:
        """
        从结构词到问题词查找所有最短的依赖路径。
        """
        visited: Set[str] = set()
        visited.add(self.structure_word)
        self._dfs(self.structure_word, self.question_word, [], visited)
        visited.remove(self.structure_word)
        # 如果有多条最短路径，保留它们；否则，保持当前找到的路径
        if self.dependency_paths_list:
            min_path_length = min(len(path) for path in self.dependency_paths_list)
            self.dependency_paths_list = [path for path in self.dependency_paths_list if len(path) == min_path_length]
            # 转换路径表示形式
            self.dependency_paths_list = [' --> '.join(path) for path in self.dependency_paths_list]

            # if len(self.dependency_paths_list) > 1:
            #     print(f'最短路不唯一！{self.sentence}\n')
        elif must_find:
            raise Exception(f'未找到路径：{self.sentence}\n句型词:{self.structure_word}, 疑问词：{self.question_word}\n'
                            f'结构：{self.dependency_relations}')

    def _dfs(self, current_word: str, target_word: str, current_path: List[str], visited: Set[str]) -> None:
        """
        深度优先搜索查找依赖路径。

        :param current_word: 当前处理的词语
        :param target_word: 目标词语
        :param current_path: 当前路径
        :param visited: 已访问节点集合
        """
        if current_word == target_word:
            if len(current_path) > 0:
                self.dependency_paths_list.append(current_path)
            return

        if current_word in self.edge_dict:
            for next_word, relation, direction in self.edge_dict[current_word]:
                if next_word not in visited:
                    visited.add(next_word)
                    self._dfs(next_word, target_word, current_path + [f'[{relation}：{direction}]'], visited)
                    visited.remove(next_word)

    def get_dependency_paths(self, must_find=True) -> Optional[List[str]]:
        """
        获取从结构词到问题词的最短依赖路径。

        :return: 最短依赖路径列表（字符串列表），如果没有路径则返回 None
        """
        if self.question_word == self.structure_word:
            return []
        self._find_shortest_dependency_paths(must_find)  # 查找最短依赖路径
        return self.dependency_paths_list if self.dependency_paths_list else None

    def find_all_shortest_paths(self, start_word: str) -> Dict[str, List[str]]:
        """
        从指定词出发，寻找到所有其他词的最短路径。

        :param start_word: 起始词语
        :return: 字典，键为目标词，值为到目标词的最短路径列表（字符串列表）
        """
        all_paths = {}  # 存储所有目标词及其对应的最短路径

        # 对图中的每个节点进行处理
        for target_word in self.edge_dict.keys():
            if start_word != target_word:
                self.dependency_paths_list = []  # 重置路径列表
                visited: Set[str] = set()
                visited.add(start_word)
                self._dfs(start_word, target_word, [], visited)

                if self.dependency_paths_list:
                    min_path_length = min(len(path) for path in self.dependency_paths_list)
                    all_paths[target_word] = [' --> '.join(path) for path in self.dependency_paths_list if
                                              len(path) == min_path_length]
                else:
                    all_paths[target_word] = []

        return all_paths


def find_structure_word(sentences: List[Sentence]) -> Tuple[str, str]:
    """
    根据给定句子识别句型词及其词性。

    参数:
    - sentences (List[Sentence]): Stanza处理过的句子对象列表。

    返回:
    Tuple[str, str]: 句型词及其词性。
    """
    # 定义辅助词集合
    auxiliary_set = {'who', 'what', 'when', 'which', 'how', 'where', 'whose'}

    for __sentence in sentences:
        # 检查第一个词是否是动词
        first_word = __sentence.words[0]
        if first_word.xpos.startswith('VB'):
            return first_word.text, first_word.xpos

        # 特殊情况 "How many"
        if (len(__sentence.words) >= 2 and __sentence.words[0].text.lower() == 'how'
                and __sentence.words[1].text.lower() == 'many'):
            return 'How many', __sentence.words[0].xpos

        # 查找辅助词
        for word in __sentence.words:
            if word.text.lower() in auxiliary_set:
                return word.text, word.xpos

        # 查找动词
        for word in __sentence.words:
            if word.xpos.startswith('V'):
                return word.text, word.xpos

        # 查找小写词
        for word in __sentence.words:
            if word.text.islower():
                return word.text, word.xpos

    raise Exception(f'{Sentence} \nError! 未找到指定的句型词!')


def find_question_word_and_pos(sentences: List[Sentence], question_word: str) -> Tuple[str, str]:
    """
    根据给定句子列表中的疑问词识别其词性。

    参数:
    - sentences (List[Sentence]): Stanza处理过的句子对象列表。
    - question_word (str): 需要识别词性的问题词。

    返回:
    Tuple[str, str]: 疑问词及其词性。
    """
    for __sentence in sentences:
        for word in __sentence.words:
            if word.text.lower() == question_word.lower():
                return word.text, word.xpos
            if word.lemma == question_word.lower():
                return word.lemma, word.xpos
    print(sentences)
    raise Exception(f'Error! 未找到疑问词的词性！在句子列表中没有找到：{question_word}')


def map_word_positions_to_relations(word: str, dependency_relations: List[Tuple[str, Tuple[str, str]]],
                                    type_prefix: str) -> List[Tuple[str, str]]:
    """
    映射给定词在各种依存关系中的位置至相应的位置标记列表中。

    参数:
    - word (str): 分析的目标给定词。
    - dependency_relations (List[Tuple[str, Tuple[str, str]]]): 每个元组包含依存关系类型和一个元组(头部词汇, 依存词汇)。

    返回:
    List[Tuple[str, int]]: 给定词在所有可能依存关系类型中的位置标记列表。
    """
    # all_relations = {
    #     'NUMMOD', 'OBL_TMOD', 'ADVCL', 'OBL_AGENT', 'CONJ', 'OBL', 'CC',
    #     'OBL_NPMOD', 'CC_PRECONJ', 'COP', 'NMOD_POSSESS', 'PUNCT', 'XCOMP',
    #     'EXPL', 'AUX', 'OBJ', 'ACL', 'CCOMP', 'ACL_RELCL', 'DEP', 'APPOS',
    #     'NSUBJ_PASS', 'FLAT', 'CASE', 'AMOD', 'ROOT', 'NMOD_NPMOD', 'AUX_PASS',
    #     'MARK', 'ADVCL_RELCL', 'ADVMOD', 'NMOD', 'IOBJ', 'DET_PREDET', 'FIXED',
    #     'DET', 'COMPOUND', 'NSUBJ'
    # }

    position_markers = []
    for rel, (head, dep) in dependency_relations:
        if word in (head, dep):
            position = 1 if head == word else 2
            position_markers.append((type_prefix + rel.upper().replace(':', '_'), str(position)))

    return position_markers


class DependencyAnalyzer:
    def __init__(self, model_dir: str, _sentences_list: List[str], question_word_list: List[str]):
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
        self.question_words_and_pos_list = []  # 所有句子的疑问词及其词性

        self.sentences_list = _sentences_list  # 数据集，包含全部的句子
        self.question_word_list = question_word_list  # 数据集，一一对应每个句子的疑问词

        self.initialize()  # 初始化操作

    def initialize(self):
        """
        初始化 Stanza 的 NLP Pipeline。

        在使用其他方法之前，必须调用此方法初始化 NLP Pipeline。
        """
        try:
            self.nlp = stanza.Pipeline('en',
                                       processors='tokenize,pos,lemma,depparse', use_gpu=True,
                                       dir=self.model_dir, download_method=None)
        except Exception as e:
            print(f"初始化 NLP Pipeline 失败: {e}")
            raise

        self._process_sentences()

    def _process_sentences(self):
        """
        处理所有句子，提取依赖关系、词性和句型词。
        """
        if len(self.question_word_list) != 0:
            assert len(self.question_word_list) == len(self.sentences_list), (
                f'错误，提取疑问词相关信息必须保证每个句子都有对应的疑问词！\n'
                f'疑问词列表长度：{len(self.question_word_list)}\n'
                f'句子列表长度：{len(self.sentences_list)}'
            )

            for sentence_, question_word in zip(self.sentences_list, self.question_word_list):
                try:
                    # 使用正则表达式进行部分匹配验证
                    assert re.search(r'\b' + re.escape(question_word) + r'\b', sentence_, re.IGNORECASE), (
                        f'疑问词不合法！\n句子：{sentence_}\n疑问词：{question_word}\n\n'
                    )
                except AssertionError as e:
                    raise ValueError(f'处理句子时出错：\n{str(e)}')

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

            if len(self.question_word_list) != 0:
                __question_word_and_pos = find_question_word_and_pos(doc.sentences, self.question_word_list[index_])
                self.question_words_and_pos_list.append(__question_word_and_pos)

    def extract_sentences_dependencies_paths(self) -> List[List[str]]:
        """
        计算并收集每个句子的依赖路径列表。

        返回:
        List[List[str]]: 每个句子的依赖路径列表。
        """
        sentences_dependencies_paths = []

        for __sentence, question_word, structure_word_pos, dependencies_relations in zip(
                self.sentences_list,
                self.question_word_list,
                self.structure_words_and_pos_list,
                self.all_words_dependencies_list
        ):
            dependency_resolver = ShortestPathFinder(
                question_word=question_word,
                structure_word=structure_word_pos[0],
                dependency_relations=dependencies_relations,
                _sentence=__sentence
            )

            sentence_dependency_paths = dependency_resolver.get_dependency_paths()
            sentences_dependencies_paths.append(sentence_dependency_paths)

        return sentences_dependencies_paths

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

    def get_question_words_and_pos(self) -> List[Tuple[str, str]]:
        """
        获取疑问词及其词性信息。

        返回:
        List[Tuple[str, str]]: 包含疑问词及其词性信息的列表。
        """
        return self.question_words_and_pos_list

    def get_sentences_dependencies_paths(self) -> List[List[str]]:
        """
        获取每个句子的依赖路径列表。

        返回:
        List[List[str]]: 每个句子的依赖路径列表。
        """
        return self.extract_sentences_dependencies_paths()

    def find_same_dependency(self, idx):
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
            if self.question_word_list[idx] == head_word and self.structure_words_and_pos_list[idx][0] == word:
                dependency_list.append(('SAME_DEPENDENCY', rel + '_1'))
            if self.question_word_list[idx] == word and self.structure_words_and_pos_list[idx][0] == head_word:
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

    def get_question_words_in_dependencies_position(self) -> List[List[Tuple[str, str]]]:
        """
        获取疑问词在依赖结构中的位置。

        返回:
        List[List[Tuple[str, int]]]: 句型词在依赖树中的位置标记列表。
        """
        question_words_in_dependencies_position = []
        for __index, dependencies in enumerate(self.all_words_dependencies_list):
            question_words_in_dependencies_position.append(map_word_positions_to_relations(
                self.question_word_list[__index], dependencies, 'QUESTION_'))
        return question_words_in_dependencies_position

    def retrieve_all_information(self):
        result_list = []
        for __index, question_word in enumerate(self.question_word_list):
            temp_list = []
            # 疑问词与句型词相同
            if question_word == self.structure_words_and_pos_list[__index][0]:
                temp_list.append(('SAME_QS_WORD', 'True'))
            # 是否同依赖
            _ = self.find_same_dependency(__index)
            if _ is not None:
                temp_list.extend(_)

            # 依赖路径
            dependency_resolver = ShortestPathFinder(
                question_word=self.question_word_list[__index],  # 当前句子的问题词
                structure_word=self.structure_words_and_pos_list[__index][0],  # 当前句子的结构词
                dependency_relations=self.all_words_dependencies_list[__index],  # 当前句子的依赖关系列表
                _sentence=self.sentences_list[__index]  # 当前句子文本
            )
            sentence_dependency_paths = dependency_resolver.get_dependency_paths()
            for sentence_dependency_path in sentence_dependency_paths:
                temp_list.append(('DEPENDENCY_PATH', sentence_dependency_path))

            # 句子类型(疑问句 or 陈述句)
            __str = '疑问句' if self.sentences_list[__index][-1] == '?' else '陈述句'
            temp_list.append(('SENTENCE_PATTERN', __str))

            # # 句型词
            # temp_list.append(('SENTENCE_STRUCTURE_WORD', self.structure_words_and_pos_list[__index][0]))

            # # 特殊处理疑问句的句型词
            # auxiliary_set = {'who', 'what', 'when', 'which', 'how', 'where', 'whose', 'how many'}
            # if (self.structure_words_and_pos_list[__index][0].lower() in auxiliary_set
            #         and __str == '疑问句'):
            #     temp_list.append(('SENTENCE_STRUCTURE_WORD', self.structure_words_and_pos_list[__index][0].lower()))
            # else:
            #     temp_list.append(('SENTENCE_STRUCTURE_WORD', self.structure_words_and_pos_list[__index][0]))

            # 特殊处理陈述句中的句型词VB
            auxiliary_set = {'who', 'what', 'when', 'which', 'how', 'where', 'whose', 'how many'}
            if ('V' in self.structure_words_and_pos_list[__index][1]
                    and self.structure_words_and_pos_list[__index][0].lower() not in auxiliary_set
                    and __str == '陈述句'):
                temp_list.append(('SENTENCE_STRUCTURE_WORD', 'VB'))
            else:
                temp_list.append(('SENTENCE_STRUCTURE_WORD', self.structure_words_and_pos_list[__index][0]))

            # 句型词词性
            temp_list.append(('SENTENCE_STRUCTURE_WORD_POS', self.structure_words_and_pos_list[__index][1]))

            # 疑问词
            temp_list.append(('QUESTION_WORD', self.question_word_list[__index]))

            # 疑问词词性
            temp_list.append(('QUESTION_WORD_POS', self.question_words_and_pos_list[__index][1]))

            # 句型词在依赖关系中的位置
            __structure_word = self.structure_words_and_pos_list[__index][0]
            if __structure_word == 'How many':
                __structure_word = 'How'
            temp_list.extend(map_word_positions_to_relations(__structure_word,
                                                             self.all_words_dependencies_list[__index], 'SENTENCE_'))

            # 疑问词在依赖关系中的位置
            temp_list.extend(map_word_positions_to_relations(self.question_word_list[__index],
                                                             self.all_words_dependencies_list[__index], 'QUESTION_'))
            result_list.append(temp_list)
        return result_list


def read_json_file(file_path):
    """
    从指定路径读取 JSON 文件并返回其内容。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def export_to_excel(_transaction_data, output_file='linguistic_analysis.xlsx'):
    # 提取所有唯一的特征名称
    feature_names = set()
    for record in _transaction_data:
        for feature_name, _ in record:
            feature_names.add(feature_name)
    feature_names = sorted(list(feature_names))

    # 创建结构化数据
    structured_data = []
    for record in _transaction_data:
        row_data = {name: None for name in feature_names}
        for feature_name, value in record:
            row_data[feature_name] = value
        structured_data.append(row_data)

    # 创建DataFrame
    df = pd.DataFrame(structured_data)

    # 导出到Excel
    try:
        df.to_excel(output_file, index=False)
        print(f"数据已成功导出到 {output_file}")
    except Exception as e:
        print(f"导出过程中发生错误: {str(e)}")

    return df


def save_to_json(_data, file_path):
    # 确保 JSON 格式统一且字段顺序一致
    formatted_data = [
        {
            "id": idx + 1,  # 重新编号
            "sentence": item[0],
            "question_word": item[1]
        }
        for idx, item in enumerate(_data)
    ]
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(formatted_data, file, indent=4, ensure_ascii=False)
    print(f"合并数据已成功保存到 {file_path}")


# # 生成训练模型的数据
# if __name__ == "__main__":
#     json_file_path = "../../data/raw/All-data.json"
#     xlsx_file_path = "../../data/train-data.xlsx"
#     data = read_json_file(json_file_path)
#     sentences_1 = [(item['sentence'], item['question_word']) for item in data]
#
#     dependency_analyzer = DependencyAnalyzer(model_dir='F:\\',
#                                              _sentences_list=[test_set[0] for test_set in sentences_1],
#                                              question_word_list=[test_set[1] for test_set in sentences_1])
#     transaction_data = dependency_analyzer.retrieve_all_information()
#
#     export_to_excel(transaction_data, output_file=xlsx_file_path)
