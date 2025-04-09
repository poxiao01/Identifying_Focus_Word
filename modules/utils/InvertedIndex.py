from collections import defaultdict
from typing import List, Dict, Tuple, Set


class InvertedIndex:
    """
    倒排索引类，用于存储和查询关联规则。

    该类提供了基于前件集合查询规则的功能，并能够根据规则的置信度*支持度和后件数量限制返回符合条件的后件。
    """

    def __init__(self, rules: Set[Tuple[frozenset, frozenset, str, str]]):
        """
        初始化倒排索引并构建索引。

        在初始化过程中，该类将传入的规则集转化为倒排索引的形式，便于快速查询。

        Args:
            rules (Set[Tuple[frozenset, frozenset, str, str]]): 包含所有规则的集合，每条规则包括前件、后件和置信度和支持度。

        """
        self.index: Dict[str, List[Tuple[int, int, float, float]]] = defaultdict(list)
        self.rules: List[Tuple[frozenset, frozenset, str, str]] = list(rules)  # 将规则集转换为列表，便于后续的索引和访问
        self._build_index()  # 构建倒排索引

    def _build_index(self):
        """
        根据规则构建倒排索引。
        """
        for rule_id, (antecedents, _, confidence, support) in enumerate(self.rules):
            antecedents_count = len(antecedents)  # 获取前件集合中元素的数量
            for item in antecedents:
                # 为每个前件项建立索引，存储规则ID、前件项数量和置信度和支持度
                self.index[item].append((rule_id, antecedents_count, float(confidence), float(support)))

    def _get_valid_consequences(self, matched_rules: Dict[int, int]) -> Set[Tuple[frozenset, float, float]]:
        """
        从匹配的规则中筛选出有效的后件。

        根据前件匹配的数量和规则中的前件数量进行比较，只有前件完全匹配的规则才会被认为是有效的。

        Args:
            matched_rules (Dict[int, int]): 存储规则ID与匹配的前件数量的字典。

        Returns:
            Set[Tuple[frozenset, float, float]]: 有效后件集合，包含后件集合和对应的置信度和支持度。
        """
        valid_consequences = set()

        for rule_id, matched_count in matched_rules.items():
            # 如果规则的前件数量与匹配的数量一致，则认为规则有效
            if len(self.rules[rule_id][0]) == matched_count:
                consequence = self.rules[rule_id][1]
                confidence = float(self.rules[rule_id][2])
                support = float(self.rules[rule_id][3])
                valid_consequences.add((consequence, confidence, support))

        return valid_consequences

    def _sort_consequences(self, valid_consequences: Set[Tuple[frozenset, float, float]]) -> (
            List)[Tuple[frozenset, float, float]]:
        """
        按支持度 * 置信度对有效后件进行降序排序。

        该方法将有效的后件按照支持度*置信度降序排列，若相同，则按字典顺序排序后件。

        Args:
            valid_consequences (Set[Tuple[Frozenset, float, float]]): 有效的后件集合，包含后件集合和对应的置信度、支持度。

        Returns:
            List[Tuple[Frozenset, float, float]]: 排序后的有效后件列表，按置信度降*支持度序排列。
        """
        return sorted(
            valid_consequences,
            key=lambda x: (x[1] * x[2], tuple(sorted(x[0]))),  # 使用支持度 * 置信度作为排序条件
            reverse=True  # 按置信度降序排列
        )

    def _collect_multiple_consequences(self, sorted_consequences: List[Tuple[frozenset, float, float]],
                                       consequences_dict: Dict, max_consequences: int) -> List[str]:
        """
        根据最大后件数量收集多个符合条件的后件。

        该方法会遍历排序后的有效后件列表，根据最大后件数量限制收集结果。

        Args:
            sorted_consequences (List[Tuple[Frozenset, float, float]]): 排序后的有效后件列表。
            consequences_dict (Dict): 后件字典，包含后件集合对应的词。
            max_consequences (int): 限制返回的最大后件数量。

        Returns:
            List[str]: 符合条件的词列表。
        """
        used_count = 0
        found_words = []

        for frozenset_consequences, confidence, support in sorted_consequences:
            if used_count >= max_consequences:  # 达到最大限制后停止
                break

            consequences = next(iter(frozenset_consequences))  # 解冻后件集合
            if consequences in consequences_dict:
                for word in consequences_dict[consequences]:
                    if used_count >= max_consequences:  # 再次检查数量限制
                        break
                    if word not in found_words:
                        used_count += 1
                        found_words.append(word)

        return found_words
        #
        # """
        # 根据最大后件数量收集多个符合条件的后件。
        #
        # 该方法会遍历排序后的有效后件列表，根据最大后件数量限制收集结果。
        # 现修改匹配规则：先按 items_frozenset 长度降序排序，再仅匹配与 can_use_frozenset_consequences
        # 重合度大于 0.7 的结果。
        #
        # Args:
        #     sorted_consequences (List[Tuple[FrozenSet, float, float]]): 排序后的有效后件列表。
        #     consequences_dict (Dict): 后件字典，包含后件集合对应的词。
        #     max_consequences (int): 限制返回的最大后件数量。
        #
        # Returns:
        #     List[str]: 符合条件的词列表。
        # """
        # used_count = 0
        # found_words = []
        #
        # # 构建 word_to_items：将每个词映射到它对应的所有项
        # word_to_items = defaultdict(list)
        # for item, words in consequences_dict.items():
        #     for word in words:
        #         word_to_items[word].append(item)
        #
        # # 构建 items_to_word：将相同项集合的词聚集在一起
        # items_to_word = defaultdict(list)
        # for word, items_list in word_to_items.items():
        #     items_fs = frozenset(items_list)
        #     items_to_word[items_fs].append(word)
        #
        # # 从排序后的后件列表中提取可用的后件（这里只取每个 frozenset 中的第一个元素）
        # can_use_items = set()
        # for frozenset_consequences, confidence, support in sorted_consequences:
        #     consequence = next(iter(frozenset_consequences))
        #     can_use_items.add(consequence)
        # can_use_items = frozenset(can_use_items)
        #
        # # 遍历 items_to_word，先按 items_frozenset 的长度降序排序
        # for items_fs in sorted(items_to_word.keys(), key=lambda s: len(s), reverse=True):
        #     if not items_fs:
        #         continue  # 跳过空集合
        #
        #     # 计算重合度：items_fs 与 can_use_items 交集大小占 items_fs 大小的比例
        #     overlap_ratio = len(items_fs & can_use_items) / len(items_fs)
        #     if overlap_ratio > 0.21:
        #         for word in items_to_word[items_fs]:
        #             if used_count >= max_consequences:
        #                 break
        #             if word not in found_words:
        #                 found_words.append(word)
        #                 used_count += 1
        #     if used_count >= max_consequences:
        #         break
        # return found_words

    def query_rules_by_sentences_with_limit(
            self,
            sentence_info_list: List[Tuple[Set[str], Dict[str, List[str]]]],
            max_consequences: int
    ) -> List[List[str]]:
        """
        根据给定的前件集合查询符合条件的规则，并限制返回的后件数量。

        该方法会遍历每个句子，根据句子的前件集合查询倒排索引，找到与之匹配的规则，并返回相应的后件词。结果会根据最大后件数量进行限制。

        Args:
            sentence_info_list (List[Tuple[Set[str], Dict[str, List[str]]]]): 包含多个句子信息的列表，每个句子由前件集合和后件字典组成。
            max_consequences (int): 每个句子返回的最大后件数量。

        Returns:
            List[List[str]]: 每个句子对应的符合条件的后件词列表。
        """
        results = []

        for antecedents_set, consequences_dict in sentence_info_list:
            temp_dict = defaultdict(int)  # 存储每条规则与其匹配的前件数量

            # 遍历前件集合，查询索引中的相关规则
            for item in antecedents_set:
                if item in self.index:
                    for rule_entry in self.index[item]:
                        rule_id = rule_entry[0]
                        temp_dict[rule_id] += 1

            # 获取有效的后件集合
            valid_consequences = self._get_valid_consequences(temp_dict)

            # 处理最大后件数量
            sorted_consequences = self._sort_consequences(valid_consequences)
            found_words = self._collect_multiple_consequences(sorted_consequences, consequences_dict,
                                                              max_consequences)
            results.append(found_words)

        return results

    def query_single_sentence_with_limit(
            self,
            antecedents_set: Set[str],
            consequences_dict: Dict[str, List[str]],
            max_consequences: int
    ) -> List[str]:
        """
        根据单个句子的前件集合查询规则，并限制返回的后件数量。

        该方法针对单个句子的前件集合进行查询，返回与之匹配的后件，按置信度排序，并根据最大后件数量进行限制。

        Args:
            antecedents_set (Set[str]): 单个句子的前件集合。
            consequences_dict (Dict[str, List[str]]): 后件字典，包含后件对应的词。
            max_consequences (int): 限制返回的最大后件数量。

        Returns:
            List[str]: 符合条件的后件词列表。
        """
        temp_dict = defaultdict(int)  # 存储每条规则与其匹配的前件数量

        # 遍历前件集合，查询索引中的相关规则
        for item in antecedents_set:
            if item in self.index:
                for rule_entry in self.index[item]:
                    rule_id = rule_entry[0]
                    temp_dict[rule_id] += 1

        # 获取有效的后件集合
        valid_consequences = self._get_valid_consequences(temp_dict)

        # 处理最大后件数量为1的情况
        sorted_consequences = self._sort_consequences(valid_consequences)
        return self._collect_multiple_consequences(sorted_consequences, consequences_dict, max_consequences)

    def show_index(self):
        """
        打印倒排索引的内容，用于调试或查看。
        """
        for key, records in self.index.items():
            print(f"{key}: {records}")

    def show_rules(self):
        for i, rule in enumerate(self.rules):
            print(f'{i} {rule}')
