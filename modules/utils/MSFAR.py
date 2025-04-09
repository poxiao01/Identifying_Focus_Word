import sys
from decimal import Decimal

from itertools import combinations

from modules.utils.PrefixTree import PrefixTree


class MSFAR:
    """
    强焦点关联规则挖掘算法实现类。
    """

    def __init__(self, min_support, min_confidence):
        self.min_support = min_support  # 最小支持度阈值
        self.min_confidence = min_confidence  # 最小置信度阈值
        self.prefix_trees = {}  # 存储每个前缀树
        self.all_frequent_itemsets = {}  # 存储全部频繁项集
        self.total_number_of_transactions = 0  # 原始事务总数

    def initialize_prefix_trees(self, transaction_items):
        """
        初始化前缀树，为每个包含 D 的事务创建独立的前缀树，并将对应的 C 项插入。

        :param transaction_items: 事务集合。
            示例：
            [
                [(D1, 1), (C1, 1), (C2, 1), (C3, 1), (C4, 1)],
                [(D1, 1), (C1, 1), (C2, 1), (C3, 1)],
                [(D2, 1), (C1, 1), (C2, 1), (C3, 1)],
            ]
        """
        self.total_number_of_transactions = len(transaction_items)
        for transaction_id, items in enumerate(transaction_items, start=1):
            # D 项集合（决策项）
            D_item = []
            # C 项集合（条件项）
            C_item = []

            # 按类型分类
            for item in items:
                if 'D' in item[0]:  # 事务项包含 'D'
                    D_item.append(item)
                else:  # 其余为C项
                    C_item.append(item)

            # 对 D 项和 C 项分别排序
            D_item.sort(key=lambda x: x[0])  # 按项名称排序
            C_item.sort(key=lambda x: x[0])  # 按项名称排序
            # 校验 D_item 和 C_item 是否为空
            if not D_item:  # 如果 D 项为空
                print(f"错误：在事务 ID 为 {transaction_id} 的数据中未找到 D 项。事务数据为: {items}")
                sys.exit(1)  # 退出程序，返回错误状态码 1

            if not C_item:  # 如果 C 项为空
                print(f"错误：在事务 ID 为 {transaction_id} 的数据中未找到 C 项。事务数据为: {items}")
                sys.exit(1)  # 退出程序，返回错误状态码 1

            def generate_combinations(c_items):
                """
                生成 C 项集合的所有非空组合项集。

                :param c_items: C 项集合（如 ['C1', 'C2', 'C3']）。
                :return: 所有非空组合项集的列表。
                """
                all_combinations = []
                n = len(c_items)

                # 遍历从 1 到 n 的组合长度
                for r in range(1, n + 1):
                    # 生成长度为 r 的所有组合
                    combinations_r = combinations(c_items, r)
                    all_combinations.extend(combinations_r)

                return all_combinations

            c_combinations = generate_combinations(C_item)

            # 遍历每个决策项 D，创建或更新对应的前缀树
            for root in D_item:
                for c_combination in c_combinations:
                    # 检查前缀树是否已存在，不存在则创建新的前缀树
                    if root not in self.prefix_trees:
                        self.prefix_trees[root] = PrefixTree(tree_id=len(self.prefix_trees) + 1)

                    # 获取当前决策项对应的前缀树
                    current_tree = self.prefix_trees[root]

                    # 插入到前缀树：将 D + C 项插入
                    current_tree.update([root] + [c_combination], transaction_id)

        # 初步获取频繁项集
        for tree in self.prefix_trees.values():
            tree.get_frequent_itemsets(self.all_frequent_itemsets)

        # 删去支持度未达到阈值的项集
        to_delete = []  # 创建一个待删除的键列表
        for name, item in self.all_frequent_itemsets.items():
            if len(item.transaction_ids) <= self.min_support:
                to_delete.append(name)

        # 遍历完成后删除这些键
        for name in to_delete:
            del self.all_frequent_itemsets[name]

    def update_trees(self, transaction_data):
        """
        更新所有前缀树，根据事务数据插入条件项集和事务ID。

        :param transaction_data: 事务数据，包含事务ID、条件项集和决策项。
        """
        for tid, condition_itemset, decision_items in transaction_data:
            for decision_item in decision_items:
                if decision_item in self.prefix_trees:
                    self.prefix_trees[decision_item].update(condition_itemset, tid)

    def prune_trees(self):
        """
        修剪前缀树，删除不满足最小支持度的条件项集。
        """
        for tree in self.prefix_trees.values():
            tree.delete_non_frequent(self.min_support)

    def generate_rules(self):
        """
        生成强焦点关联规则。

        :return: 强焦点关联规则列表。
        """
        rules = []
        for root, tree in self.prefix_trees.items():
            one_tree_frequent_itemsets = {}
            tree.get_frequent_itemsets(one_tree_frequent_itemsets)

            for itemset_name, itemset in one_tree_frequent_itemsets.items():
                support = len(itemset.transaction_ids) / self.total_number_of_transactions
                confidence = len(itemset.transaction_ids) / len(
                    self.all_frequent_itemsets[itemset_name].transaction_ids)
                if confidence > 1:
                    print("Error!, 置信度大于1！！！！！")
                if Decimal(confidence) > Decimal(self.min_confidence):
                    rules.append({
                        "condition": itemset_name,
                        "decision": root,
                        "support": support,
                        "confidence": confidence
                    })
        return rules

    def display(self):
        """
        打印全部前缀树的结构。

        """
        for key, root in self.prefix_trees.items():
            root.display()
