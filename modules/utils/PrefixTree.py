from modules.utils.Itemset import Itemset


class PrefixTreeNode:
    """
    前缀树节点类，每个节点存储事务ID集合以及子节点信息。
    """

    def __init__(self, tree_id=None):
        self.transaction_ids = set()  # 存储与当前节点相关的事务ID集合
        self.children = {}  # 子节点映射，键为条件项集，值为子节点
        self.count = 0  # 当前节点的事务计数
        self.tree_id = tree_id  # 当前树的ID，仅根节点有意义


class PrefixTree:
    """
    前缀树类，用于存储和处理事务数据的分支。
    """

    def __init__(self, tree_id):
        """
        初始化前缀树。

        :param tree_id: 当前前缀树的ID。
        """
        self.root = PrefixTreeNode(tree_id=tree_id)  # 为根节点设置树的ID

    def update(self, condition_itemset, transaction_id):
        """
        更新前缀树，插入条件项集，并仅在路径的最后一个节点记录事务ID。

        :param condition_itemset: 条件项集（如 [(D1, 1), (C1, 1), (C2, 1)]）。
        :param transaction_id: 当前事务的ID。
        """
        current_node = self.root

        # 遍历路径上的每个元素，直到最后一个元素
        for idx, item in enumerate(condition_itemset):
            if item not in current_node.children:
                # 如果子节点不存在，则创建新节点
                current_node.children[item] = PrefixTreeNode()

            # 移动到下一个节点
            current_node = current_node.children[item]

            # 仅在最后一个节点记录事务ID和更新计数
            if idx == len(condition_itemset) - 1:
                current_node.transaction_ids.add(transaction_id)
                current_node.count += 1

    def delete_non_frequent(self, min_support):
        """
        删除不满足最小支持度的项集。

        :param min_support: 最小支持度阈值。
        """
        def recursive_prune(node):
            # 递归检查并删除不满足支持度的节点
            keys_to_delete = []
            for key, child_node in node.children.items():
                recursive_prune(child_node)
                if child_node.count <= min_support:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del node.children[key]

        for child in self.root.children.values():
            recursive_prune(child)

    def get_frequent_itemsets(self, frequent_itemsets):
        """
        遍历当前前缀树，收集所有频繁项集及其事务ID集合和树ID集合。

        :param frequent_itemsets: 存储频繁项集的字典，键为项集名称（元组形式），值为 Itemset 对象。
        """
        current_tree_id = self.root.tree_id  # 当前前缀树的唯一标识（树ID）

        def traverse_tree(node, current_prefix):
            """
            递归遍历前缀树，收集频繁项集。

            :param node: 当前前缀树节点。
            :param current_prefix: 当前路径中的项集前缀（表示项集）。
            """
            for child_key, child_node in node.children.items():
                # 构造新的项集前缀
                updated_prefix = tuple(current_prefix) + child_key

                # 如果项集不在 frequent_itemsets 中，则创建新项集
                if updated_prefix not in frequent_itemsets:
                    frequent_itemsets[updated_prefix] = Itemset(
                        item_name=updated_prefix,
                        tree_ids=current_tree_id,
                        transaction_ids=child_node.transaction_ids
                    )
                else:
                    # 如果项集已存在，则合并
                    frequent_itemsets[updated_prefix].merge(
                        Itemset(
                            item_name=updated_prefix,
                            tree_ids=current_tree_id,
                            transaction_ids=child_node.transaction_ids
                        )
                    )

                # 递归处理子节点
                traverse_tree(child_node, list(updated_prefix))

        # 从根节点开始递归遍历
        for child in self.root.children.values():
            traverse_tree(child, [])

    def display(self, node=None, depth=0):
        """
        打印前缀树的结构。

        :param node: 当前节点。
        :param depth: 当前深度。
        """
        if node is None:
            node = self.root

        indent = "  " * depth
        if depth == 0:  # 根节点
            print(f"{indent}Root (Tree ID: {node.tree_id})")
        for key, child_node in node.children.items():
            print(f"{indent}{key}: {child_node.transaction_ids} (Count: {child_node.count})")
            self.display(child_node, depth + 1)
