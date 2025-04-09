class Itemset:
    """
    表示频繁项集的数据结构，包含项集名称、树ID集合和事务ID集合。
    """

    def __init__(self, item_name, tree_ids=None, transaction_ids=None):
        """
        初始化项集。

        :param item_name: 项集名称，唯一标识项集。
        :param tree_ids: 树ID集合（可以是单个整数或集合）。
        :param transaction_ids: 事务ID集合（可以是单个整数或集合）。
        """
        self.item_name = item_name  # 项集名称

        # 如果 tree_ids 是整数，则包装为集合；否则直接转为集合
        if isinstance(tree_ids, int):
            self.tree_ids = {tree_ids}
        else:
            self.tree_ids = set(tree_ids) if tree_ids else set()  # 树ID集合，默认为空集合

        # 如果 transaction_ids 是整数，则包装为集合；否则直接转为集合
        if isinstance(transaction_ids, int):
            self.transaction_ids = {transaction_ids}
        else:
            self.transaction_ids = set(transaction_ids) if transaction_ids else set()  # 事务ID集合，默认为空集合

    def merge(self, other):
        """
        合并另一个项集，将其树ID集合和事务ID集合合并到当前项集。

        :param other: 要合并的另一个 Itemset 对象。
        :raises ValueError: 如果项集名称不同，则无法合并。
        """
        if self.item_name != other.item_name:
            raise ValueError(f"无法合并：项集名称不同（{self.item_name} != {other.item_name}）")

        # 合并树ID和事务ID
        self.tree_ids.update(other.tree_ids)
        self.transaction_ids.update(other.transaction_ids)

    def __repr__(self):
        """
        返回项集的字符串表示，用于调试。
        """
        return f"Itemset(item_name={self.item_name}, tree_ids={self.tree_ids}, transaction_ids={self.transaction_ids})"

    def __eq__(self, other):
        """
        判断两个项集是否相等（基于项集名称）。
        """
        return self.item_name == other.item_name

    def __hash__(self):
        """
        定义哈希规则，支持在集合或字典中存储。
        """
        return hash(self.item_name)
