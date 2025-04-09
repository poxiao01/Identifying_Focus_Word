from typing import Tuple, Dict


def check(item: Tuple[str, str]) -> bool:
    """
    判断项是否可以作为前件。

    :param item: 输入的项。
    :return: 若可以作为前件返回 True，否则返回 False。
    """
    return not (item[0].find('QUESTION') != -1 or item[0] in {'SAME_QS_WORD', 'SAME_DEPENDENCY', 'DEPENDENCY_PATH'})


class Hasher:
    def __init__(self, reserved_numbers: int):
        """
        初始化哈希器，设置初始保留的整数。

        :param reserved_numbers: 已经占用的整数（如 [0, 1, 2]）。
        """
        self.str_to_int_map: Dict[str, int] = {}  # 字符串到整数的映射
        self.int_to_str_map: Dict[int, str] = {}  # 整数到字符串的映射
        self.current_index = reserved_numbers   # 当前可用的最小整数索引

        # 初始化保留的整数（0 到 reserved_numbers - 1）
        for i in range(reserved_numbers):
            self.str_to_int_map[str(i)] = i
            self.int_to_str_map[i] = str(i)

    def _map_string(self, string: str) -> int:
        """
        将字符串映射到整数，如果不存在映射则创建新的映射。

        :param string: 输入字符串。
        :return: 映射后的整数。
        """
        if string not in self.str_to_int_map:
            self.current_index += 1
            self.str_to_int_map[string] = self.current_index
            self.int_to_str_map[self.current_index] = string
        return self.str_to_int_map[string]

    def hash(self, item: Tuple[str, str]) -> Tuple[str, str]:
        """
        将输入项映射为一对字符串，其中第一项根据是否为前件分别添加 'C' 或 'D'。

        :param item: 需要哈希的项，格式为 (前件, 后件)。
        :return: 映射后的值，格式为 ('C/D+整数', '整数')。
        """
        prefix = 'C' if check(item) else 'D'

        # 映射前件和后件
        first_hashed = self._map_string(item[0])
        second_hashed = self._map_string(item[1])

        # 返回带前缀的映射值
        return f"{prefix}{first_hashed}", f"{second_hashed}"

    def get_item(self, hash_value: int) -> str:
        """
        根据整数获取原始字符串。

        :param hash_value: 哈希值（整数）。
        :return: 对应的字符串，如果不存在则返回 None。
        """
        return self.int_to_str_map.get(hash_value, None)

    def get_mapping(self) -> Dict[str, int]:
        """
        获取当前的所有映射。

        :return: 字符串到整数的映射字典。
        """
        return self.str_to_int_map

    def decode_hashed_item(self, hashed_item: Tuple[str, str]) -> Tuple[str, str]:
        """
        根据哈希后的值 ('C35', '27') 返回原始值。

        :param hashed_item: 哈希后的项，格式为 ('C/D+整数', '整数')。
        :return: 对应的原始项，格式为 (原始字符串, 原始字符串)。
        """
        first_hashed = hashed_item[0]  # 获取第一个值，例如 'C35'
        second_hashed = hashed_item[1]  # 获取第二个值，例如 '27'

        # 提取整数部分（去掉 'C' 或 'D' 前缀）
        first_number = int(first_hashed[1:])  # 提取 '35' 部分
        second_number = int(second_hashed)  # 直接转为整数

        # 通过 hasher 获取原始值
        original_first = self.get_item(first_number)
        original_second = self.get_item(second_number)

        return original_first, original_second
