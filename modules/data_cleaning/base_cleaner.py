"""
模块名称: base_cleaner
功能描述: 数据清洗模块的抽象基类。
设计目的: 提供统一的接口，确保不同类型的清洗器可以动态调用。
"""


class BaseCleaner:
    """
    数据清洗模块基类，所有清洗器需继承此类。
    """

    def __init__(self, name: str):
        """
        初始化清洗模块。

        :param name: 清洗模块名称，用于标识。
        """
        self.name = name

    def clean(self, data: dict) -> dict:
        """
        清洗逻辑主接口，需在子类中实现。

        :param data: 待清洗的数据（字典格式）。
        :return: 清洗后的数据（字典格式）。
        """
        raise NotImplementedError("子类必须实现 `clean` 方法")
