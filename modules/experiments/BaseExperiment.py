"""
模块名称: base_experiment
功能描述: 定义实验基类，为所有具体实验提供标准流程。
设计目的: 提供一个抽象基类，确保所有实验实现具有统一的接口和标准化的执行流程。
主要功能:
    - 定义 `setup`、`run`、`teardown` 抽象方法。
    - 提供实验的初始化方法，支持实验配置参数。
适用场景:
    - 所有实验类都需要继承该基类并实现其方法，以确保符合标准流程。
"""

class BaseExperiment:
    """
    实验基类，用于定义实验的标准流程。
    """
    def __init__(self, config):
        """
        初始化实验。

        :param config: 实验配置字典。
        """
        self.config = config

    def setup(self):
        """
        实验初始化，通常用于加载资源或设置环境。
        子类必须实现该方法。
        """
        raise NotImplementedError("子类必须实现 setup 方法")

    def run(self):
        """
        实验运行逻辑。
        子类必须实现该方法。
        """
        raise NotImplementedError("子类必须实现 run 方法")

    def teardown(self):
        """
        实验清理逻辑，通常用于释放资源。
        子类必须实现该方法。
        """
        raise NotImplementedError("子类必须实现 teardown 方法")
