"""
模块名称: experiment_registry
功能描述: 实验注册表，用于集中管理和动态加载实验类。
设计目的:
    - 通过注册机制简化实验类的动态加载过程。
    - 提供一个集中式的注册和查询接口，便于扩展和维护。
主要功能:
    - `register`: 注册实验类。
    - `get`: 根据实验名称获取实验类。
适用场景:
    - 当需要添加新的实验类型时，只需将其注册到注册表中，无需修改其他代码。
"""
class ExperimentRegistry:
    """
    实验注册表，用于集中管理和动态加载实验类。
    """
    registry = {}

    @staticmethod
    def register(name, experiment_class):
        """
        注册实验类。

        :param name: 实验名称。
        :param experiment_class: 实验类。
        """
        ExperimentRegistry.registry[name] = experiment_class

    @staticmethod
    def get(name):
        """
        根据名称获取实验类。

        :param name: 实验名称。
        :return: 注册的实验类。
        """
        if name not in ExperimentRegistry.registry:
            raise ValueError(f"实验 '{name}' 未注册")
        return ExperimentRegistry.registry[name]
