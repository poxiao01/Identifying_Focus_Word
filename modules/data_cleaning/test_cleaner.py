"""
模块名称: TestDataCleaner
功能描述: 生成测试集输入数据格式的数据。
设计目的: 清洗原始测试数据并生成模型预测所需的输入格式。
"""

from modules.data_cleaning.test_data_processor import DataProcessor


class TestDataCleaner:
    """
    测试数据清洗器，负责清洗测试数据并生成模型所需的输入格式。
    """

    def __init__(self, model_dir: str, raw_data_path: str, _output_file_path: str):
        """
        初始化测试数据清洗器。

        :param model_dir: Stanza 模型路径
        :param raw_data_path: 原始数据的 JSON 文件路径
        :param _output_file_path: 输出清洗后的数据的文件路径
        """
        self.model_dir = model_dir
        self.raw_data_path = raw_data_path
        self.output_file_path = _output_file_path

    def clean_and_generate(self):
        """
        主流程：读取数据 -> 清洗数据 -> 调用依赖分析 -> 保存结果。
        """
        # 创建数据处理器实例
        processor = DataProcessor(self.raw_data_path, self.model_dir, self.output_file_path)

        # 执行数据处理和保存
        processor.process_and_save()


# 独立运行脚本
if __name__ == "__main__":
    # 配置路径
    json_file_path = "../../data/raw/test-data.json"
    output_file_path = "../../data/cleaned/test-data-cleaned.json"

    # 初始化并运行清洗器
    cleaner = TestDataCleaner(
        model_dir="../../resources/stanza_resources",  # Stanza 模型路径
        raw_data_path=json_file_path,
        _output_file_path=output_file_path
    )
    cleaner.clean_and_generate()
