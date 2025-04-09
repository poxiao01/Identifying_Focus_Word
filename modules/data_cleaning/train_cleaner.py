"""
模块名称: TrainDataCleaner
功能描述: 生成训练集输入数据格式的数据。
设计目的: 生成清洗后的训练数据，以便训练模型。
完成时间: 2025/1/13
"""

from modules.data_cleaning.dependency_analyzer import DependencyAnalyzer, export_to_excel, read_json_file


class TrainDataCleaner:
    """
    训练数据清洗器，负责清洗训练数据并生成模型所需的输入格式。
    """

    def __init__(self, model_dir: str, raw_data_path: str, output_file_path: str):
        """
        初始化训练数据清洗器。

        :param model_dir: Stanza 模型路径
        :param raw_data_path: 原始数据的 JSON 文件路径
        :param output_file_path: 输出清洗后的数据的文件路径
        """
        self.model_dir = model_dir
        self.raw_data_path = raw_data_path
        self.output_file_path = output_file_path

    def clean_and_generate(self):
        """
        主流程：读取数据 -> 清洗数据 -> 调用依赖分析 -> 保存结果。
        """
        # Step 1: 读取原始数据
        print(f"读取原始数据文件: {self.raw_data_path}")
        raw_data = read_json_file(self.raw_data_path)

        # Step 2: 提取句子和疑问词
        sentences_1 = [(item['sentence'], item['question_word']) for item in raw_data]

        # Step 3: 初始化依赖分析器
        print("初始化依赖分析器...")
        dependency_analyzer = DependencyAnalyzer(
            model_dir=self.model_dir,
            _sentences_list=[test_set[0] for test_set in sentences_1],
            question_word_list=[test_set[1] for test_set in sentences_1]
        )

        # Step 4: 获取依赖分析结果
        print("提取依赖路径信息...")
        transaction_data = dependency_analyzer.retrieve_all_information()

        # Step 5: 保存结果到 Excel 文件
        print(f"保存结果到文件: {self.output_file_path}")
        export_to_excel(transaction_data, output_file=self.output_file_path)

        print("训练数据生成完成！")


# 独立运行脚本
if __name__ == "__main__":
    # 配置路径
    json_file_path = "../../data/raw/train-data.json"
    xlsx_file_path = "../../data/train-data-cleand.xlsx"

    # 初始化并运行清洗器
    cleaner = TrainDataCleaner(
        model_dir="../../resources/stanza_resources",  # Stanza 模型路径
        raw_data_path=json_file_path,
        output_file_path=xlsx_file_path
    )
    cleaner.clean_and_generate()
