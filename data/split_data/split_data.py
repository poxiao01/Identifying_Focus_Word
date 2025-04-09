# import pandas as pd
# import json
# import random
# import os
#
#
# def create_datasets(excel_file, json_file):
#     # 读取Excel文件和JSON文件
#     df_excel = pd.read_excel(excel_file)
#     with open(json_file, 'r', encoding='utf-8') as f:
#         data_json = json.load(f)
#
#     # 确保两个文件的行数匹配
#     assert len(df_excel) == len(data_json), "Excel文件行数与JSON文件不匹配"
#
#     # 随机去掉280条问句
#     random_indices = random.sample(range(1, len(df_excel)), 280)  # 去掉的索引
#     df_excel_filtered = df_excel.drop(random_indices)
#     data_json_filtered = [data_json[i] for i in range(len(data_json)) if i not in random_indices]
#
#     # 将剩余的5000条均分成10份，每份500条
#     n = len(df_excel_filtered)  # 5000条数据
#     step = n // 10  # 每份500条
#
#     datasets = []
#     for i in range(10):
#         # 随机选择数据集，依次增加
#         indices = random.sample(range(i * step, (i + 1) * step), step)
#
#         # 获取对应的数据
#         excel_subset = df_excel_filtered.iloc[indices]
#         json_subset = [data_json_filtered[j] for j in indices]
#
#         datasets.append((excel_subset, json_subset))
#
#     # 现在创建每个数据集，并逐步扩展
#     final_datasets = []
#     for i in range(10):
#         current_excel_data = pd.concat([datasets[j][0] for j in range(i + 1)], ignore_index=True)
#         current_json_data = [item for j in range(i + 1) for item in datasets[j][1]]
#
#         final_datasets.append((current_excel_data, current_json_data))
#
#     return final_datasets
#
#
# def save_datasets(final_datasets, output_dir):
#     """
#     保存最终的10个数据集，每个数据集保存为Excel文件和JSON文件。
#
#     :param final_datasets: 包含10个数据集的列表，每个数据集包含Excel数据和JSON数据
#     :param output_dir: 输出目录路径
#     """
#     # 如果输出目录不存在，则创建目录
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     for i, (excel_data, json_data) in enumerate(final_datasets):
#         # 保存Excel文件，文件名格式为 train-data-数字.xlsx
#         excel_filename = os.path.join(output_dir, f'train-data-{i + 1}.xlsx')
#         excel_data.to_excel(excel_filename, index=False)
#
#         # 保存JSON文件，文件名格式为 train-data-数字.json
#         json_filename = os.path.join(output_dir, f'train-data-{i + 1}.json')
#         with open(json_filename, 'w', encoding='utf-8') as f:
#             json.dump(json_data, f, ensure_ascii=False, indent=4)
#
#     print(f"数据集已保存至 {output_dir} 目录下")
#
#
# # 使用示例
# excel_file = '../cleaned/train-data-cleand.xlsx'
# json_file = '../raw/train-data.json'
# output_dir = './'
#
# final_datasets = create_datasets(excel_file, json_file)
#
# # 保存数据集
# save_datasets(final_datasets, output_dir)


import pandas as pd
import json
import os


def create_datasets(excel_file, full_json_file, partial_json_file):
    """
    根据部分集 JSON 中的 "sentence" 和 "question_word" 字段，
    在全集 JSON 中查找匹配的记录（忽略部分集中的 id），
    并利用匹配到的全集记录对应的索引，从全集 Excel 中选取对应的行。

    最终输出的数据集中：
      - Excel 文件保留全集的表头和对应行；
      - JSON 文件采用全集中匹配到的记录（即 id 以全集为准）。

    :param excel_file: 全集 Excel 文件路径
    :param full_json_file: 全集 JSON 文件路径（与 Excel 数据顺序一一对应）
    :param partial_json_file: 部分集 JSON 文件路径（仅用于匹配 "sentence" 和 "question_word"）
    :return: 包含一个元组 (筛选后的 Excel DataFrame, 对应的全集 JSON 记录列表) 的列表
    """
    # 读取全集 Excel 与 JSON 文件
    df_excel = pd.read_excel(excel_file)
    with open(full_json_file, 'r', encoding='utf-8') as f:
        full_json = json.load(f)
    with open(partial_json_file, 'r', encoding='utf-8') as f:
        partial_json = json.load(f)

    # 构建从 (sentence, question_word) 到全集 JSON 中索引的映射
    mapping = {}
    for idx, record in enumerate(full_json):
        key = (record.get("sentence"), record.get("question_word"))
        mapping[key] = idx

    indices = []

    with open('../raw/test-data.json', 'r', encoding='utf-8') as f:
        test_json = json.load(f)

    # 遍历部分集 JSON，根据 (sentence, question_word) 匹配全集记录
    for record in partial_json:
        key = (record.get("sentence"), record.get("question_word"))
        if key in mapping:
            indices.append(mapping[key])
        else:
            sentence = record.get("sentence")
            ok = None
            for item in test_json:
                if item['sentence'] == sentence:
                    ok = True
                    break
            if ok is None:
                print(sentence)
            # print("Warning: no matching record for", key)
    # 为保证输出顺序与全集一致，根据全集顺序对索引排序
    indices = sorted(indices)

    # 根据索引筛选出对应的 Excel 行，保留原有表头
    filtered_excel_df = df_excel.iloc[indices].copy()

    # 根据索引构建输出 JSON 数据，直接使用全集中的记录（即 id 以全集为准）
    matched_json_data = [full_json[i] for i in indices]

    # 返回数据集（这里只生成一个数据集）
    final_datasets = [(filtered_excel_df, matched_json_data)]
    return final_datasets


def save_datasets(final_datasets, output_dir):
    """
    保存最终数据集为 Excel 文件（xlsx格式）和 JSON 文件。

    :param final_datasets: 包含数据集元组 (Excel数据, JSON数据) 的列表
    :param output_dir: 输出目录路径
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (excel_data, json_data) in enumerate(final_datasets):
        # 保存 Excel 文件，确保表头与全集一致
        excel_filename = os.path.join(output_dir, f'train-data-QACD-data.xlsx')
        excel_data.to_excel(excel_filename, index=False)

        # 保存 JSON 文件（采用全集中的数据，即 id 以全集为准）
        json_filename = os.path.join(output_dir, f'train-data-QACD-data.json')
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"数据集已保存至 {output_dir} 目录下")


# 使用示例
if __name__ == "__main__":
    excel_file = '../cleaned/train-data-cleand.xlsx'
    full_json_file = '../raw/train-data.json'
    partial_json_file = '../raw/All-QACD-data.json'
    output_dir = '../cleaned/'

    final_datasets = create_datasets(excel_file, full_json_file, partial_json_file)
    save_datasets(final_datasets, output_dir)
