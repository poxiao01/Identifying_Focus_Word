import pandas as pd

# 读取两个 Excel 文件
df1 = pd.read_excel("../../data/cleaned/All-train-data.xlsx")
df2 = pd.read_excel("../../data/cleaned/Train-data-cleand.xlsx")

# 比较两个 DataFrame
if df1.equals(df2):
    print("两个文件完全一致")
else:
    print("两个文件不一致")
