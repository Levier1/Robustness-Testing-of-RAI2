import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 加载数据
file_path = '/home/featurize/work/RAI2project/RAI2result/RotationProcessed_Test_accuracy.csv'
data = pd.read_csv(file_path)

# 设置保存路径
save_path = '/home/featurize/work/RAI2project/Fig&Tab/noiseFig'
os.makedirs(save_path, exist_ok=True)

# 检查 Intersection 列的内容，确保其格式正确
print("检查 Intersection 列内容：")
print(data['Intersection'].unique())

# 尝试将 Intersection 转换为浮点数，并处理异常值
def clean_intersection(value):
    try:
        return float(value)
    except ValueError:
        print(f"无法转换的 Intersection 值: {value}")
        return None

data['Intersection'] = data['Intersection'].apply(clean_intersection)

# 删除无法转换的行
data = data.dropna(subset=['Intersection'])

# 遍历数据集并绘制图形
datasets = data['Dataset'].unique()

for dataset in datasets:
    subset = data[data['Dataset'] == dataset]

    plt.figure()
    for model in subset['Model'].unique():
        model_data = subset[subset['Model'] == model]
        plt.plot(model_data['Intersection'], model_data['Top-1'], label=f"{model} Top-1")
        plt.plot(model_data['Intersection'], model_data['Top-5'], label=f"{model} Top-5")

    # 设置横轴刻度，确保显示所有预期的比例
    plt.xticks(np.arange(0.0, 1.1, 0.1))  # 强制横轴刻度从 0.0 到 1.0，步长为 0.1

    # 添加图表信息
    plt.xlabel("Intersection Ratio")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset} Rotation-UC")
    plt.legend()
    plt.grid(True)

    # 生成安全文件名，将数据集名称中的特殊字符替换为下划线
    safe_dataset_name = dataset.replace(" ", "_").replace("-", "_")
    file_name = f"{safe_dataset_name}_Rotation-UC_Model_Accuracy.png"

    # 保存图表到指定路径
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()  # 关闭图表以释放内存

print(f"图表已成功保存至 {save_path} 路径下")
