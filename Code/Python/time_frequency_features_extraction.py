import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from multiprocessing import freeze_support, cpu_count
import warnings
import os

# 忽略警告信息
warnings.filterwarnings("ignore")

def main():
    # 读取CSV文件
    print("正在读取数据...")
    df = pd.read_csv('F:/Research_Project/EP_ukb/Revision_machine_learning/original_time_series.csv')

    # 准备数据格式
    print("正在准备数据...")
    id_vars = ['eid']
    value_vars = [col for col in df.columns if col.startswith('minute_')]
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='time', value_name='value')

    # 将'time'列转换为数值型
    df_long['time'] = df_long['time'].str.replace('minute_', '').astype(int)

    # 确定可用的CPU核心数
    n_jobs = max(1, cpu_count() - 1)  # 使用全部核心数减1，至少为1
    print(f"使用 {n_jobs} 个CPU核心进行处理")

    # 使用tsfresh提取全部特征
    print("正在提取特征，这可能需要一些时间...")
    features = extract_features(df_long, column_id="eid", column_sort="time", column_value="value",
                                default_fc_parameters=EfficientFCParameters(), n_jobs=n_jobs)

    # 重置索引，确保'eid'成为一个列
    features = features.reset_index()

    # 保存结果到新的CSV文件
    print("正在保存结果...")
    output_path = 'F:/Research_Project/EP_ukb/Revision_machine_learning/PA_feature.csv'
    features.to_csv(output_path, index=False)

    print(f"特征提取完成，结果已保存到 '{output_path}'")
    print(f"提取的特征数量: {features.shape[1] - 1}")  # 减去eid列

if __name__ == "__main__":
    freeze_support()
    main()