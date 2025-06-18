import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 源文件夹和目标文件夹路径
source_folder = 'E:/EP_ukb/machinelearning/EP_ML_FILE/'
destination_folder = 'E:/EP_ukb/machinelearning/Minutes_epoch/'

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)


def process_file(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 获取列名
    column_name = df.columns[0]

    # 从列名中提取开始时间
    start_time_str = column_name.split(' - ')[1]
    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')

    # 创建时间索引
    time_index = pd.date_range(start=start_time, periods=len(df), freq='5s')
    df['timestamp'] = time_index

    # 设置时间索引
    df.set_index('timestamp', inplace=True)

    # 重命名PA强度列
    df.rename(columns={column_name: 'PA_intensity'}, inplace=True)

    # 重采样为每分钟数据，使用平均值
    df_resampled = df.resample('1min').mean()

    return df_resampled['PA_intensity']


# 用于存储所有处理后的数据
all_processed_data = {}
max_length = 0

# 处理文件夹中的所有CSV文件
for filename in os.listdir(source_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(source_folder, filename)
        try:
            # 从文件名中提取eid
            eid = filename.split('_')[0]

            processed_data = process_file(file_path)

            # 存储处理后的数据
            all_processed_data[eid] = processed_data.values

            # 更新最大长度
            max_length = max(max_length, len(processed_data))

            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print("All files have been processed.")

# 创建合并的DataFrame
merged_data = []
for eid, data in all_processed_data.items():
    # 如果数据长度小于最大长度，用NaN填充
    padded_data = np.pad(data, (0, max_length - len(data)), 'constant', constant_values=np.nan)
    merged_data.append([eid] + padded_data.tolist())

# 创建列名
column_names = ['eid'] + [f'minute_{i + 1}' for i in range(max_length)]

# 创建DataFrame
merged_df = pd.DataFrame(merged_data, columns=column_names)

# 保存合并的CSV文件
merged_file_path = os.path.join(destination_folder, 'merged_data.csv')
merged_df.to_csv(merged_file_path, index=False, float_format='%.2f')

print("Merged data has been saved to: merged_data.csv")