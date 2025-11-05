import pandas as pd
import numpy as np

# 读取合并文件的路径
merged_file_path = 'F:/Research_Project/EP_ukb/Revision_machine_learning/pa_processed/merged_data_day_aligned.csv'

# 读取合并的CSV文件
df = pd.read_csv(merged_file_path)


def calculate_daily_average(row):
    # 提取PA值（跳过'eid'列）
    pa_values = row.iloc[1:].values

    # 移除NaN值
    pa_values = pa_values[~np.isnan(pa_values)]

    # 计算完整天数
    num_days = len(pa_values) // 1440

    if num_days == 0:
        return pd.Series([row['eid']] + [np.nan] * 1440)

    # 重塑数组为每天1440分钟
    daily_values = pa_values[:num_days * 1440].reshape(num_days, 1440)

    # 计算每分钟的平均值
    daily_average = np.mean(daily_values, axis=0)

    # 返回eid和1440个平均值
    return pd.Series([row['eid']] + list(daily_average))


# 应用函数到每一行
result_df = df.apply(calculate_daily_average, axis=1)

# 设置列名
result_df.columns = ['eid'] + [f'minute_{i}' for i in range(1440)]

# 保存结果
output_file_path = 'F:/Research_Project/EP_ukb/Revision_machine_learning/pa_processed/daily_average_PA_aligned.csv'
result_df.to_csv(output_file_path, index=False, float_format='%.2f')

print(f"Daily average PA values have been saved to: {output_file_path}")