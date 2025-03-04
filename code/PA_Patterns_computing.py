import pandas as pd
import numpy as np
import itertools


def calculate_metrics(file_path, output_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 将字符串转换为数字列表
    df['Light'] = df['Light_Dayhouraverage'].apply(lambda x: [float(i) for i in x.split(',')])
    df['MVPA'] = df['Moderate_Vigorous_Day_hour_average'].apply(lambda x: [float(i) for i in x.split(',')])
    df['Sedentary'] = df['Sedentary_Day_hour_average'].apply(lambda x: [float(i) for i in x.split(',')])

    # 创建结果DataFrame
    result_df = pd.DataFrame()
    result_df['eid'] = df['eid']

    # 1. 计算总时间（分钟/天）
    result_df['Light_Total_Min'] = df['Light'].apply(lambda x: sum(x) * 60)
    result_df['MVPA_Total_Min'] = df['MVPA'].apply(lambda x: sum(x) * 60)
    result_df['Sedentary_Total_Min'] = df['Sedentary'].apply(lambda x: sum(x) * 60)

    # 2. 计算活动占比
    total_time = result_df['Light_Total_Min'] + result_df['MVPA_Total_Min'] + result_df['Sedentary_Total_Min']
    result_df['Light_Percentage'] = result_df['Light_Total_Min'] / total_time * 100
    result_df['MVPA_Percentage'] = result_df['MVPA_Total_Min'] / total_time * 100
    result_df['Sedentary_Percentage'] = result_df['Sedentary_Total_Min'] / total_time * 100

    # 3. 活动高峰和频率
    result_df['Light_Peak_Hour'] = df['Light'].apply(lambda x: np.argmax(x))
    result_df['MVPA_Peak_Hour'] = df['MVPA'].apply(lambda x: np.argmax(x))
    result_df['Sedentary_Peak_Hour'] = df['Sedentary'].apply(lambda x: np.argmax(x))
    result_df['Activity_Frequency_Min'] = df.apply(
        lambda row: sum((np.array(row['Light']) + np.array(row['MVPA'])) > 0) * 60, axis=1)

    # 4. 不同时间段的活动时间
    def time_period_activity(row, start, end):
        light = sum(row['Light'][start:end]) * 60
        mvpa = sum(row['MVPA'][start:end]) * 60
        sedentary = sum(row['Sedentary'][start:end]) * 60
        return pd.Series([light, mvpa, sedentary])

    result_df[['Morning_Light', 'Morning_MVPA', 'Morning_Sedentary']] = df.apply(
        lambda row: time_period_activity(row, 6, 12), axis=1)
    result_df[['Afternoon_Light', 'Afternoon_MVPA', 'Afternoon_Sedentary']] = df.apply(
        lambda row: time_period_activity(row, 12, 18), axis=1)
    result_df[['Evening_Light', 'Evening_MVPA', 'Evening_Sedentary']] = df.apply(
        lambda row: time_period_activity(row, 18, 24), axis=1)

    # 5. 最长连续活动和久坐时间
    def longest_consecutive(arr):
        arr = np.array(arr) > 0
        consecutive_lengths = [sum(1 for _ in group) for val, group in itertools.groupby(arr) if val]
        return max(consecutive_lengths) if consecutive_lengths else 0

    # 在 calculate_metrics 函数中更新这一行
    result_df['Longest_Activity_Hours'] = df.apply(
        lambda row: longest_consecutive(np.array(row['Light']) + np.array(row['MVPA'])), axis=1)

    # 同样更新久坐时间的计算
    result_df['Longest_Sedentary_Hours'] = df['Sedentary'].apply(longest_consecutive)

    # 6. 活动时间的标准差和变异系数
    result_df['Activity_Std'] = df.apply(lambda row: np.std(np.array(row['Light']) + np.array(row['MVPA'])), axis=1)
    result_df['Activity_CV'] = result_df['Activity_Std'] / df.apply(
        lambda row: np.mean(np.array(row['Light']) + np.array(row['MVPA'])), axis=1)

    # 7. 连续久坐超过1小时的次数
    def count_long_sedentary(arr):
        return sum(sum(1 for _ in group) >= 1 for val, group in itertools.groupby(arr) if val >= 0.5)

    result_df['Long_Sedentary_Count'] = df['Sedentary'].apply(count_long_sedentary)

    # 保存结果到新的CSV文件
    result_df.to_csv(output_path, index=False)

    return result_df

# 使用函数
input_file = 'LAP_SB_MVPA_FILTERED.csv'
output_file = 'LAP_SB_MVPA_DERIVED.csv'
result_df = calculate_metrics(input_file, output_file)
print("数据已保存到", output_file)
print(result_df.head())  # 显示前几行数据以供检查