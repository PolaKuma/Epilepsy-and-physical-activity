import pandas as pd
import numpy as np
import itertools

def calculate_metrics(file_path, output_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 将字符串转换为数字列表（每小时为单位）
    df['Light'] = df['Light_Dayhouraverage'].apply(lambda x: [float(i) for i in x.split(',')])
    df['MVPA'] = df['Moderate_Vigorous_Day_hour_average'].apply(lambda x: [float(i) for i in x.split(',')])
    df['Sedentary'] = df['Sedentary_Day_hour_average'].apply(lambda x: [float(i) for i in x.split(',')])

    # 创建结果DataFrame
    result_df = pd.DataFrame()
    result_df['eid'] = df['eid']

    # 1. 计算总时间（小时/天）
    result_df['Light_Total_Hour'] = df['Light'].apply(sum)
    result_df['MVPA_Total_Hour'] = df['MVPA'].apply(sum)
    result_df['Sedentary_Total_Hour'] = df['Sedentary'].apply(sum)

    # 2. 计算活动占比（%），分母为全天活动小时数
    total_time = result_df['Light_Total_Hour'] + result_df['MVPA_Total_Hour'] + result_df['Sedentary_Total_Hour']
    result_df['Light_Percentage'] = result_df['Light_Total_Hour'] / total_time * 100
    result_df['MVPA_Percentage'] = result_df['MVPA_Total_Hour'] / total_time * 100
    result_df['Sedentary_Percentage'] = result_df['Sedentary_Total_Hour'] / total_time * 100

    # 3. 各活动的高峰小时（0-23）
    result_df['Light_Peak_Hour'] = df['Light'].apply(np.argmax)
    result_df['MVPA_Peak_Hour'] = df['MVPA'].apply(np.argmax)
    result_df['Sedentary_Peak_Hour'] = df['Sedentary'].apply(np.argmax)

    # 活动频率（有活动的小时数）
    result_df['Activity_Frequency_Hour'] = df.apply(
        lambda row: sum((np.array(row['Light']) + np.array(row['MVPA'])) > 0), axis=1)

    # 4. 不同时间段的活动时间（小时）
    def time_period_activity(row, start, end):
        light = sum(row['Light'][start:end])
        mvpa = sum(row['MVPA'][start:end])
        sedentary = sum(row['Sedentary'][start:end])
        return pd.Series([light, mvpa, sedentary])

    result_df[['Morning_Light', 'Morning_MVPA', 'Morning_Sedentary']] = df.apply(
        lambda row: time_period_activity(row, 6, 12), axis=1)
    result_df[['Afternoon_Light', 'Afternoon_MVPA', 'Afternoon_Sedentary']] = df.apply(
        lambda row: time_period_activity(row, 12, 18), axis=1)
    result_df[['Evening_Light', 'Evening_MVPA', 'Evening_Sedentary']] = df.apply(
        lambda row: time_period_activity(row, 18, 24), axis=1)

    # 5. 最长连续活动和久坐时间（小时）
    def longest_consecutive(arr):
        arr = np.array(arr) > 0
        lengths = [sum(1 for _ in group) for val, group in itertools.groupby(arr) if val]
        return max(lengths) if lengths else 0

    result_df['Longest_Activity_Hours'] = df.apply(
        lambda row: longest_consecutive(np.array(row['Light']) + np.array(row['MVPA'])), axis=1)
    result_df['Longest_Sedentary_Hours'] = df['Sedentary'].apply(longest_consecutive)

    # 6. 活动时间的标准差和变异系数（小时制）
    activity_sum = df.apply(lambda row: np.array(row['Light']) + np.array(row['MVPA']), axis=1)
    result_df['Activity_Std'] = activity_sum.apply(np.std)
    result_df['Activity_CV'] = result_df['Activity_Std'] / activity_sum.apply(np.mean)

    # 7. 连续久坐≥1小时的次数（小时为单位）
    def count_long_sedentary(arr):
        arr = np.array(arr)
        return sum(sum(1 for _ in group) >= 1 for val, group in itertools.groupby(arr >= 1) if val)

    result_df['Long_Sedentary_Count'] = df['Sedentary'].apply(count_long_sedentary)

    # 保存结果到新的CSV文件
    result_df.to_csv(output_path, index=False)

    return result_df

# 使用函数
input_file = 'E:/EP_ukb/PA_PROCESSING/LAP_SB_MVPA_FILTERED.csv'
output_file = 'E:/EP_ukb/PA_PROCESSING/LAP_SB_MVPA_DERIVED.csv'
result_df = calculate_metrics(input_file, output_file)
print("数据已保存到", output_file)
print(result_df.head())