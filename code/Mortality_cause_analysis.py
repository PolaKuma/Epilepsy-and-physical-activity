import pandas as pd
import re

# 读取CSV文件
# 请替换为您的实际文件路径
#df = pd.read_csv('nGP_death.csv')
df = pd.read_csv('death_causes_ukb.csv')
# 更新死因模式
death_patterns = {
   # 'epilepsy': r'G40\.?[0-9]*|G41\.?[0-9]*',
    'cardiovascular': r'I1[0-5]\.?[0-9]*|I02\.0|I01\.?[0-9]*|I0[5-9]\.?[0-9]*|I2[0-5]\.?[0-9]*|I27\.?[0-9]*|I3[0-9]\.?[0-9]*|I4[0-9]\.?[0-9]*|I5[0-2]\.?[0-9]*',
     'cerebrovascular': r'I6[0-9]\.?[0-9]*|G46\.?[0-9]*',
    'cancer': r'C[0-9]{2}\.?[0-9]*'
}

def check_cause_of_death(row, pattern):
    primary = str(row['primary_cause'])
    secondary = str(row['secondary_causes'])
    return 1 if re.search(pattern, primary) or re.search(pattern, secondary) else 0

# 为每种死因创建新列
for cause, pattern in death_patterns.items():
    df[f'{cause}_death'] = df.apply(lambda row: check_cause_of_death(row, pattern), axis=1)

# 选择需要的列
#columns_to_keep = ['eid', 'epilepsy_death', 'cardiovascular_death', 'cerebrovascular_death', 'cancer_death']
columns_to_keep = ['eid', 'cardiovascular_death', 'cerebrovascular_death', 'cancer_death']
result_df = df[columns_to_keep]

# 保存结果到新的CSV文件
# 请替换为您想要保存的文件路径
result_df.to_csv('E:/EP_ukb/Death_cause_analysis/death_cause_all_death.csv', index=False)

print("处理完成。结果已保存到新的CSV文件中。")