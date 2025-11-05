import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# 读取原始药物数据并筛查抗癫痫药物
def screen_antiseizure_medications(input_file, output_file):
    print(f"Reading data file: {input_file}")
    try:
        # 尝试读取CSV文件
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Successfully read data, {len(df)} rows total")

    # 定义抗癫痫药物字典，格式为 {标准名称: [搜索关键词列表]}
    antiseizure_meds = {
        'VPA': ['valpro', 'epilim', 'depakote'],
        'CBZ': ['carbamazepine', 'tegretol'],
        'LEV': ['levetiracetam', 'keppra'],
        'LTG': ['lamotrigine', 'lamictal'],
        'OXC': ['oxcarbazepine', 'trileptal'],
        'Phenobarbital': ['phenobarb'],
        'Phenytoin': ['phenyto', 'epanutin'],
        'CNP': ['clonazepam', 'rivotril'],
        'TPM': ['topiramate', 'topamax'],
        'GBP': ['gabapentin', 'neurontin'],
        'PGB': ['pregabalin', 'lyrica'],
        'Primidone': ['primidone', 'mysoline'],
        'Ethosuximide': ['ethosuximide', 'zarontin'],
        'Vigabatrin': ['vigabatrin', 'sabril'],
        'Tiagabine': ['tiagabine', 'gabitril'],
        'Beclamide': ['beclamide'],
        'Clobazam': ['clobazam', 'frisium'],
        'ZNS': ['zonisamide', 'zonegran'],
        'LCM': ['lacosamide', 'vimpat'],
        'Perampanel': ['perampanel', 'fycompa'],
        'Brivaracetam': ['brivaracetam', 'briviact'],
        'Eslicarbazepine': ['eslicarbazepine', 'aptiom', 'zebinix'],
        'Rufinamide': ['rufinamide', 'banzel', 'inovelon'],
        'Cenobamate': ['cenobamate', 'xcopri', 'ontozry'],
        'Stiripentol': ['stiripentol', 'diacomit'],
        'Felbamate': ['felbamate', 'felbatol'],
        'Methsuximide': ['methsuximide', 'celontin'],
        'Trimethadione': ['trimethadione'],
        'Paramethadione': ['paramethadione']
    }

    # 定义药物机制分类
    mechanism_categories = {
        # 1: Sodium channel blockers
        1: ['CBZ', 'LTG', 'OXC', 'Phenytoin', 'LCM', 'Eslicarbazepine'],

        # 2: Gamma-aminobutyric acid analog
        2: ['CNP', 'GBP', 'PGB', 'Phenobarbital', 'Primidone',
            'Vigabatrin', 'Tiagabine', 'Clobazam'],

        # 3: Synaptic vesicle protein 2A binding
        3: ['LEV', 'Brivaracetam'],

        # 4: Multiple mechanisms
        4: ['VPA', 'TPM', 'ZNS', 'Felbamate', 'Rufinamide',
            'Perampanel', 'Cenobamate', 'Stiripentol'],

        # 5: Other/Not clearly classified
        5: ['Ethosuximide', 'Beclamide', 'Methsuximide', 'Trimethadione', 'Paramethadione']
    }

    # 定义包含钠通道阻滞作用的药物，包括主要和次要机制
    sodium_channel_drugs = [
        # 主要为钠通道阻滞剂的药物
        'CBZ', 'LTG', 'OXC', 'Phenytoin', 'LCM', 'Eslicarbazepine',
        # 多重机制中具有钠通道阻滞作用的药物
        'TPM', 'ZNS', 'Felbamate', 'Rufinamide', 'Cenobamate', 'VPA'  # Added VPA
    ]

    # 创建反向查找字典，用于根据药物名称查找机制类别
    drug_to_mechanism = {}
    for mech_id, drug_list in mechanism_categories.items():
        for drug in drug_list:
            drug_to_mechanism[drug] = mech_id

    # 创建结果DataFrame，增加any_sodium_channel_effect列
    result_df = pd.DataFrame(columns=['eid', 'is_on_asm', 'drug_name', 'drug_count',
                                      'mechanism_category', 'sodium_channel_blocker',
                                      'gaba_analog', 'sv2a_binding', 'multiple_mechanisms',
                                      'any_sodium_channel_effect'])

    # 提取eid列
    result_df['eid'] = df['eid']
    result_df['is_on_asm'] = False
    result_df['drug_name'] = ''
    result_df['drug_count'] = 0
    result_df['mechanism_category'] = 0  # 0代表不服药
    result_df['sodium_channel_blocker'] = 0
    result_df['gaba_analog'] = 0
    result_df['sv2a_binding'] = 0
    result_df['multiple_mechanisms'] = 0
    result_df['any_sodium_channel_effect'] = 0  # 新增列

    print("Starting ASM screening...")

    # 检查每行数据
    for index, row in df.iterrows():
        detected_drugs = []

        # 遍历每个药物单元格(从第2列开始)
        for col in df.columns[1:]:
            cell_value = str(row[col]).lower()  # 转换为小写
            if pd.isna(row[col]) or cell_value == 'nan' or cell_value == '':
                continue

            # 检查是否包含任何抗癫痫药物关键词
            for drug_standard_name, keywords in antiseizure_meds.items():
                for keyword in keywords:
                    if keyword.lower() in cell_value:
                        detected_drugs.append(drug_standard_name)
                        break  # 找到一个关键词就跳出内层循环

        # 如果找到任何抗癫痫药物
        if detected_drugs:
            unique_drugs = list(set(detected_drugs))  # 去重
            result_df.at[index, 'is_on_asm'] = True
            result_df.at[index, 'drug_name'] = '; '.join(unique_drugs)
            result_df.at[index, 'drug_count'] = len(unique_drugs)

            # 确定机制类别
            mechanism_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

            # 计算每个机制类别的药物数量
            for drug in unique_drugs:
                mechanism = drug_to_mechanism.get(drug, 5)  # 默认为类别5 (其他)
                mechanism_counts[mechanism] += 1

                # 同时更新对应的机制标志列
                if mechanism == 1:
                    result_df.at[index, 'sodium_channel_blocker'] = 1
                elif mechanism == 2:
                    result_df.at[index, 'gaba_analog'] = 1
                elif mechanism == 3:
                    result_df.at[index, 'sv2a_binding'] = 1
                elif mechanism == 4:
                    result_df.at[index, 'multiple_mechanisms'] = 1

                # 检查药物是否具有钠通道阻滞作用（包括主作用和次要作用）
                if drug in sodium_channel_drugs:
                    result_df.at[index, 'any_sodium_channel_effect'] = 1

            # 确定主要机制类别 (方法1: 使用最多药物的类别)
            main_mechanism = max(mechanism_counts, key=mechanism_counts.get)

            # 如果有多个类别药物数量相同，优先选择数字较小的类别
            max_count = mechanism_counts[main_mechanism]
            if max_count > 0:
                # 查找所有具有相同最大数量的机制
                tied_mechanisms = [m for m, count in mechanism_counts.items() if count == max_count]
                main_mechanism = min(tied_mechanisms)  # 选择数字最小的类别

            result_df.at[index, 'mechanism_category'] = main_mechanism

        # 显示进度
        if (index + 1) % 1000 == 0 or (index + 1) == len(df):
            print(f"Processed {index + 1}/{len(df)} rows")

    # 生成统计报告并保存结果
    generate_statistics(result_df, output_file, mechanism_categories, sodium_channel_drugs)
    return result_df


# 生成详细统计报告
def generate_statistics(result_df, output_file, mechanism_categories, sodium_channel_drugs):
    # 基本统计信息
    total_patients = len(result_df)
    total_asm_patients = result_df['is_on_asm'].sum()
    asm_percentage = (total_asm_patients / total_patients) * 100

    print("\n" + "=" * 50)
    print("ASM screening results summary")
    print("=" * 50)
    print(f"Total participants: {total_patients}")
    print(f"Participants taking ASM: {total_asm_patients} ({asm_percentage:.2f}%)")

    # 钠通道阻滞效应统计
    sodium_count = result_df['any_sodium_channel_effect'].sum()
    sodium_percentage = (sodium_count / total_asm_patients) * 100 if total_asm_patients > 0 else 0
    print(
        f"Participants taking drugs with sodium channel blocking effect: {sodium_count} ({sodium_percentage:.2f}% of ASM users)")

    # 机制类别统计
    mechanism_names = {
        0: "No medication",
        1: "Sodium channel blockers",
        2: "GABA analogs",
        3: "SV2A binding",
        4: "Multiple mechanisms",
        5: "Others/Not clearly classified"
    }

    # 添加机制分类统计
    mech_counts = result_df['mechanism_category'].value_counts().sort_index()
    print("\nClassification by mechanism of action:")
    for mech, count in mech_counts.items():
        if mech > 0:  # 不显示无药物类别
            percentage = (count / total_asm_patients) * 100
            print(f"{mechanism_names.get(mech, f'Category {mech}')}: {count} participants ({percentage:.2f}%)")

    # 单药、双药和多药情况统计
    drug_count_stats = result_df['drug_count'].value_counts().sort_index()

    print("\nMedication count distribution:")
    for count, num_patients in drug_count_stats.items():
        if count > 0:  # 只显示服药的患者
            percentage = (num_patients / total_asm_patients) * 100
            print(f"{count} medications: {num_patients} participants ({percentage:.2f}%)")

    # 各药物使用统计
    all_drugs = []
    for drugs in result_df[result_df['is_on_asm']]['drug_name']:
        all_drugs.extend(drugs.split('; '))

    drug_counter = Counter(all_drugs)

    print("\nASM usage statistics:")
    for drug, count in drug_counter.most_common():
        percentage = (count / total_asm_patients) * 100
        sodium_effect = "Yes" if drug in sodium_channel_drugs else "No"
        print(f"{drug}: {count} participants ({percentage:.2f}%) - Sodium channel effect: {sodium_effect}")

    # 常见药物组合统计
    if total_asm_patients > 0:
        multi_drug_patients = result_df[result_df['drug_count'] > 1]
        if len(multi_drug_patients) > 0:
            drug_combinations = multi_drug_patients['drug_name'].value_counts().head(10)

            print("\nMost common drug combinations (top 10):")
            for combo, count in drug_combinations.items():
                percentage = (count / total_asm_patients) * 100
                print(f"{combo}: {count} participants ({percentage:.2f}%)")

    # 不同机制类别组合统计
    mechanism_combinations = {}
    for _, row in result_df[result_df['is_on_asm']].iterrows():
        combo = []
        if row['sodium_channel_blocker'] == 1:
            combo.append("Sodium channel blockers")
        if row['gaba_analog'] == 1:
            combo.append("GABA analogs")
        if row['sv2a_binding'] == 1:
            combo.append("SV2A binding")
        if row['multiple_mechanisms'] == 1:
            combo.append("Multiple mechanisms")

        combo_str = " + ".join(combo) if combo else "Unknown"
        mechanism_combinations[combo_str] = mechanism_combinations.get(combo_str, 0) + 1

    print("\nMechanism combinations:")
    for combo, count in sorted(mechanism_combinations.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_asm_patients) * 100
        print(f"{combo}: {count} participants ({percentage:.2f}%)")

    # 保存结果
    result_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # 创建统计数据文件
    stats_file = output_file.replace('.csv', '_statistics.csv')

    # 药物使用统计
    drug_stats = pd.DataFrame(drug_counter.most_common(), columns=['Drug', 'Count'])
    drug_stats['Percentage'] = drug_stats['Count'] / total_asm_patients * 100
    drug_stats['Sodium_Channel_Effect'] = drug_stats['Drug'].apply(
        lambda x: "Yes" if x in sodium_channel_drugs else "No")

    # 药物数量分布
    count_stats = pd.DataFrame(drug_count_stats)
    count_stats.columns = ['Count']
    count_stats['Percentage'] = count_stats['Count'] / total_asm_patients * 100
    count_stats = count_stats.reset_index()
    count_stats.columns = ['Number_of_Drugs', 'Patient_Count', 'Percentage']
    count_stats = count_stats[count_stats['Number_of_Drugs'] > 0]  # 只保留服药患者

    # 药物机制分布
    mech_stats = pd.DataFrame(mech_counts)
    mech_stats.columns = ['Count']
    mech_stats['Percentage'] = mech_stats['Count'] / total_asm_patients * 100
    mech_stats = mech_stats.reset_index()
    mech_stats.columns = ['Mechanism_Category', 'Patient_Count', 'Percentage']
    mech_stats = mech_stats[mech_stats['Mechanism_Category'] > 0]  # 只保留服药患者

    # 添加机制名称
    mech_stats['Mechanism_Name'] = mech_stats['Mechanism_Category'].map(mechanism_names)

    # 机制组合分布
    combo_data = [(combo, count, count / total_asm_patients * 100) for combo, count in mechanism_combinations.items()]
    mech_combo_stats = pd.DataFrame(combo_data, columns=['Mechanism_Combination', 'Patient_Count', 'Percentage'])
    mech_combo_stats = mech_combo_stats.sort_values('Patient_Count', ascending=False)

    # 钠通道效应统计
    sodium_stats = pd.DataFrame({
        'Category': ['With Sodium Channel Effect', 'Without Sodium Channel Effect'],
        'Count': [result_df['any_sodium_channel_effect'].sum(),
                  total_asm_patients - result_df['any_sodium_channel_effect'].sum()],
        'Percentage': [sodium_percentage, 100 - sodium_percentage]
    })

    # 保存统计结果
    with pd.ExcelWriter(stats_file.replace('.csv', '.xlsx')) as writer:
        summary = pd.DataFrame({
            'Metric': ['Total Patients', 'Patients on ASM', 'Percentage on ASM',
                       'Patients with Sodium Channel Blocking Drugs',
                       'Percentage of ASM Users with Sodium Channel Blocking Drugs'],
            'Value': [total_patients, total_asm_patients, f"{asm_percentage:.2f}%",
                      sodium_count, f"{sodium_percentage:.2f}%"]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
        drug_stats.to_excel(writer, sheet_name='Drug_Usage', index=False)
        count_stats.to_excel(writer, sheet_name='Drug_Count', index=False)
        mech_stats.to_excel(writer, sheet_name='Mechanism_Categories', index=False)
        mech_combo_stats.to_excel(writer, sheet_name='Mechanism_Combinations', index=False)
        sodium_stats.to_excel(writer, sheet_name='Sodium_Channel_Effect', index=False)

        # 如果有多药患者，保存药物组合
        if len(multi_drug_patients) > 0:
            combo_stats = pd.DataFrame(drug_combinations).reset_index()
            combo_stats.columns = ['Combination', 'Count']
            combo_stats['Percentage'] = combo_stats['Count'] / total_asm_patients * 100
            combo_stats.to_excel(writer, sheet_name='Drug_Combinations', index=False)

    print(f"Statistical report saved to: {stats_file.replace('.csv', '.xlsx')}")

    # 创建可视化图表
    create_visualizations(result_df, drug_stats, count_stats, mech_stats, mech_combo_stats, sodium_stats, output_file)


# 创建可视化图表
def create_visualizations(result_df, drug_stats, count_stats, mech_stats, mech_combo_stats, sodium_stats, output_file):
    output_base = output_file.replace('.csv', '')

    # 设置matplotlib英文显示
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    # 图1: 前10种最常用药物
    plt.figure(figsize=(12, 8))
    top_drugs = drug_stats.head(10).copy()

    # 设置药物柱状图颜色，根据是否有钠通道阻滞效应
    bar_colors = ['#ff9999' if effect == 'Yes' else '#66b3ff' for effect in top_drugs['Sodium_Channel_Effect']]

    # 绘制柱状图
    ax = sns.barplot(x='Drug', y='Count', data=top_drugs, palette=bar_colors)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff9999', label='With sodium channel effect'),
        Patch(facecolor='#66b3ff', label='Without sodium channel effect')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title('Top 10 commonly used ASMs', fontsize=16)
    plt.xlabel('Medicine', fontsize=14)
    plt.ylabel('Number of participants', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_base}_top_drugs.png", dpi=300)

    # 图2: 药物数量分布
    plt.figure(figsize=(10, 7))
    sns.barplot(x='Number_of_Drugs', y='Patient_Count', data=count_stats)
    plt.title('ASM distribution', fontsize=16)
    plt.xlabel('Number of medicines', fontsize=14)
    plt.ylabel('Number of participants', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_base}_drug_count.png", dpi=300)

    # 图3: 服药vs不服药比例饼图
    plt.figure(figsize=(9, 9))
    asm_counts = [result_df['is_on_asm'].sum(), len(result_df) - result_df['is_on_asm'].sum()]
    plt.pie(asm_counts, labels=['With ASM', 'Without ASM'],
            autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    plt.title('ASM Usage', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{output_base}_asm_usage.png", dpi=300)

    # 图4: 药物作用机制分布
    plt.figure(figsize=(12, 8))
    # 确保有正确的顺序
    mech_order = mech_stats.sort_values('Mechanism_Category')
    sns.barplot(x='Mechanism_Name', y='Patient_Count', data=mech_order)
    plt.title('Distribution of ASM mechanisms', fontsize=16)
    plt.xlabel('Mechanism', fontsize=14)
    plt.ylabel('Number of participants', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_base}_mechanism_distribution.png", dpi=300)

    # 图5: 前5种最常见机制组合
    plt.figure(figsize=(14, 8))
    top_combos = mech_combo_stats.head(5)
    sns.barplot(x='Mechanism_Combination', y='Patient_Count', data=top_combos)
    plt.title('Most common mechanism combinations', fontsize=16)
    plt.xlabel('Mechanism combination', fontsize=14)
    plt.ylabel('Number of participants', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_base}_mechanism_combinations.png", dpi=300)

    # 图6: 钠通道阻滞效应比例
    plt.figure(figsize=(9, 9))
    plt.pie(sodium_stats['Count'], labels=sodium_stats['Category'],
            autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    plt.title('Sodium Channel Blocking Effect', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{output_base}_sodium_channel_effect.png", dpi=300)

    print(f"Charts saved to: {output_base}_*.png")


# 使用示例
if __name__ == "__main__":
    input_file = "/20003.csv"  # 替换为您的输入文件路径
    output_file = "/asm_type_screening_results.csv"  # 替换为您的输出文件路径
    screen_antiseizure_medications(input_file, output_file)