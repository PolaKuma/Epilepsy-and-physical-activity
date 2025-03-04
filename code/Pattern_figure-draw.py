import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
import os

# 读取主CSV文件
main_df = pd.read_csv('nGP_combines.csv')

# 读取死因CSV文件
death_df = pd.read_csv('nGP_death.csv')

# 合并数据集
merged_df = main_df.merge(death_df, on='eid', how='left')

# 处理死因数据
death_causes = ['epilepsy_death', 'cardiovascular_death', 'cerebrovascular_death', 'cancer_death']
for cause in death_causes:
    merged_df[cause] = merged_df[cause].fillna(0)


# 定义函数来处理每个小时的数据
def process_hourly_data(column):
    return pd.DataFrame(merged_df[column].apply(lambda x: [float(i) * 60 for i in x.split(',')]).tolist())


# 处理三种活动类型的数据
light_data = process_hourly_data('Light_Dayhouraverage')
mvpa_data = process_hourly_data('Moderate_Vigorous_Day_hour_average')
sedentary_data = process_hourly_data('Sedentary_Day_hour_average')


def calculate_daily_total_p(group1_mask, group2_mask, variable):
    """
    计算每日总活动量的组间差异
    """
    group1_total = merged_df.loc[group1_mask, variable].dropna()
    group2_total = merged_df.loc[group2_mask, variable].dropna()

    stat, p_value = stats.mannwhitneyu(group1_total,
                                       group2_total,
                                       alternative='two-sided')
    return p_value


def format_p_value(p_value):
    """
    格式化P值显示
    """
    if p_value < 0.001:
        return "P < 0.001"
    else:
        return f"P = {p_value:.3f}"


# 定义保存路径
save_path = r'/Results/Figures/Figure2/figure2_with_p/'
os.makedirs(save_path, exist_ok=True)


# 定义平滑函数
def smooth_data(data, window_length=5, polyorder=3):
    return savgol_filter(data, window_length, polyorder, mode='wrap')


# 定义颜色映射和线型
style_map = {
    'Non-death': {'color': '#00008B', 'linestyle': '--'},  # 深蓝色
    'All-cause death': {'color': '#8B0000', 'linestyle': '-'},  # 深红色
    'Epilepsy death': {'color': '#006400', 'linestyle': '-'},  # 深绿色
    'Cardiovascular death': {'color': '#4B0082', 'linestyle': '-'},  # 靛蓝色
    'Cerebrovascular death': {'color': '#FF4500', 'linestyle': '-'},  # 橙红色
    'Cancer death': {'color': '#8B4513', 'linestyle': '-'}  # 棕色
}


def plot_activity(data, title, ylabel, filename, groups_to_plot, total_var):
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(20, 12), dpi=300)

    # 如果是两组比较，进行统计分析
    if len(groups_to_plot) == 2:
        p_value = calculate_daily_total_p(groups_to_plot[0][1],
                                          groups_to_plot[1][1],
                                          total_var)

    # 绘制每组数据
    for label, condition in groups_to_plot:
        group_data = data[condition]

        if len(group_data) == 0:
            print(f"警告: {label} 组没有数据")
            continue

        mean = group_data.mean()
        sem = group_data.sem()
        ci = sem * stats.t.ppf((1 + 0.95) / 2, group_data.shape[0] - 1)

        smooth_mean = smooth_data(mean)
        smooth_ci_lower = smooth_data(mean - ci)
        smooth_ci_upper = smooth_data(mean + ci)

        ax.plot(range(24), smooth_mean, color=style_map[label]['color'],
                linestyle=style_map[label]['linestyle'],
                label=f"{label} (n={len(group_data)})",
                linewidth=4)
        ax.fill_between(range(24), smooth_ci_lower, smooth_ci_upper,
                        color=style_map[label]['color'], alpha=0.1)

    # 调整图例位置
    ax.legend(fontsize=32, loc='upper left', bbox_to_anchor=(0, 1.02))

    # 如果是两组比较，添加统计结果
    if len(groups_to_plot) == 2:
        stat_text = format_p_value(p_value)
        ax.text(0.98, 0.98, stat_text,
                transform=ax.transAxes,
                fontsize=32,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5))

    ax.set_title(title, fontsize=48, fontweight='bold', pad=50)
    ax.set_xlabel('Hour of Day', fontsize=42)
    ax.set_ylabel(ylabel, fontsize=42)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels(range(0, 25, 2), fontsize=36)
    ax.set_yticklabels(ax.get_yticks(), fontsize=36)
    ax.set_xlim(0, 23)
    ax.grid(True, linestyle='--', alpha=0.7)

    # 调整图表布局
    plt.tight_layout()

    # 保存图表
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"图像已保存至: {full_path}")


# 定义所有组
all_groups = [
    ('Non-death', merged_df['is_dead'] == 0),
    ('All-cause death', merged_df['is_dead'] == 1),
    ('Epilepsy death', merged_df['epilepsy_death'] == 1),
    ('Cardiovascular death', merged_df['cardiovascular_death'] == 1),
    ('Cerebrovascular death', merged_df['cerebrovascular_death'] == 1),
    ('Cancer death', merged_df['cancer_death'] == 1)
]

# 定义活动类型和对应的总分钟数变量
activity_vars = [
    (light_data, 'Light PA', 'Minutes of Light PA per Hour', 'Light_Total_Min'),
    (mvpa_data, 'MVPA', 'Minutes of MVPA per Hour', 'MVPA_Total_Min'),
    (sedentary_data, 'Sedentary', 'Minutes of Sedentary Behavior per Hour', 'Sedentary_Total_Min')
]

# 绘制并保存包含所有组的图表
for data, title_prefix, ylabel, total_var in activity_vars:
    plot_activity(data, f'{title_prefix} Pattern by Death Cause', ylabel,
                  f'{title_prefix.lower()}_pattern_all_groups.png', all_groups, total_var)

# 绘制并保存每种死亡原因与非死亡组的对比图
for death_group in all_groups[1:]:  # 跳过 'Non-death'
    for data, title_prefix, ylabel, total_var in activity_vars:
        groups_to_plot = [all_groups[0], death_group]  # 'Non-death' 和特定死亡原因
        plot_activity(data, f'{title_prefix} : Non-death vs {death_group[0]}', ylabel,
                      f'{title_prefix.lower()}_pattern_{death_group[0].lower().replace(" ", "_")}.png',
                      groups_to_plot, total_var)

print("所有图表生成完成！")