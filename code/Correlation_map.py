import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
data = pd.read_csv('nGP_combines.csv')

# 定义我们感兴趣的变量
main_variables = ['Light_Total_Min', 'MVPA_Total_Min', 'Sedentary_Total_Min']
other_variables = ['gender','Townsend_index', 'overall_health', 'BMI', 'sleep_condition',
                   'hpb_combined', 'cvd_combined', 'bd_combined', 'zcd_combined',
                   'diabetes_combined', 'education_level', 'smoking', 'alcohol_use',
                   'ethic']


# 计算斯皮尔曼相关性
correlation_matrix = data[main_variables + other_variables].corr(method='spearman')

# 只保留我们需要的相关性
correlation_matrix = correlation_matrix.loc[main_variables, other_variables]

# 创建热图
plt.figure(figsize=(40, 15))  # 进一步增大图表尺寸
sns.set(font_scale=2.0)  # 增大整体字体大小

# 创建自定义颜色映射
colors = plt.cm.RdBu_r(np.linspace(0, 1, 256))
custom_cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors)

# 创建热图，设置色阶范围为 -0.5 到 0.5
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap=custom_cmap,
                      cbar_kws={'label': 'Spearman Correlation'},
                      linewidths=0.5, center=0, vmin=-0.5, vmax=0.5)

# 设置标题和标签
plt.title('Spearman Correlation of Physical Activity Metrics with Other Variables', fontsize=36, pad=20)
plt.xlabel('Other Variables', fontsize=32, labelpad=20)
plt.ylabel('Physical Activity Metrics', fontsize=32, labelpad=20)

# 调整坐标轴标签的字体大小和旋转
plt.xticks(rotation=90, fontsize=24)
plt.yticks(fontsize=28)

# 调整色条标签的字体大小
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=24)
cbar.set_label('Spearman Correlation', size=28)

# 调整布局，确保所有标签都可见
plt.tight_layout()

# 保存图像
plt.savefig('correlation_heatmap_adjusted_larger_font.png', dpi=300, bbox_inches='tight')
plt.close()

print("调整后的相关性热图（更大字体版）已生成完毕。")