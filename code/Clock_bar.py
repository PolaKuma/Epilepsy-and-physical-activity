import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('nGP_combine_cluster.csv')

# 将24小时数据分割为单独的列，并转换为分钟
activity_columns = {
    'Light PA': 'Light_Dayhouraverage',
    'MVPA': 'Moderate_Vigorous_Day_hour_average',
    'Sedentary': 'Sedentary_Day_hour_average'
}

for activity, column in activity_columns.items():
    df[f'{activity}_minutes'] = df[column].str.split(',').apply(lambda x: [float(i) * 60 for i in x if i])  # 转换为分钟

# 定义多种配色方案
color_schemes = {
    'blue_orange': {'Alive': '#1E90FF', 'Dead': '#FFA500'},  # 蓝色和橘色
}



def plot_circular_side_by_side(data, activity, color_scheme):
    dead = np.mean(data[data['is_dead'] == 1][f'{activity}_minutes'].tolist(), axis=0)
    alive = np.mean(data[data['is_dead'] == 0][f'{activity}_minutes'].tolist(), axis=0)

    fig, ax = plt.subplots(figsize=(20, 18), subplot_kw=dict(projection='polar'))

    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    width = np.pi / 24

    # Side by side bars
    ax.bar(theta - width / 2, dead, width=width, color=color_scheme['Dead'], alpha=0.7, label='Dead')
    ax.bar(theta + width / 2, alive, width=width, color=color_scheme['Alive'], alpha=0.7, label='Alive')

    ax.set_ylim(0, max(np.max(dead), np.max(alive)) * 1.1)
    ax.set_yticks([])
    ax.set_xticks(theta)
    ax.set_xticklabels(range(24), fontsize=60, fontweight='bold')

    # Add radial lines for each hour
    for i in range(24):
        angle = i * 2 * np.pi / 24
        ax.plot([angle, angle], [0, ax.get_ylim()[1]], color='gray', linewidth=0.5, alpha=0.5)

    ax.set_title(f"{activity.replace('_', ' ')} ", fontsize=50, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=40, prop={'weight': 'bold'})

    plt.tight_layout()
    #plt.savefig(
    #    f'E:/EP_ukb/2PAgroup_analysis/{activity}_circular_{list(color_schemes.keys())[list(color_schemes.values()).index(color_scheme)]}.png',
    #    dpi=300, bbox_inches='tight')
    plt.savefig(
        f'E:/EP_ukb/Test/{activity}_circular_{list(color_schemes.keys())[list(color_schemes.values()).index(color_scheme)]}.png',
        dpi=300, bbox_inches='tight',transparent=True )
    plt.close()


def plot_radar(data, activity, color_scheme):
    dead = np.mean(data[data['is_dead'] == 1][f'{activity}_minutes'].tolist(), axis=0)
    alive = np.mean(data[data['is_dead'] == 0][f'{activity}_minutes'].tolist(), axis=0)

    fig, ax = plt.subplots(figsize=(20, 18), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    # Close the plot
    dead = np.concatenate((dead, [dead[0]]))
    alive = np.concatenate((alive, [alive[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    ax.plot(angles, dead, 'o-', linewidth=3, label='Dead', color=color_scheme['Dead'])
    ax.plot(angles, alive, 'o-', linewidth=3, label='Alive', color=color_scheme['Alive'])
    ax.fill(angles, dead, alpha=0.25, color=color_scheme['Dead'])
    ax.fill(angles, alive, alpha=0.25, color=color_scheme['Alive'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(range(24), fontsize=60, fontweight='bold')
    ax.set_title(f"{activity.replace('_', ' ')} ", fontsize=50, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=40, prop={'weight': 'bold'})

    # Set reasonable y-ticks
    max_value = max(np.max(dead), np.max(alive))
    ax.set_yticks(np.linspace(0, max_value, 5))
    ax.set_yticklabels([f'{int(x)}' for x in ax.get_yticks()], fontsize=60, fontweight='bold')

    plt.tight_layout()
    #plt.savefig(
    #    f'E:/EP_ukb/2PAgroup_analysis/{activity}_radar_{list(color_schemes.keys())[list(color_schemes.values()).index(color_scheme)]}.png',
    #    dpi=300, bbox_inches='tight')
    plt.savefig(
        f'E:/EP_ukb/Test/{activity}_radar_{list(color_schemes.keys())[list(color_schemes.values()).index(color_scheme)]}.png',
        dpi=300, bbox_inches='tight', transparent=True  # 添加 transparent=True
    )


    plt.close()


# Generate individual plots for each activity and color scheme
for activity in ['Light PA', 'MVPA', 'Sedentary']:
    for scheme_name, color_scheme in color_schemes.items():
        plot_circular_side_by_side(df, activity, color_scheme)
        plot_radar(df, activity, color_scheme)