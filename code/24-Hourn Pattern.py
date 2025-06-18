import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy import stats


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    survived = df[df['is_dead'] == 0].iloc[:, 2:].values
    deceased = df[df['is_dead'] == 1].iloc[:, 2:].values
    return survived, deceased


def calculate_confidence_interval(data, confidence=0.95):
    mean = np.mean(data, axis=0)
    sem = stats.sem(data, axis=0)
    ci = sem * stats.t.ppf((1 + confidence) / 2, data.shape[0] - 1)
    return mean, mean - ci, mean + ci


def smooth_curve(x, y, n_points=300):
    x_smooth = np.linspace(x.min(), x.max(), n_points)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth


def plot_minute_level(survived, deceased):
    plt.rcParams['font.size'] = 14
    fig, ax = plt.subplots(figsize=(15, 8))

    survived_mean, survived_lower, survived_upper = calculate_confidence_interval(survived)
    deceased_mean, deceased_lower, deceased_upper = calculate_confidence_interval(deceased)

    x = np.arange(1440)

    x_smooth, survived_mean_smooth = smooth_curve(x, survived_mean)
    _, survived_lower_smooth = smooth_curve(x, survived_lower)
    _, survived_upper_smooth = smooth_curve(x, survived_upper)

    _, deceased_mean_smooth = smooth_curve(x, deceased_mean)
    _, deceased_lower_smooth = smooth_curve(x, deceased_lower)
    _, deceased_upper_smooth = smooth_curve(x, deceased_upper)

    # 增加线条粗细
    ax.plot(x_smooth, survived_mean_smooth, color='#077E97', label=f'Survived (n={survived.shape[0]})', linewidth=3)
    ax.fill_between(x_smooth, survived_lower_smooth, survived_upper_smooth, color='#077E97', alpha=0.2)

    ax.plot(x_smooth, deceased_mean_smooth, color='#FF7F0E', label=f'Deceased (n={deceased.shape[0]})', linewidth=3)
    ax.fill_between(x_smooth, deceased_lower_smooth, deceased_upper_smooth, color='#FF7F0E', alpha=0.2)

    ax.set_xlabel('Time (minutes)', fontsize=16)
    ax.set_ylabel('Average acceleration (milligal)', fontsize=16)
    ax.set_title('Minute-level Physical Activity Pattern: Survived vs Epilepsy-Cause Mortality', fontsize=18, pad=20)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)

    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig('F:/Research_Project/EP_ukb/Revision_figure2/minute_level_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hour_level(survived, deceased):
    plt.rcParams['font.size'] = 20
    survived_hourly = np.array([survived[:, i:i + 60].mean(axis=1) for i in range(0, 1440, 60)]).T
    deceased_hourly = np.array([deceased[:, i:i + 60].mean(axis=1) for i in range(0, 1440, 60)]).T

    fig, ax = plt.subplots(figsize=(24, 8))

    survived_mean, survived_lower, survived_upper = calculate_confidence_interval(survived_hourly)
    deceased_mean, deceased_lower, deceased_upper = calculate_confidence_interval(deceased_hourly)

    x = np.arange(24)

    x_smooth, survived_mean_smooth = smooth_curve(x, survived_mean)
    _, survived_lower_smooth = smooth_curve(x, survived_lower)
    _, survived_upper_smooth = smooth_curve(x, survived_upper)

    _, deceased_mean_smooth = smooth_curve(x, deceased_mean)
    _, deceased_lower_smooth = smooth_curve(x, deceased_lower)
    _, deceased_upper_smooth = smooth_curve(x, deceased_upper)

    # 增加线条粗细
    ax.plot(x_smooth, survived_mean_smooth, color='#077E97', label=f'Survived (n={survived.shape[0]})', linewidth=4)
    ax.fill_between(x_smooth, survived_lower_smooth, survived_upper_smooth, color='#077E97', alpha=0.2)

    ax.plot(x_smooth, deceased_mean_smooth, color='#FF7F0E', label=f'Deceased (n={deceased.shape[0]})', linewidth=4)
    ax.fill_between(x_smooth, deceased_lower_smooth, deceased_upper_smooth, color='#FF7F0E', alpha=0.2)

    # 设置时间轴标签
    time_labels = [f'{i:02d}:00-{i:02d}:59' for i in range(24)]
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels(time_labels, rotation=45, ha='right')

    ax.set_xlabel('Time (1 h intervals)', fontsize=20)
    ax.set_ylabel('Average acceleration (milligal)', fontsize=20)
    ax.set_title('Hour-level Physical Activity Pattern: Survived vs Epilepsy-Cause Mortality', fontsize=24, pad=20)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.3)

    ax.tick_params(axis='both', which='major', labelsize=22)

    plt.tight_layout()
    plt.savefig('F:/Research_Project/EP_ukb/Revision_figure2/hour_level_pattern.png', dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    file_path = "F:/Research_Project/EP_ukb/Revision_figure2/EP_PA_Intensity_Imputed.csv"
    survived, deceased = load_and_prepare_data(file_path)
    plot_minute_level(survived, deceased)
    plot_hour_level(survived, deceased)