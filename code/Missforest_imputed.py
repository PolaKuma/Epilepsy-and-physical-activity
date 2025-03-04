import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def missforest_imputation(data, max_iter=10, threshold=0.1):
    """
    使用MISSFOREST算法进行数据插补

    参数:
    data: pandas DataFrame
    max_iter: 最大迭代次数
    threshold: 收敛阈值
    """

    # 复制原始数据
    df = data.copy()

    # 分离连续变量和分类变量
    continuous_vars = ['BMI', 'age', 'Townsend_index']
    categorical_vars = [col for col in df.columns if col not in continuous_vars + ['id']]

    # 为分类变量创建标签编码器字典
    label_encoders = {}
    original_categories = {}
    for cat_var in categorical_vars:
        label_encoders[cat_var] = LabelEncoder()
        # 记录原始类别
        original_categories[cat_var] = df[cat_var].dropna().unique()
        # 对非空值进行编码
        non_null_mask = df[cat_var].notna()
        df.loc[non_null_mask, cat_var] = label_encoders[cat_var].fit_transform(
            df.loc[non_null_mask, cat_var]
        )

    # 初始化
    previous_matrix = df.fillna(df.mean()).values
    current_matrix = df.values.copy()
    n_iter = 0

    while n_iter < max_iter:
        # 存储上一次迭代的结果
        previous_matrix = current_matrix.copy()

        # 对每个含有缺失值的列进行迭代
        for col_idx in range(1, df.shape[1]):  # 跳过id列
            col_name = df.columns[col_idx]

            # 获取当前列的缺失值索引
            missing_idx = np.where(pd.isnull(df.iloc[:, col_idx]))[0]
            if len(missing_idx) == 0:
                continue

            # 准备训练数据
            known_idx = np.where(~pd.isnull(df.iloc[:, col_idx]))[0]
            X_train = current_matrix[known_idx, 1:]  # 跳过id列
            y_train = current_matrix[known_idx, col_idx]
            X_missing = current_matrix[missing_idx, 1:]  # 跳过id列

            # 选择适当的模型并训练
            if col_name in continuous_vars:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                imputed_values = model.predict(X_missing)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                imputed_values = model.predict(X_missing)
                # 确保分类变量的预测值在原始类别范围内
                valid_categories = np.unique(y_train)
                imputed_values = np.clip(imputed_values,
                                         valid_categories.min(),
                                         valid_categories.max())

            # 更新缺失值
            current_matrix[missing_idx, col_idx] = imputed_values

        # 检查是否收敛
        diff = np.sum((previous_matrix - current_matrix) ** 2) / np.sum(previous_matrix ** 2)
        if diff < threshold:
            break

        n_iter += 1

    # 将结果转换回DataFrame
    imputed_df = pd.DataFrame(current_matrix, columns=df.columns)

    # 将分类变量转换回原始类别
    for cat_var in categorical_vars:
        imputed_df[cat_var] = label_encoders[cat_var].inverse_transform(
            imputed_df[cat_var].astype(int)
        )
        # 确保值在原始类别中
        imputed_df.loc[~imputed_df[cat_var].isin(original_categories[cat_var]), cat_var] = \
            np.random.choice(original_categories[cat_var])

    return imputed_df


# 使用示例
def main():
    # 读取数据
    data = pd.read_csv('.csv')

    # 运行插补
    imputed_data = missforest_imputation(data)

    # 保存结果
    imputed_data.to_csv('.csv', index=False)


if __name__ == "__main__":
    main()