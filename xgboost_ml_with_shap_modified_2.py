import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
import os
import warnings

warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    从Excel或CSV文件中读取数据
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        print(f"读取CSV文件: {file_path}")
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
        print(f"读取Excel文件: {file_path}")
    else:
        raise ValueError("不支持的文件格式！请使用 .csv, .xlsx 或 .xls 格式")

    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    feature_names = df.columns[1:].tolist()
    target_name = df.columns[0]

    print(f"数据加载成功！")
    print(f"目标变量: {target_name}")
    print(f"特征数量: {X.shape[1]}")
    print(f"样本数量: {X.shape[0]}")
    print(f"特征名称: {feature_names}")

    return X, y, feature_names, target_name


def train_xgboost(X, y, feature_names):
    """
    训练XGBoost模型，使用网格搜索进行超参数优化
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    print(f"\n训练集样本数: {X_train.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")

    # 定义超参数网格
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 150],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
    }

    print("\n超参数搜索空间:")
    for key, value in param_grid.items():
        print(f"  {key}: {value}")
    print(f"\n总共需要训练的模型数量: {np.prod([len(v) for v in param_grid.values()])} 个")

    xgb_model = xgb.XGBRegressor(
        random_state=42,
        objective='reg:squarederror'
    )

    print("\n开始网格搜索（10折交叉验证）...")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=10,
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("\n" + "=" * 60)
    print("网格搜索完成！")
    print("=" * 60)
    print(f"\n最佳超参数组合:")
    for key, value in grid_search.best_params_.items():
        print(f"  {key}: {value}")
    print(f"\n最佳交叉验证R²分数: {grid_search.best_score_:.4f}")

    return best_model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    评估模型性能
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)

    print("\n训练集性能:")
    print(f"  R² (决定系数):        {train_r2:.4f}")
    print(f"  RMSE (均方根误差):    {train_rmse:.4f}")
    print(f"  MAE (平均绝对误差):   {train_mae:.4f}")

    print("\n测试集性能:")
    print(f"  R² (决定系数):        {test_r2:.4f}")
    print(f"  RMSE (均方根误差):    {test_rmse:.4f}")
    print(f"  MAE (平均绝对误差):   {test_mae:.4f}")

    overfit_r2 = train_r2 - test_r2
    print(f"\n过拟合分析:")
    print(f"  R²差值 (训练-测试):   {overfit_r2:.4f}")
    if overfit_r2 < 0.05:
        print("  模型拟合良好，未出现明显过拟合")
    elif overfit_r2 < 0.15:
        print("  模型存在轻微过拟合")
    else:
        print("  模型存在较严重过拟合，建议调整超参数")

    print("\n" + "=" * 60)

    return {
        'train': {'R2': train_r2, 'RMSE': train_rmse, 'MAE': train_mae},
        'test': {'R2': test_r2, 'RMSE': test_rmse, 'MAE': test_mae}
    }


def save_results(metrics, file_path='model_results.xlsx'):
    """
    将评估结果保存到Excel文件
    """
    results_df = pd.DataFrame({
        '数据集': ['训练集', '测试集'],
        'R²': [metrics['train']['R2'], metrics['test']['R2']],
        'RMSE': [metrics['train']['RMSE'], metrics['test']['RMSE']],
        'MAE': [metrics['train']['MAE'], metrics['test']['MAE']]
    })

    results_df.to_excel(file_path, index=False)
    print(f"\n评估结果已保存到: {file_path}")


# ==================== SHAP 绘图相关函数 ====================

def find_knee_point(x_data, y_data, window_length=5, polyorder=2):
    """
    通过基于曲率的方法寻找曲线上趋势变化最显著的点（即"拐点"或"膝点"）。
    """
    if len(x_data) < window_length:
        return np.median(x_data)

    if window_length % 2 == 0:
        window_length += 1

    if polyorder >= window_length:
        polyorder = window_length - 1
        if polyorder < 1:
            polyorder = 1

    y_second_deriv = savgol_filter(y_data, window_length, polyorder, deriv=2)
    knee_index = np.argmax(np.abs(y_second_deriv))
    sorted_x = np.array(x_data)[np.argsort(x_data)]
    return sorted_x[knee_index]


def save_mean_shap_values(shap_importance, feature_names, output_dir="SHAP_OUTPUT", top_n=None):
    """
    保存每个特征的平均绝对SHAP值（直接使用条形图计算的值）
    
    参数:
        shap_importance: 已计算好的平均绝对SHAP值数组
        feature_names: 特征名称列表
        output_dir: 输出目录
        top_n: 使用的top样本数量（用于打印信息）
    """
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建DataFrame（直接使用传入的重要性值）
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Absolute_SHAP_Value': shap_importance
    })
    
    # 按平均绝对SHAP值排序（从大到小）
    shap_df_sorted = shap_df.sort_values('Mean_Absolute_SHAP_Value', ascending=False)
    
    # 保存到Excel
    output_path = os.path.join(output_dir, "mean_shap_values_top_samples.xlsx")
    shap_df_sorted.to_excel(output_path, index=False)
    
    # 打印结果
    print("\n" + "=" * 60)
    if top_n:
        print(f"基于Top {top_n}样本计算的各特征平均绝对SHAP值")
    else:
        print("各特征平均绝对SHAP值")
    print("=" * 60)
    
    for idx, row in shap_df_sorted.iterrows():
        print(f"{row['Feature']:30s}: {row['Mean_Absolute_SHAP_Value']:10.6f}")
    
    print(f"\n✅ 平均绝对SHAP值已保存到: {output_path}")
    print("=" * 60 + "\n")
    
    return shap_df_sorted


def plot_shap_combined_analysis(model, X_test, feature_names, output_dir="SHAP_OUTPUT", top_n=2000):
    """
    绘制SHAP组合分析图（选取SHAP值最大的top_n个样本）
    
    参数:
        model: 训练好的XGBoost模型
        X_test: 测试集特征数据
        feature_names: 特征名称列表
        output_dir: 输出目录
        top_n: 选取SHAP绝对值最大的样本数量
    """
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建文件夹: {output_dir}")

    # ==================== 美学参数配置区 ====================
    aesthetic_params = {
        'suptitle_size': 22,
        'ax_label_size': 16,
        'tick_label_size': 16,
        'legend_size': 14,
        'cbar_label_size': 15,
        'summary_cbar_width': 0.015,
        'summary_cbar_height_shrink': 1.0,
        'summary_cbar_pad': 0.01,
        'dep_cbar_width': 0.005,
        'dep_cbar_height_shrink': 1.0,
        'dep_cbar_pad': 0.002,
        'dep_cbar_tick_length': 1,
        'grid_wspace': 0.45,
        'grid_hspace': 0.4
    }

    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'

    # ==================== 计算全部测试集的SHAP值 ====================
    print("=" * 60)
    print("开始计算全部测试集的SHAP值...")
    print("=" * 60)
    
    # 创建 SHAP 解释器
    print("初始化SHAP解释器...")
    explainer = shap.TreeExplainer(model)
    print("正在计算全部测试集的 SHAP 值...")
    
    # 分批计算所有测试集样本的SHAP值
    batch_size = 500
    shap_values_list = []
    
    print(f"测试集总样本数: {len(X_test)}")
    print(f"批次大小: {batch_size}")
    
    for i in range(0, len(X_test), batch_size):
        batch_end = min(i + batch_size, len(X_test))
        batch_data = X_test[i:batch_end]
        batch_shap = explainer(batch_data)
        shap_values_list.append(batch_shap.values)
        print(f"已处理 {batch_end}/{len(X_test)} 个样本")

    shap_values_array_all = np.vstack(shap_values_list)
    print("✅ 全部测试集SHAP值计算完成！")
    
    # ==================== 选取SHAP绝对值最大的top_n个样本 ====================
    print("\n" + "=" * 60)
    print(f"选取SHAP绝对值最大的 {top_n} 个样本用于绘图和保存")
    print("=" * 60)
    
    # 计算每个样本的SHAP绝对值总和
    shap_abs_sum = np.abs(shap_values_array_all).sum(axis=1)
    
    # 选取top_n个样本的索引
    if len(X_test) <= top_n:
        print(f"⚠️ 测试集样本数({len(X_test)})小于等于top_n({top_n})，使用全部样本")
        top_indices = np.arange(len(X_test))
    else:
        top_indices = np.argsort(shap_abs_sum)[-top_n:][::-1]  # 降序排列
        print(f"✅ 已选取SHAP绝对值最大的 {top_n} 个样本")
    
    # 筛选出top_n个样本的数据
    X_test_sampled = X_test[top_indices]
    shap_values_array = shap_values_array_all[top_indices]
    
    print(f"选取样本的SHAP绝对值总和范围: [{shap_abs_sum[top_indices].min():.4f}, {shap_abs_sum[top_indices].max():.4f}]")
    print(f"用于绘图的样本数量: {len(X_test_sampled)}")


    # 创建SHAP Explanation对象
    shap_values = shap.Explanation(
        values=shap_values_array,
        base_values=np.full(len(X_test_sampled), explainer.expected_value),
        data=X_test_sampled,
        feature_names=feature_names
    )

    # 计算特征重要性
    print("计算特征重要性...")
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    all_indices = np.argsort(shap_importance)[::-1]
    all_features = [feature_names[i] for i in all_indices]
    all_importance = shap_importance[all_indices]

    # ==================== 保存平均SHAP值（使用条形图的计算结果）====================
    save_mean_shap_values(shap_importance, feature_names, output_dir, top_n=len(X_test_sampled))
    # ============================================================

    # ==================== SHAP组合分析图 ====================
    print("绘制SHAP组合分析图...")

    # 确定要绘制的特征（前6个最重要的特征）
    features_to_plot = all_features[:6]
    print(f"将绘制前 6 个最重要特征的依赖图: {features_to_plot}")

    # 转换为DataFrame
    X_test_df = pd.DataFrame(X_test_sampled, columns=feature_names)

    # 计算每个特征的SHAP绝对值的平均值并排序
    mean_abs_shaps = np.abs(shap_values.values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shaps
    }).sort_values('importance', ascending=True)

    # 创建画布和网格布局
    fig = plt.figure(figsize=(30, 18))
    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        wspace=aesthetic_params['grid_wspace'],
        hspace=aesthetic_params['grid_hspace']
    )

    # --- 摘要图 (左侧) ---
    ax_main = fig.add_subplot(gs[:, :2])

    ax_main.set_yticks(range(len(feature_importance_df)))
    ax_main.set_yticklabels(feature_importance_df['feature'], fontsize=aesthetic_params['tick_label_size'])

    ax_top = ax_main.twiny()
    ax_top.set_zorder(ax_main.get_zorder() - 1)
    ax_top.patch.set_visible(False)
    ax_main.patch.set_visible(False)

    ax_top.barh(
        range(len(feature_importance_df)),
        feature_importance_df['importance'],
        color="lightgray", alpha=0.6, height=0.7
    )

    ax_top.set_xlabel("Mean Absolute SHAP Value (Global Importance)", fontsize=aesthetic_params['ax_label_size'])
    ax_top.tick_params(axis='x', labelsize=aesthetic_params['tick_label_size'])
    ax_top.grid(False)

    cmap = plt.get_cmap("viridis_r")

    for i, feature_name in enumerate(feature_importance_df['feature']):
        original_idx = feature_names.index(feature_name)
        shap_vals_for_feature = shap_values.values[:, original_idx]
        feature_vals_for_color = X_test_df.iloc[:, original_idx]
        y_jitter = np.random.normal(0, 0.08, shap_vals_for_feature.shape[0])
        ax_main.scatter(
            shap_vals_for_feature, i + y_jitter, c=feature_vals_for_color,
            cmap=cmap, s=15, alpha=0.8, zorder=2
        )

    ax_main.set_xlabel("SHAP value (impact on model output)", fontsize=aesthetic_params['ax_label_size'])
    ax_main.tick_params(axis='x', labelsize=aesthetic_params['tick_label_size'])
    ax_main.grid(True, axis='x', linestyle='--', alpha=0.6)

    # --- 摘要图颜色条 ---
    fig.canvas.draw()
    ax_main_pos = ax_main.get_position()
    cax_left = ax_main_pos.x1 + aesthetic_params['summary_cbar_pad']
    cax_bottom = ax_main_pos.y0 + (ax_main_pos.height * (1 - aesthetic_params['summary_cbar_height_shrink']) / 2)
    cax_width = aesthetic_params['summary_cbar_width']
    cax_height = ax_main_pos.height * aesthetic_params['summary_cbar_height_shrink']

    cax = fig.add_axes([cax_left, cax_bottom, cax_width, cax_height])
    norm = plt.Normalize(vmin=X_test_df.values.min(), vmax=X_test_df.values.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Feature value', rotation=90, labelpad=-15, fontsize=aesthetic_params['cbar_label_size'])
    cbar.outline.set_visible(False)
    cbar.set_ticks([])

    cbar.ax.text(0.6, 1.02, 'High', ha='center', va='top', transform=cbar.ax.transAxes,
                 fontsize=aesthetic_params['tick_label_size'])
    cbar.ax.text(0.6, -0.02, 'Low', ha='center', va='bottom', transform=cbar.ax.transAxes,
                 fontsize=aesthetic_params['tick_label_size'])

    # --- 依赖图 (右侧) ---
    axes_scatter = []
    for i in range(3):
        for j in range(2):
            axes_scatter.append(fig.add_subplot(gs[i, j + 2]))

    for i, feature in enumerate(features_to_plot[:6]):
        try:
            if feature not in feature_names:
                print(f"⚠️ 警告：特征 '{feature}' 不在 feature_names 中，跳过！")
                continue

            feature_index = feature_names.index(feature)

            if feature_index >= X_test_sampled.shape[1]:
                print(f"⚠️ 警告：特征索引 {feature_index} 超出范围，跳过特征 '{feature}'！")
                continue

            feature_values = X_test_sampled[:, feature_index]
            feature_shap_values = shap_values.values[:, feature_index]

            valid_mask = feature_values >= -1
            feature_values_filtered = feature_values[valid_mask]
            feature_shap_values_filtered = feature_shap_values[valid_mask]

            if len(feature_values_filtered) == 0 or len(feature_shap_values_filtered) == 0:
                print(f"⚠️ 警告：特征 '{feature}' 的所有特征值都小于-1，跳过绘制该特征的依赖图！")
                continue

            ax = axes_scatter[i]

            y_pred_filtered = model.predict(X_test_sampled[valid_mask])

            scatter = ax.scatter(feature_values_filtered, feature_shap_values_filtered,
                                 c=y_pred_filtered, cmap=cmap, s=25, alpha=0.8)

            # --- 依赖图颜色条 ---
            fig.canvas.draw()
            ax_pos = ax.get_position()
            cax_dep_left = ax_pos.x1 + aesthetic_params['dep_cbar_pad']
            cax_dep_bottom = ax_pos.y0 + (ax_pos.height * (1 - aesthetic_params['dep_cbar_height_shrink']) / 2)
            cax_dep_width = aesthetic_params['dep_cbar_width']
            cax_dep_height = ax_pos.height * aesthetic_params['dep_cbar_height_shrink']

            cax_dep = fig.add_axes([cax_dep_left, cax_dep_bottom, cax_dep_width, cax_dep_height])
            cbar_dep = fig.colorbar(scatter, cax=cax_dep)
            cbar_dep.ax.set_title('Predicted', fontsize=10)
            cbar_dep.outline.set_visible(False)
            cbar_dep.ax.tick_params(
                axis='y',
                length=aesthetic_params['dep_cbar_tick_length'],
                labelsize=aesthetic_params['tick_label_size']
            )

            ax.set_xlabel(feature, fontsize=aesthetic_params['ax_label_size'])
            ax.set_ylabel(f"SHAP", fontsize=12, labelpad=6)

            median_val = np.median(feature_values_filtered)
            threshold_val = find_knee_point(feature_values_filtered, feature_shap_values_filtered)

            ax.axvline(median_val, color='black', linestyle='--', linewidth=1)
            ax.axvline(threshold_val, color='red', linestyle=':', linewidth=1.2)

            line_handles = [
                Line2D([0], [0], color='black', lw=1, linestyle='--',
                       label=f'Median: {median_val:.2f}'),
                Line2D([0], [0], color='red', lw=1, linestyle=':',
                       label=f'Threshold: {threshold_val:.2f}')
            ]
            ax.legend(handles=line_handles, loc='best', fontsize=aesthetic_params['legend_size'])
            ax.tick_params(axis='both', which='major', labelsize=aesthetic_params['tick_label_size'])

            print(f"✅ 成功绘制特征 '{feature}' 的依赖图")

        except Exception as e:
            print(f"❌ 绘制特征 '{feature}' 依赖图时出错: {str(e)}")
            continue

    # --- 最终布局与保存 ---
    output_image_path = os.path.join(output_dir, "shap_combined_analysis.png")
    plt.savefig(output_image_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP 组合分析图已成功保存到文件: {output_image_path}")

    print("✅ SHAP组合分析图绘制完成！")


def main():
    """
    主函数
    """
    # 设置数据文件路径
    data_file = 'ERI.csv'  # 可以是 'data.csv' 或 'data.xlsx'

    try:
        # 1. 加载数据
        X, y, feature_names, target_name = load_data(data_file)

        # 2. 训练模型
        best_model, X_train, X_test, y_train, y_test = train_xgboost(
            X, y, feature_names
        )

        # 3. 评估模型
        metrics = evaluate_model(best_model, X_train, X_test, y_train, y_test)

        # 4. 保存结果
        save_results(metrics)

        # 5. 绘制SHAP组合分析图（直接使用训练好的模型）
        print("\n" + "=" * 60)
        print("开始SHAP可解释性分析与绘图")
        print("=" * 60)
        plot_shap_combined_analysis(
            model=best_model,
            X_test=X_test,
            feature_names=feature_names,
            output_dir="SHAP_OUTPUT",
            top_n=2000  # 选取SHAP绝对值最大的2000个样本
        )

        print("\n✓ 所有步骤执行完成！")

    except FileNotFoundError:
        print(f"\n错误: 找不到文件 '{data_file}'")
        print("请确保文件路径正确，或修改代码中的 data_file 变量")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
