import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
from datetime import datetime

# 設置中文字體（如果需要）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 設置 seaborn 風格
sns.set_style("whitegrid")
sns.set_palette("husl")


def shorten_model_name(full_path: str, max_length: int = 30) -> str:
    """
    縮短模型路徑名稱，只保留關鍵信息
    
    Args:
        full_path: 完整路徑
        max_length: 最大長度
    
    Returns:
        縮短後的名稱
    """
    # 提取最後的資料夾名稱
    # folder_name = Path(full_path).name
    
    # # 提取關鍵信息：時間戳、環境名、模型類型
    # parts = folder_name.split('_')
    
    # # 嘗試找到模型類型 (PPO, PBPPO 等)
    # model_type = None
    # for part in parts:
    #     if 'PPO' in part.upper() or 'SAC' in part.upper():
    #         model_type = part
    #         break
    
    # if model_type:
    #     # 如果有時間戳，也加上
    #     if len(parts) >= 5:
    #         timestamp = f"{parts[0]}_{parts[1]}_{parts[2]}"
    #         short_name = f"{timestamp}_{model_type}"
    #     else:
    #         short_name = model_type
    # else:
    #     # 如果找不到模型類型，使用最後幾個部分
    #     short_name = '_'.join(parts[-3:])
    
    # # 確保不超過最大長度
    # if len(short_name) > max_length:
    #     short_name = short_name[:max_length-3] + "..."
    
    # return short_name
    return full_path


def create_heatmap(mean_csv: str, std_csv: str = None, index_permutation: list = None, output_path: str = None):
    """
    創建熱力圖，顏色表示性能
    
    Args:
        mean_csv: mean rewards 的 CSV 路徑
        std_csv: std rewards 的 CSV 路徑（可選）
        output_path: 輸出圖片路徑
    """
    # 讀取數據
    df_mean = permutate_csv_df(mean_csv, index_permutation)
    
    # 縮短模型名稱
    df_mean.index = [shorten_model_name(name) for name in df_mean.index]
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(12, max(6, len(df_mean) * 0.5)))
    
    # 繪製熱力圖
    sns.heatmap(
        df_mean,
        annot=True,  # 顯示數值
        fmt='.1f',   # 格式化為小數點後1位
        cmap='RdYlGn',  # 紅-黃-綠配色，綠色表示好
        center=df_mean.values.mean(),  # 中心值
        cbar_kws={'label': 'Mean Reward'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    
    plt.title('Model Performance Heatmap\n(Mean Rewards across Different Initial States)', 
              fontsize=14, pad=20)
    plt.xlabel('Environment Index', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmap saved to: {output_path}")
    
    plt.show()


def create_grouped_bar_chart(mean_csv: str, std_csv: str = None, index_permutation: list = None, output_path: str = None):
    """
    創建分組條形圖，方便比較不同模型
    
    Args:
        mean_csv: mean rewards 的 CSV 路徑
        std_csv: std rewards 的 CSV 路徑（可選，用於誤差棒）
        output_path: 輸出圖片路徑
    """
    # 讀取數據
    df_mean = permutate_csv_df(mean_csv, index_permutation)
    df_std = permutate_csv_df(std_csv, index_permutation) if std_csv else None
    
    # 縮短模型名稱
    df_mean.index = [shorten_model_name(name) for name in df_mean.index]
    if df_std is not None:
        df_std.index = [shorten_model_name(name) for name in df_std.index]
    
    # 準備數據
    n_models = len(df_mean)
    n_indices = len(df_mean.columns)
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 設置條形寬度和位置
    bar_width = 0.8 / n_models
    x = np.arange(n_indices)
    
    # 繪製每個模型的條形
    for i, (model_name, row) in enumerate(df_mean.iterrows()):
        offset = (i - n_models/2) * bar_width + bar_width/2
        yerr = df_std.loc[model_name] if df_std is not None else None
        
        ax.bar(
            x + offset,
            row.values,
            bar_width,
            label=model_name,
            yerr=yerr,
            capsize=3,
            alpha=0.8
        )
    
    ax.set_xlabel('Environment Index', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Model Performance Comparison across Different Initial States', 
                 fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df_mean.columns)
    ax.legend(loc='best', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Grouped bar chart saved to: {output_path}")
    
    plt.show()


def create_radar_chart(mean_csv: str, index_permutation: list = None, output_path: str = None):
    """
    創建雷達圖，展示模型在各索引的均衡性
    
    Args:
        mean_csv: mean rewards 的 CSV 路徑
        output_path: 輸出圖片路徑
    """
    # 讀取數據
    df_mean = permutate_csv_df(mean_csv, index_permutation)
    
    # 縮短模型名稱
    df_mean.index = [shorten_model_name(name) for name in df_mean.index]
    
    # 準備數據
    categories = df_mean.columns.tolist()
    n_cats = len(categories)
    
    # 計算角度
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # 閉合圖形
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 繪製每個模型
    for model_name, row in df_mean.iterrows():
        values = row.tolist()
        values += values[:1]  # 閉合圖形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.15)
    
    # 設置標籤
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, df_mean.values.max() * 1.1)
    ax.set_title('Model Performance Balance\n(Radar Chart)', 
                 fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Radar chart saved to: {output_path}")
    
    plt.show()


def create_box_plot(mean_csv: str, std_csv: str, index_permutation: list = None, output_path: str = None):
    """
    創建箱型圖，展示每個模型的性能分布和穩定性
    
    Args:
        mean_csv: mean rewards 的 CSV 路徑
        std_csv: std rewards 的 CSV 路徑
        output_path: 輸出圖片路徑
    """
    # 讀取數據
    df_mean = permutate_csv_df(mean_csv, index_permutation)
    df_std = permutate_csv_df(std_csv, index_permutation)
    
    # 縮短模型名稱
    df_mean.index = [shorten_model_name(name) for name in df_mean.index]
    df_std.index = [shorten_model_name(name) for name in df_std.index]
    
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左圖：平均獎勵的分布
    df_mean_T = df_mean.T
    bp1 = ax1.boxplot(
        [df_mean_T[col].values for col in df_mean_T.columns],
        labels=df_mean_T.columns,
        patch_artist=True,
        showmeans=True
    )
    
    # 設置顏色
    colors = plt.cm.Set3(np.linspace(0, 1, len(df_mean_T.columns)))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Mean Reward', fontsize=12)
    ax1.set_title('Distribution of Mean Rewards\n(across all indices)', fontsize=13)
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 右圖：標準差的分布（穩定性）
    df_std_T = df_std.T
    bp2 = ax2.boxplot(
        [df_std_T[col].values for col in df_std_T.columns],
        labels=df_std_T.columns,
        patch_artist=True,
        showmeans=True
    )
    
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Standard Deviation', fontsize=12)
    ax2.set_title('Distribution of Std (Stability)\n(across all indices)', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Box plot saved to: {output_path}")
    
    plt.show()


def create_performance_summary(mean_csv: str, std_csv: str, index_permutation: list = None, output_path: str = None):
    """
    創建性能摘要圖：平均性能 vs 穩定性
    
    Args:
        mean_csv: mean rewards 的 CSV 路徑
        std_csv: std rewards 的 CSV 路徑
        output_path: 輸出圖片路徑
    """
    # 讀取數據
    df_mean = permutate_csv_df(mean_csv, index_permutation)
    df_std = permutate_csv_df(std_csv, index_permutation)
    
    # 縮短模型名稱
    short_names = [shorten_model_name(name) for name in df_mean.index]
    
    # 計算總體平均和平均標準差
    avg_mean = df_mean.mean(axis=1)
    avg_std = df_std.mean(axis=1)
    
    # 創建圖表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 繪製散點圖
    scatter = ax.scatter(
        avg_mean,
        avg_std,
        s=200,
        alpha=0.6,
        c=range(len(avg_mean)),
        cmap='viridis',
        edgecolors='black',
        linewidth=1.5
    )
    
    # 添加模型標籤
    for i, name in enumerate(short_names):
        ax.annotate(
            name,
            (avg_mean.iloc[i], avg_std.iloc[i]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
    
    ax.set_xlabel('Average Mean Reward (across all indices)', fontsize=12)
    ax.set_ylabel('Average Std (Stability)', fontsize=12)
    ax.set_title('Model Performance Summary\n(Higher mean & Lower std is better)', 
                 fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    
    # 添加參考線
    ax.axhline(avg_std.mean(), color='red', linestyle='--', alpha=0.3, label='Mean Std')
    ax.axvline(avg_mean.mean(), color='blue', linestyle='--', alpha=0.3, label='Mean Reward')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Performance summary saved to: {output_path}")
    
    plt.show()

def permutate_csv_df(csv_path: str, index_permutation: list = None):
    # 1. 讀取數據 (假設第一欄是 index)
    df = pd.read_csv(csv_path, index_col=0)
    
    # 2. 縮短模型名稱 (假設 shorten_model_name 已經定義在外部)
    # 這裡加個簡單的防呆，確認 index 是字串才處理
    if not df.empty:
        df.index = [shorten_model_name(name) for name in df.index]
    
    # 3. 處理排列
    if index_permutation is not None:
        # --- 檢查條件 1: 長度是否吻合 ---
        # df.columns 包含所有的 Index0, Index1...
        num_columns = len(df.columns)
        if len(index_permutation) != num_columns:
            raise ValueError(
                f"錯誤: Permutation list 長度 ({len(index_permutation)}) "
                f"與資料欄位數量 ({num_columns}) 不符。"
            )
            
        # --- 執行動作 2: 排列數值 ---
        # 使用 iloc[:, list] 語法
        # : 代表選取所有列 (Rows)
        # index_permutation 代表依照此順序選取欄 (Columns)
        df = df.iloc[:, index_permutation]

        # (選用) 如果你希望排列後，欄位名稱也要改回預設的 0, 1, 2... 
        # 可以取消下面這行的註解。否則欄位名稱會跟著數據移動 (例如欄位順序變成 Index2, Index0, Index1)
        # df.columns = range(len(df.columns))

    return df


def visualize_all(mean_csv: str, std_csv: str, output_dir: str = "visualization_results", index_permutation: list = None):
    """
    生成所有視覺化圖表
    
    Args:
        mean_csv: mean rewards 的 CSV 路徑
        std_csv: std rewards 的 CSV 路徑
        output_dir: 輸出目錄
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("生成視覺化圖表...")
    print("="*70 + "\n")
    
    # 1. 熱力圖
    print("1. 生成熱力圖...")
    create_heatmap(
        mean_csv, 
        std_csv, 
        index_permutation,
        os.path.join(output_dir, "heatmap.png")
    )
    
    # 2. 分組條形圖
    print("\n2. 生成分組條形圖...")
    create_grouped_bar_chart(
        mean_csv, 
        std_csv, 
        index_permutation,
        os.path.join(output_dir, "grouped_bar_chart.png")
    )
    
    # 3. 雷達圖
    print("\n3. 生成雷達圖...")
    create_radar_chart(
        mean_csv, 
        index_permutation,
        os.path.join(output_dir, "radar_chart.png")
    )
    
    # 4. 箱型圖
    print("\n4. 生成箱型圖...")
    create_box_plot(
        mean_csv, 
        std_csv, 
        index_permutation,
        os.path.join(output_dir, "box_plot.png")
    )
    
    # 5. 性能摘要圖
    print("\n5. 生成性能摘要圖...")
    create_performance_summary(
        mean_csv, 
        std_csv, 
        index_permutation,
        os.path.join(output_dir, "performance_summary.png")
    )
    
    print("\n" + "="*70)
    print(f"✓ 所有圖表已保存至: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    # 使用範例
    mean_csv = "evaluation_results\\multiagent\\2025_12_24_04_06_CartPoleSwingUpV1WithAdjustablePole-v0_c3\\algorithm_comparison_mean.csv"
    std_csv = "evaluation_results\\multiagent\\2025_12_24_04_06_CartPoleSwingUpV1WithAdjustablePole-v0_c3\\algorithm_comparison_std.csv"
    index_permutation = [6, 2, 4, 0, 3, 1, 5]
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 生成所有視覺化圖表
    visualize_all(mean_csv, std_csv, output_dir="visualization_results/" + time_str, index_permutation=index_permutation)
    
    # 或者單獨生成某個圖表
    # create_heatmap(mean_csv, std_csv)
    # create_grouped_bar_chart(mean_csv, std_csv)
    # create_radar_chart(mean_csv)
    # create_box_plot(mean_csv, std_csv)
    # create_performance_summary(mean_csv, std_csv)