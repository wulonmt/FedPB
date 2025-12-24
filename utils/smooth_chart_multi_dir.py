import argparse
import os
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import matplotlib.pyplot as plt
from tsmoothie import LowessSmoother
import numpy as np

# 新增：消除環境偏差的統計方法
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_names", nargs='+', help="data log names (can be multiple dirs per algorithm)", type=str, required=True)
parser.add_argument("-n", "--custom_names", nargs='+', help="custom names for each algorithm", type=str)
parser.add_argument("-s", "--save_dir", help="directory to save plots", type=str, required=True)
parser.add_argument("-o", "--remove_outliers", action="store_true", help="remove outliers from the data")
parser.add_argument("--iqr_factor", type=float, default=1.5, help="IQR factor for outlier removal (default: 1.5)")
parser.add_argument("-p", "--prefixes", type=str, default="0,1,2,3,4", help="Comma-separated list of prefixes of subdirectories to process (default: 0,1,2,3,4)")
# 新增參數：每個演算法有幾個資料夾
parser.add_argument("-r", "--repeat_counts", nargs='+', type=int, help="Number of repeated experiments for each algorithm (e.g., 3 2 4 means first alg has 3 dirs, second has 2, third has 4)")
args = parser.parse_args()

def process_folder(folder_path, prefix):
    event_loader_list = []
    true_folder_path = [f.path for f in os.scandir(folder_path) if f.is_dir() and f.name.startswith(prefix)]
    assert len(true_folder_path) == 1, f"Prefix {prefix} found {len(true_folder_path)} folders, expected 1"
    
    subdirectories = [f.path for f in os.scandir(true_folder_path[0]) if f.is_dir()]
    
    for subdir in subdirectories:
        files = [f.path for f in os.scandir(subdir) if f.is_file()]
        if len(files) == 1:
            file_path = files[0]
            event_loader_list.append(EventFileLoader(file_path))
        else:
            print(f"Error: Found {len(files)} files in {subdir}. Expected only one.")
    
    # 預定義要追蹤的metrics
    metrics = [
        "rollout/ep_rew_mean",
        "train/value_loss",
        "train/std",
        "train/delta_norm",
        "train/kl_difference",
        "train/lag_loss",
        "train/lambda",
        "train/perturb_loss",
    ]
    metrics_dict = {m: ([], []) for m in metrics}
    
    for event_file in event_loader_list:
        for event in event_file.Load():
            if len(event.summary.value) > 0:
                tag = event.summary.value[0].tag
                # 只收集預定義的metrics
                if tag in metrics_dict:
                    metrics_dict[tag][0].append(event.step)
                    metrics_dict[tag][1].append(event.summary.value[0].tensor.float_val[0])
    
    # 排序所有metrics的數據
    for k, (x, y) in metrics_dict.items():
        combined = list(zip(x, y))
        sorted_combined = sorted(combined, key=lambda item: item[0])
        if sorted_combined:
            metrics_dict[k] = tuple(zip(*sorted_combined))
        else:
            metrics_dict[k] = ([], [])
    
    return metrics_dict

def custom_smoother(y, smoother_y, coef=0.3):
    low, up = [], []
    last_range = 0
    for data, target in zip(y, smoother_y):
        data_range = abs(data - target)
        last_range = last_range + coef * (data_range - last_range)
        up.append(target + last_range)
        low.append(target - last_range)
    return low, up

def remove_outliers(x, y, iqr_factor):
    x = np.array(x)
    y = np.array(y)
    
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR
    
    mask = (y >= lower_bound) & (y <= upper_bound)
    
    return x[mask], y[mask]

def group_logs_by_algorithm(log_names, repeat_counts):
    """將log資料夾按演算法分組"""
    if repeat_counts is None:
        # 如果沒有指定repeat_counts，假設每個log_name是一個演算法
        return [[log] for log in log_names]
    
    if sum(repeat_counts) != len(log_names):
        raise ValueError(f"Sum of repeat_counts ({sum(repeat_counts)}) must equal number of log_names ({len(log_names)})")
    
    grouped_logs = []
    idx = 0
    for count in repeat_counts:
        grouped_logs.append(log_names[idx:idx+count])
        idx += count
    
    return grouped_logs

def aggregate_algorithm_data(log_group, prefix):
    """聚合同一演算法多個重複實驗的數據"""
    all_metrics_list = []
    
    for log_name in log_group:
        try:
            metrics_data = process_folder(log_name, prefix + "_")
            all_metrics_list.append(metrics_data)
        except Exception as e:
            print(f"Warning: {e} at {log_name}, {prefix}")
            continue
    
    if not all_metrics_list:
        return None
    
    # 聚合數據：對每個metric計算平均值和標準差
    aggregated_metrics = {}
    
    # 使用第一個實驗的metrics作為基準（包含所有預定義的metrics）
    metrics = list(all_metrics_list[0].keys())
    
    for metric in metrics:
        # 收集該metric在所有重複實驗中的數據
        all_x_y_pairs = []
        for metrics_data in all_metrics_list:
            x, y = metrics_data.get(metric, ([], []))
            if len(x) > 0 and len(y) > 0:
                if args.remove_outliers:
                    x, y = remove_outliers(x, y, args.iqr_factor)
                all_x_y_pairs.append((x, y))
        
        if not all_x_y_pairs:
            # 即使沒有數據也保留空的結構
            aggregated_metrics[metric] = ([], [], [])
            continue
        
        # 找到最短的數據長度（確保對齊）
        min_length = min(len(y) for x, y in all_x_y_pairs)
        
        # 收集對齊後的數據
        aligned_y_values = []
        x_values = None
        
        for x, y in all_x_y_pairs:
            if x_values is None:
                x_values = np.array(x[:min_length])
            aligned_y_values.append(np.array(y[:min_length]))
        
        if not aligned_y_values:
            aggregated_metrics[metric] = ([], [], [])
            continue
        
        # 轉換為numpy array
        aligned_y_values = np.array(aligned_y_values)
        
        # 計算平均值和標準差
        mean_y = np.mean(aligned_y_values, axis=0)
        std_y = np.std(aligned_y_values, axis=0)
        
        aggregated_metrics[metric] = (x_values, mean_y, std_y)
    
    return aggregated_metrics

def remove_environment_bias(data_dict, method='z_score_within_env'):
    """消除不同環境間的偏差"""
    normalized_data = {}
    
    for custom_name, env_data_list in data_dict.items():
        normalized_env_data = []
        
        for env_idx, (x, y, std) in enumerate(env_data_list):
            y = np.array(y)
            
            if method == 'z_score_within_env':
                if np.std(y) > 0:
                    y_normalized = (y - np.mean(y)) / np.std(y)
                    std_normalized = std / np.std(y) if std is not None else None
                else:
                    y_normalized = y - np.mean(y)
                    std_normalized = std
                    
            elif method == 'min_max_within_env':
                y_min, y_max = np.min(y), np.max(y)
                if y_max - y_min > 0:
                    y_normalized = (y - y_min) / (y_max - y_min)
                    std_normalized = std / (y_max - y_min) if std is not None else None
                else:
                    y_normalized = np.zeros_like(y)
                    std_normalized = std
                    
            elif method == 'robust_within_env':
                median = np.median(y)
                q75, q25 = np.percentile(y, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    y_normalized = (y - median) / iqr
                    std_normalized = std / iqr if std is not None else None
                else:
                    y_normalized = y - median
                    std_normalized = std
                    
            elif method == 'percent_change':
                if len(y) > 0 and y[0] != 0:
                    y_normalized = ((y - y[0]) / abs(y[0])) * 100
                    std_normalized = (std / abs(y[0])) * 100 if std is not None else None
                else:
                    y_normalized = np.zeros_like(y)
                    std_normalized = std
                    
            elif method == 'baseline_normalization':
                if len(y) > 0:
                    baseline_length = max(1, len(y) // 10)
                    baseline = np.mean(y[:baseline_length])
                    if baseline != 0:
                        y_normalized = y / baseline
                        std_normalized = std / baseline if std is not None else None
                    else:
                        y_normalized = np.ones_like(y)
                        std_normalized = std
                else:
                    y_normalized = y
                    std_normalized = std
                    
            elif method == 'rank_normalization':
                y_normalized = stats.rankdata(y) / len(y)
                std_normalized = std
                
            else:
                y_normalized = y
                std_normalized = std
            
            normalized_env_data.append((x, y_normalized, std_normalized))
        
        normalized_data[custom_name] = normalized_env_data
    
    return normalized_data

def create_summary_with_bias_removal():
    """創建消除偏差後的統計比較圖"""
    print("Creating bias-removed summary comparison plots...")
    
    # 收集所有環境下每個演算法的metrics數據
    raw_metrics_data = {}  # {metric: {algorithm_name: [(x, y, std), ...]}}
    
    for prefix in prefixes:
        for algorithm_name, log_group in zip(algorithm_names, grouped_logs):
            try:
                aggregated_data = aggregate_algorithm_data(log_group, prefix)
                if aggregated_data is None:
                    continue
                    
                for metric, (x, y, std) in aggregated_data.items():
                    if len(x) == 0:
                        continue
                    
                    if metric not in raw_metrics_data:
                        raw_metrics_data[metric] = {}
                    if algorithm_name not in raw_metrics_data[metric]:
                        raw_metrics_data[metric][algorithm_name] = []
                    
                    raw_metrics_data[metric][algorithm_name].append((x, y, std))
            except Exception as e:
                print(f"Error processing {algorithm_name}, {prefix}: {e}")
    
    # 創建多種偏差消除方法的summary資料夾
    bias_removal_methods = {
        'z_score_within_env': 'Z-Score Normalization',
        'robust_within_env': 'Robust Normalization',
        'percent_change': 'Percent Change',
        'baseline_normalization': 'Baseline Normalization',
        'no_normalization': 'No Normalization (Original)'
    }
    
    for method_key, method_name in bias_removal_methods.items():
        print(f"Processing with {method_name}...")
        
        summary_dir = os.path.join(args.save_dir, f"summary_{method_key}")
        os.makedirs(summary_dir, exist_ok=True)
        
        for metric, alg_data in raw_metrics_data.items():
            if method_key != 'no_normalization':
                normalized_alg_data = remove_environment_bias(alg_data, method_key)
            else:
                normalized_alg_data = alg_data
            
            plt.figure(figsize=(12, 7))
            
            for algorithm_name, env_data_list in normalized_alg_data.items():
                if not env_data_list:
                    continue
                
                min_length = min(len(y) for x, y, std in env_data_list)
                
                all_y_values = []
                x_values = None
                
                for x, y, std in env_data_list:
                    if x_values is None:
                        x_values = x[:min_length]
                    y_trimmed = y[:min_length]
                    all_y_values.append(y_trimmed)
                
                if not all_y_values:
                    continue
                
                all_y_values = np.array(all_y_values)
                mean_y = np.mean(all_y_values, axis=0)
                std_y = np.std(all_y_values, axis=0)
                stderr_y = std_y / np.sqrt(len(all_y_values))
                
                smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
                smoother.smooth(mean_y)
                smoothed_mean = smoother.smooth_data[0]
                
                smoother_stderr = LowessSmoother(smooth_fraction=0.05, iterations=1)
                smoother_stderr.smooth(stderr_y)
                smoothed_stderr = smoother_stderr.smooth_data[0]
                
                plt.plot(x_values, smoothed_mean, linewidth=2.5, label=f"{algorithm_name}")
                
                ci_95 = 1.96 * smoothed_stderr
                plt.fill_between(x_values,
                                smoothed_mean - ci_95,
                                smoothed_mean + ci_95,
                                alpha=0.2)
            
            plt.xlabel("steps")
            
            if method_key == 'z_score_within_env':
                plt.ylabel(f"{metric} (Z-score normalized)")
            elif method_key == 'robust_within_env':
                plt.ylabel(f"{metric} (Robust normalized)")
            elif method_key == 'percent_change':
                plt.ylabel(f"{metric} (% change from initial)")
            elif method_key == 'baseline_normalization':
                plt.ylabel(f"{metric} (normalized to baseline)")
            else:
                plt.ylabel(metric)
            
            plt.title(f"{metric} - {method_name}\n(95% CI across {len(prefixes)} environments)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            safe_metric_name = metric.replace('/', '_')
            save_path = os.path.join(summary_dir, f"{safe_metric_name}_summary.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Plots with {method_name} saved to {summary_dir}")

if __name__ == "__main__":
    prefixes = args.prefixes.split(',')
    
    # 將log資料夾按演算法分組
    grouped_logs = group_logs_by_algorithm(args.log_names, args.repeat_counts)
    
    # 確定演算法名稱
    if args.custom_names:
        if len(args.custom_names) != len(grouped_logs):
            raise ValueError(f"Number of custom names ({len(args.custom_names)}) must match number of algorithms ({len(grouped_logs)})")
        algorithm_names = args.custom_names
    else:
        algorithm_names = [f"Algorithm_{i+1}" for i in range(len(grouped_logs))]
    
    print(f"Processing {len(grouped_logs)} algorithms:")
    for i, (name, logs) in enumerate(zip(algorithm_names, grouped_logs)):
        print(f"  {name}: {len(logs)} repeated experiments")
    
    # 為每個prefix創建比較圖
    for prefix in prefixes:
        all_metrics_data = {}
        
        for algorithm_name, log_group in zip(algorithm_names, grouped_logs):
            try:
                # 聚合同一演算法的數據
                aggregated_data = aggregate_algorithm_data(log_group, prefix)
                if aggregated_data is None:
                    continue
                
                for metric, (x, mean_y, std_y) in aggregated_data.items():
                    if len(x) == 0:
                        continue  # 這個演算法沒有這個metric，跳過
                    
                    if metric not in all_metrics_data:
                        all_metrics_data[metric] = []
                    
                    all_metrics_data[metric].append((algorithm_name, x, mean_y, std_y))
            except Exception as e:
                print(f"Error processing {algorithm_name}, {prefix}: {e}")
        
        # 繪製每個metric的圖（即使只有部分演算法有數據）
        for metric, data_list in all_metrics_data.items():
            if not data_list:
                continue  # 完全沒有演算法有這個metric
                
            plt.figure(figsize=(11, 6))
            
            for algorithm_name, x, mean_y, std_y in data_list:
                # 平滑平均值
                smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
                smoother.smooth(mean_y)
                smoothed_mean = smoother.smooth_data[0]
                
                # 平滑標準差
                smoother_std = LowessSmoother(smooth_fraction=0.05, iterations=1)
                smoother_std.smooth(std_y)
                smoothed_std = smoother_std.smooth_data[0]
                
                # 繪製平均值線
                plt.plot(x, smoothed_mean, linewidth=2, label=algorithm_name)
                
                # 繪製標準差範圍
                plt.fill_between(x,
                                smoothed_mean - smoothed_std,
                                smoothed_mean + smoothed_std,
                                alpha=0.2)
            
            plt.xlabel("steps")
            plt.ylabel(metric)
            plt.title(f"{metric} (Prefix: {prefix})")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            save_subdir = os.path.join(args.save_dir, prefix)
            os.makedirs(save_subdir, exist_ok=True)
            
            safe_metric_name = metric.replace('/', '_')
            save_path = os.path.join(save_subdir, f"{safe_metric_name}.png")
            plt.savefig(save_path)
            plt.close()
    
    # 創建summary比較圖
    print("Creating summary comparison plots...")
    
    summary_metrics_data = {}  # {metric: {algorithm_name: [(x, y, std), ...]}}
    
    for prefix in prefixes:
        for algorithm_name, log_group in zip(algorithm_names, grouped_logs):
            try:
                aggregated_data = aggregate_algorithm_data(log_group, prefix)
                if aggregated_data is None:
                    continue
                
                for metric, (x, mean_y, std_y) in aggregated_data.items():
                    if len(x) == 0:
                        continue
                    
                    if metric not in summary_metrics_data:
                        summary_metrics_data[metric] = {}
                    if algorithm_name not in summary_metrics_data[metric]:
                        summary_metrics_data[metric][algorithm_name] = []
                    
                    summary_metrics_data[metric][algorithm_name].append((x, mean_y, std_y))
            except Exception as e:
                print(f"Error in summary: {e}")
    
    # 創建summary資料夾
    summary_dir = os.path.join(args.save_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # 為每個metric創建統計比較圖
    for metric, alg_data in summary_metrics_data.items():
        plt.figure(figsize=(12, 7))
        
        for algorithm_name, env_data_list in alg_data.items():
            if not env_data_list:
                continue
            
            # 找到最短長度
            min_length = min(len(y) for x, y, std in env_data_list)
            
            # 收集所有環境的數據
            all_y_values = []
            x_values = None
            
            for x, y, std in env_data_list:
                if x_values is None:
                    x_values = x[:min_length]
                y_trimmed = y[:min_length]
                all_y_values.append(y_trimmed)
            
            if not all_y_values:
                continue
            
            all_y_values = np.array(all_y_values)
            
            # 計算跨環境的平均值和標準誤差
            mean_y = np.mean(all_y_values, axis=0)
            std_y = np.std(all_y_values, axis=0)
            stderr_y = std_y / np.sqrt(len(all_y_values))
            
            # 平滑處理
            smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
            smoother.smooth(mean_y)
            smoothed_mean = smoother.smooth_data[0]
            
            smoother_stderr = LowessSmoother(smooth_fraction=0.05, iterations=1)
            smoother_stderr.smooth(stderr_y)
            smoothed_stderr = smoother_stderr.smooth_data[0]
            
            # 繪製平均值線
            plt.plot(x_values, smoothed_mean, linewidth=2.5, label=f"{algorithm_name}")
            
            # 繪製95%信賴區間
            ci_95 = 1.96 * smoothed_stderr
            plt.fill_between(x_values,
                            smoothed_mean - ci_95,
                            smoothed_mean + ci_95,
                            alpha=0.2)
        
        plt.xlabel("steps")
        plt.ylabel(metric)
        plt.title(f"{metric} - Summary Comparison\n(Mean ± 95% CI across {len(prefixes)} environments)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        safe_metric_name = metric.replace('/', '_')
        save_path = os.path.join(summary_dir, f"{safe_metric_name}_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Summary plots saved to {summary_dir}")
    
    # 執行偏差消除的統計分析
    create_summary_with_bias_removal()
    print("Bias removal analysis completed!")