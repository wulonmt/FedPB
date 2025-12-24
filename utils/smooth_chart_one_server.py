import argparse
import os
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import matplotlib.pyplot as plt
from tsmoothie import LowessSmoother
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_names", nargs='+', help="log names for repeated experiments", type=str, required=True)
parser.add_argument("-n", "--custom_name", help="custom name for the algorithm", type=str)
parser.add_argument("-s", "--save_dir", help="directory to save plots", type=str, required=True)
parser.add_argument("-o", "--remove_outliers", action="store_true", help="remove outliers from the data")
parser.add_argument("--iqr_factor", type=float, default=1.5, help="IQR factor for outlier removal (default: 1.5)")
parser.add_argument("-p", "--prefixes", type=str, default="0,1,2,3,4", help="Comma-separated list of prefixes of subdirectories to process (default: 0,1,2,3,4)")
parser.add_argument("--show_individual", action="store_true", help="Show individual experiment curves in addition to mean")
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

def aggregate_repeated_experiments(log_names, prefix):
    """聚合多個重複實驗的數據"""
    all_metrics_list = []
    
    for log_name in log_names:
        try:
            metrics_data = process_folder(log_name, prefix + "_")
            all_metrics_list.append(metrics_data)
        except Exception as e:
            print(f"Warning: {e} at {log_name}, {prefix}")
            continue
    
    if not all_metrics_list:
        return None
    
    # 聚合數據
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
            aggregated_metrics[metric] = {
                'mean': ([], []),
                'std': ([], []),
                'individual': []
            }
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
            aggregated_metrics[metric] = {
                'mean': ([], []),
                'std': ([], []),
                'individual': []
            }
            continue
        
        # 轉換為numpy array
        aligned_y_values = np.array(aligned_y_values)
        
        # 計算平均值和標準差
        mean_y = np.mean(aligned_y_values, axis=0)
        std_y = np.std(aligned_y_values, axis=0)
        
        aggregated_metrics[metric] = {
            'mean': (x_values, mean_y),
            'std': (x_values, std_y),
            'individual': all_x_y_pairs  # 保存個別實驗數據
        }
    
    return aggregated_metrics

if __name__ == "__main__":
    prefixes = args.prefixes.split(',')
    custom_name = args.custom_name if args.custom_name else "algorithm"
    
    print(f"Processing {len(args.log_names)} repeated experiments for {custom_name}")
    
    # 為每個prefix創建圖表
    all_prefix_metrics = {}
    
    for prefix in prefixes:
        print(f"Processing prefix: {prefix}")
        try:
            aggregated_data = aggregate_repeated_experiments(args.log_names, prefix)
            if aggregated_data is None:
                continue
            
            for metric, data in aggregated_data.items():
                if len(data['mean'][0]) == 0:
                    continue
                
                if metric not in all_prefix_metrics:
                    all_prefix_metrics[metric] = []
                
                all_prefix_metrics[metric].append({
                    'prefix': prefix,
                    'mean': data['mean'],
                    'std': data['std'],
                    'individual': data['individual']
                })
        except Exception as e:
            print(f"Error at prefix {prefix}: {e}")
    
    # 繪製圖表
    for metric, prefix_data_list in all_prefix_metrics.items():
        if not prefix_data_list:
            continue
            
        plt.figure(figsize=(12, 7))
        
        for data in prefix_data_list:
            prefix = data['prefix']
            x, mean_y = data['mean']
            
            if len(x) == 0:
                continue
            
            _, std_y = data['std']
            
            # 平滑平均值
            smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
            smoother.smooth(mean_y)
            smoothed_mean = smoother.smooth_data[0]
            
            # 平滑標準差
            smoother_std = LowessSmoother(smooth_fraction=0.05, iterations=1)
            smoother_std.smooth(std_y)
            smoothed_std = smoother_std.smooth_data[0]
            
            # 如果需要顯示個別實驗曲線
            if args.show_individual:
                for i, (ind_x, ind_y) in enumerate(data['individual']):
                    plt.plot(ind_x, ind_y, linewidth=0.8, alpha=0.3, 
                            linestyle='--', color=f'C{prefix_data_list.index(data)}')
            
            # 繪製平均值線
            plt.plot(x, smoothed_mean, linewidth=2.5, 
                    label=f"{prefix} (mean ± std, n={len(data['individual'])})")
            
            # 繪製標準差範圍
            plt.fill_between(x,
                            smoothed_mean - smoothed_std,
                            smoothed_mean + smoothed_std,
                            alpha=0.25)
        
        plt.xlabel("steps", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        
        if len(args.log_names) > 1:
            plt.title(f"{metric}\n{custom_name} - {len(args.log_names)} repeated experiments", 
                     fontsize=13, fontweight='bold')
        else:
            plt.title(f"{metric}\n{custom_name}", fontsize=13, fontweight='bold')
        
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        save_subdir = os.path.join(args.save_dir, custom_name)
        os.makedirs(save_subdir, exist_ok=True)
        
        safe_metric_name = metric.replace('/', '_')
        save_path = os.path.join(save_subdir, f"{safe_metric_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"All plots saved to {os.path.join(args.save_dir, custom_name)}")
    
    # 創建summary圖（跨所有prefix的平均）
    print("Creating summary plots across all environments...")
    
    summary_metrics_data = {}
    
    for prefix in prefixes:
        try:
            aggregated_data = aggregate_repeated_experiments(args.log_names, prefix)
            if aggregated_data is None:
                continue
            
            for metric, data in aggregated_data.items():
                x, mean_y = data['mean']
                _, std_y = data['std']
                
                if len(x) == 0:
                    continue
                
                if metric not in summary_metrics_data:
                    summary_metrics_data[metric] = []
                
                summary_metrics_data[metric].append((x, mean_y, std_y))
        except Exception as e:
            print(f"Error in summary for prefix {prefix}: {e}")
    
    # 創建summary資料夾
    summary_dir = os.path.join(args.save_dir, custom_name, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # 為每個metric創建跨環境的統計圖
    for metric, env_data_list in summary_metrics_data.items():
        if not env_data_list:
            continue
        
        plt.figure(figsize=(12, 7))
        
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
        plt.plot(x_values, smoothed_mean, linewidth=3, 
                label=f"{custom_name} (mean across {len(prefixes)} envs)")
        
        # 繪製95%信賴區間
        ci_95 = 1.96 * smoothed_stderr
        plt.fill_between(x_values,
                        smoothed_mean - ci_95,
                        smoothed_mean + ci_95,
                        alpha=0.3,
                        label='95% Confidence Interval')
        
        # 也繪製標準差範圍作為參考
        plt.fill_between(x_values,
                        smoothed_mean - std_y[:min_length],
                        smoothed_mean + std_y[:min_length],
                        alpha=0.15,
                        label='±1 Standard Deviation')
        
        plt.xlabel("steps", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f"{metric} - Summary\n{custom_name} ({len(args.log_names)} runs × {len(prefixes)} environments)", 
                 fontsize=13, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        safe_metric_name = metric.replace('/', '_')
        save_path = os.path.join(summary_dir, f"{safe_metric_name}_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Summary plots saved to {summary_dir}")
    print("Processing complete!")