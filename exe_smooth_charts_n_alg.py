import subprocess
import os
import time
import datetime
import argparse

def find_tensorboard_root(start_path, max_depth=5):
    """
    遞迴尋找包含prefix格式資料夾的真實根目錄
    例如：尋找包含 '0_XXX', '1_XXX' 等資料夾的目錄
    
    Args:
        start_path: 開始搜尋的路徑
        max_depth: 最大搜尋深度
        
    Returns:
        找到的tensorboard根目錄，如果找不到則返回None
    """
    if max_depth <= 0:
        return None
    
    try:
        # 檢查當前目錄是否包含prefix格式的資料夾 (0_XXX, 1_XXX, etc.)
        subdirs = [f for f in os.scandir(start_path) if f.is_dir()]
        has_prefix_folders = any(
            f.name.split('_')[0].isdigit() and f.name.startswith(('0_', '1_', '2_', '3_', '4_', '5_', '6_', '7_', '8_', '9_'))
            for f in subdirs
        )
        
        if has_prefix_folders:
            return start_path
        
        # 如果當前目錄只有一個子目錄，繼續往下找
        if len(subdirs) == 1:
            return find_tensorboard_root(subdirs[0].path, max_depth - 1)
        
        # 如果有多個子目錄，檢查每一個
        for subdir in subdirs:
            result = find_tensorboard_root(subdir.path, max_depth - 1)
            if result:
                return result
                
    except PermissionError:
        pass
    
    return None

def scan_experiment_structure(root_dir, auto_find_tb_root=True, exclude_dir=None):
    """
    掃描實驗資料夾結構
    
    資料夾結構範例:
    root/
    ├── alg1/
    │   ├── rep1/
    │   │   └── [nested_folders]/
    │   │       └── environment_name/
    │   │           ├── 0_XXX/
    │   │           ├── 1_XXX/
    │   │           └── ...
    │   └── rep2/
    │       └── ...
    └── alg2/
        └── ...
    
    Args:
        root_dir: 根目錄路徑
        auto_find_tb_root: 是否自動尋找tensorboard根目錄
        
    Returns:
        dict: {
            'algorithm_name': {
                'paths': [rep1_tb_root, rep2_tb_root, ...],
                'count': number_of_repetitions,
                'original_paths': [rep1_path, rep2_path, ...]
            }
        }
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    
    experiments = {}
    
    # 掃描第一層：演算法資料夾
    alg_dirs = [f for f in os.scandir(root_dir) if f.is_dir()]
    
    for alg_dir in sorted(alg_dirs, key=lambda x: x.name):
        alg_name = alg_dir.name
        
        # 掃描第二層：重複實驗資料夾
        rep_dirs = [f for f in os.scandir(alg_dir.path) if f.is_dir()]
        
        if not rep_dirs:
            continue
        
        tb_roots = []
        original_paths = []

        def in_exclude(path):
            if exclude_dir is None:
                return False
            for e in exclude_dir:
                if os.path.samefile(path, e):
                    return True
        
        
        for rep_dir in sorted(rep_dirs, key=lambda x: x.name):
            if in_exclude(rep_dir.path):
                print(f"Skipping excluded directory: {rep_dir.path}")
                continue

            original_paths.append(rep_dir.path)
            
            if auto_find_tb_root:
                # 自動尋找包含tensorboard資料的真實根目錄
                tb_root = find_tensorboard_root(rep_dir.path)
                if tb_root:
                    if in_exclude(tb_root):
                        print(f"Skipping excluded directory: {tb_root}")
                        continue
                    tb_roots.append(tb_root)
                    print(f"  Found TB root for {alg_name}/{rep_dir.name}:")
                    print(f"    {tb_root}")
                else:
                    print(f"  Warning: Could not find TB root for {alg_name}/{rep_dir.name}")
            else:
                tb_roots.append(rep_dir.path)
        
        if tb_roots:
            experiments[alg_name] = {
                'paths': tb_roots,
                'count': len(tb_roots),
                'original_paths': original_paths
            }
    
    return experiments

def generate_comparison_plots(experiments, save_root='results', iqr_factor=2.0, prefixes='0,1,2,3,4'):
    """
    生成所有比較圖
    
    Args:
        experiments: 從 scan_experiment_structure 得到的實驗結構
        save_root: 儲存結果的根目錄
        iqr_factor: IQR因子
        prefixes: 環境前綴
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 準備演算法資訊
    alg_names = list(experiments.keys())
    num_algs = len(alg_names)
    
    if num_algs == 0:
        print("No algorithms found!")
        return
    
    print("=" * 60)
    print(f"Found {num_algs} algorithms:")
    for alg_name, info in experiments.items():
        print(f"  - {alg_name}: {info['count']} repetitions")
    print("=" * 60)
    print()
    
    # 1. 生成多演算法比較圖（如果有多於1個演算法）
    if num_algs > 1:
        print("=" * 60)
        print("STEP 1: Generating multi-algorithm comparison plots")
        print("=" * 60)
        
        # 準備所有log路徑和repeat counts
        all_log_paths = []
        repeat_counts = []
        
        for alg_name in alg_names:
            all_log_paths.extend(experiments[alg_name]['paths'])
            repeat_counts.append(experiments[alg_name]['count'])
        
        # 構建命令
        cmd = [
            'python', './utils/smooth_chart_multi_dir.py',
            '-l', *all_log_paths,
            '-n', *alg_names,
            '-r', *[str(c) for c in repeat_counts],
            '-s', os.path.join(save_root, f'{now}_comparison'),
            '-p', prefixes,
            '--iqr_factor', str(iqr_factor)
        ]
        
        print(f"Command: python smooth_chart_multi_dir.py -l [paths...] -n [names...] -r [counts...]")
        print(f"Processing {len(all_log_paths)} total experiments across {num_algs} algorithms")
        print()
        
        try:
            subprocess.run(cmd, check=True)
            print("✓ Multi-algorithm comparison completed!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error in multi-algorithm comparison: {e}")
        except FileNotFoundError:
            print(f"✗ Script not found: ./utils/smooth_chart_multi_dir.py")
        
        print()
        time.sleep(1)
    
    # 2. 生成每個演算法的單獨分析圖
    print("=" * 60)
    print("STEP 2: Generating individual algorithm plots")
    print("=" * 60)
    
    for i, (alg_name, info) in enumerate(experiments.items(), 1):
        print(f"[{i}/{num_algs}] Processing: {alg_name}")
        print(f"  Repetitions: {info['count']}")
        
        # 構建命令
        cmd = [
            'python', './utils/smooth_chart_one_server.py',
            '-l', *info['paths'],
            '-n', alg_name,
            '-s', os.path.join(save_root, f'{now}_{alg_name}'),
            '-p', prefixes,
            '--iqr_factor', str(iqr_factor)
        ]
        
        print(f"  Experiment paths:")
        for j, (orig_path, tb_path) in enumerate(zip(info['original_paths'], info['paths']), 1):
            print(f"    [{j}] {os.path.basename(orig_path)}")
            print(f"        → {os.path.relpath(tb_path, orig_path)}")
        
        try:
            subprocess.run(cmd, check=True)
            print(f"  ✓ {alg_name} completed!")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error processing {alg_name}: {e}")
        except FileNotFoundError:
            print(f"  ✗ Script not found: ./utils/smooth_chart_one_server.py")
        
        print()
        time.sleep(0.5)
    
    print("=" * 60)
    print("ALL PLOTS GENERATED!")
    print("=" * 60)
    print(f"Results saved to: {save_root}/{now}_*")
    print()

def print_structure_preview(experiments):
    """打印資料夾結構預覽"""
    print("\nDetected experiment structure:")
    print("=" * 60)
    for alg_name, info in experiments.items():
        print(f"📁 {alg_name}/ ({info['count']} repetitions)")
        for i, (orig_path, tb_path) in enumerate(zip(info['original_paths'], info['paths']), 1):
            rep_name = os.path.basename(orig_path)
            rel_path = os.path.relpath(tb_path, orig_path)
            if rel_path == '.':
                print(f"   ├── {rep_name}/")
            else:
                print(f"   ├── {rep_name}/")
                print(f"   │    └── {rel_path}/")
    print("=" * 60)
    print()

def main():
    parser = argparse.ArgumentParser(
        description='Automatically generate comparison plots from experiment directory structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example directory structure:
  root/
  ├── PBPPO_regul0/
  │   ├── round_01/
  │   │   └── 2025_12_22.../
  │   │       └── CartPoleSwingUpV1.../
  │   │           ├── 0_PerturbPPO/
  │   │           ├── 1_PerturbPPO/
  │   │           └── ...
  │   └── round_02/
  │       └── ...
  └── A2C/
      └── ...

Usage:
  python auto_plot.py -r ./multiagent
  python auto_plot.py -r ./experiments -s ./my_results --iqr_factor 2.5
  python auto_plot.py -r ./experiments --no-auto-find  # 不自動尋找TB根目錄
        """
    )
    
    parser.add_argument('-r', '--root_dir', 
                        type=str, 
                        required=True,
                        help='Root directory containing algorithm folders')
    
    parser.add_argument('-s', '--save_dir', 
                        type=str, 
                        default='results',
                        help='Directory to save plots (default: results)')
    
    parser.add_argument('-e', '--exclude_dir', 
                        nargs='+',
                        type=str, 
                        default=None,
                        help='Directory linst not to include in plots(default: None)')
    
    parser.add_argument('--iqr_factor', 
                        type=float, 
                        default=2.0,
                        help='IQR factor for outlier removal (default: 2.0)')
    
    parser.add_argument('-p', '--prefixes', 
                        type=str, 
                        default='0,1,2',
                        help='Comma-separated list of environment prefixes (default: 0,1,2,3,4)')
    
    parser.add_argument('--no-auto-find', 
                        action='store_true',
                        help='Disable automatic tensorboard root finding')
    
    parser.add_argument('--dry-run', 
                        action='store_true',
                        help='Only show detected structure without running plots')
    
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Verbose output for debugging')
    
    args = parser.parse_args()
    
    try:
        # 掃描實驗結構
        print(f"Scanning directory: {args.root_dir}")
        if not args.no_auto_find:
            print("Auto-finding tensorboard roots (this may take a moment)...")
        print()
        
        experiments = scan_experiment_structure(
            args.root_dir, 
            auto_find_tb_root=not args.no_auto_find,
            exclude_dir=args.exclude_dir
        )
        
        if not experiments:
            print("No experiments found in the root directory!")
            print("\nTroubleshooting:")
            print("  1. Make sure the directory structure is correct")
            print("  2. Check if subdirectories contain folders with prefixes like '0_XXX', '1_XXX'")
            print("  3. Try with --verbose flag for more information")
            return
        
        print()
        
        # 顯示結構預覽
        print_structure_preview(experiments)
        
        # 如果是dry-run，只顯示結構
        if args.dry_run:
            print("Dry-run mode: No plots will be generated.")
            print("\nTo generate plots, run without --dry-run flag:")
            print(f"  python {os.path.basename(__file__)} -r {args.root_dir}")
            return
        
        # 確認執行
        print(f"IQR Factor: {args.iqr_factor}")
        print(f"Environment Prefixes: {args.prefixes}")
        print(f"Save Directory: {args.save_dir}")
        print()
        
        response = input("Proceed with plot generation? [Y/n]: ").strip().lower()
        if response and response not in ['y', 'yes']:
            print("Cancelled by user.")
            return
        
        print()
        
        # 生成圖表
        generate_comparison_plots(
            experiments, 
            save_root=args.save_dir,
            iqr_factor=args.iqr_factor,
            prefixes=args.prefixes
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()