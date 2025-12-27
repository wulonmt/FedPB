import gymnasium as gym
from matplotlib import pyplot as plt
import argparse
import Env
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from utils.init_pos_config import get_init_pos, get_init_list, assert_alarm

import cv2
import pandas as pd

from collections import OrderedDict
import torch as th
import os
from typing import Dict, List, Optional, Tuple
from glob import glob
from pathlib import Path

from utils.init_pos_config import is_valid_env, is_costum_env
from utils.PerturbationPPO import PerturbationPPO
from normalized_env import get_eval_norm_env

MODEL_DICT = {
    "PPO": PPO,
    "RNPPO": PPO,
    "PBPPO_regul0": PerturbationPPO,
    "PBPPO_regul1": PerturbationPPO,
}

def make_standard_eval_env(env_name: str, npz_path: str, index: int):
    """
    創建評估用的環境
    
    Args:
        env_name: 環境名稱
        index: 環境索引
    
    Returns:
        gym.Env: 評估環境
    """
    env_root = Path(npz_path).parent
    vec_norm_path = env_root / Path(env_name) / Path("0_PerturbPPO")
    env = get_eval_norm_env(
        env_name=env_name,
        path=str(vec_norm_path),
        index=index)
    return env

def load_params_to_model(env, npz_path, model_class=PPO, verbose=False):
    """
    從 .npz 文件加載 policy 參數到 PPO 模型（不含 value_net）
    
    Args:
        env: gym 環境
        npz_path: .npz 文件路徑
        model_class: PPO 模型類別 (PerturbationPPO 或 RandomNoisePPO)
        verbose: 是否打印調試信息
    
    Returns:
        loaded_model: 加載參數後的模型
    """
    # 加載 .npz 文件中的參數
    parameters = np.load(npz_path)
    ndarrays = [parameters[key] for key in parameters.files]
    
    # 創建新模型
    model = model_class("MlpPolicy", 
                        env, 
                        policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
                        device="cpu")
    
    # 獲取當前 policy 的 state_dict
    current_state_dict = model.policy.state_dict()
    
    # 只取 policy 相關的 keys（排除 value_net）
    policy_keys = [key for key in current_state_dict.keys() if "value_net" not in key]
    
    if len(ndarrays) != len(policy_keys):
        print(f"Warning: Parameter count mismatch!")
        print(f"Expected {len(policy_keys)}, got {len(ndarrays)}")
        if verbose:
            print(f"Policy keys: {policy_keys}")
    
    # 創建新的 state_dict
    new_state_dict = OrderedDict()
    
    # 載入 policy 參數
    for key, array in zip(policy_keys, ndarrays):
        new_state_dict[key] = th.tensor(array)
    
    # 保留本地的 value_net 參數
    for key in current_state_dict.keys():
        if 'value_net' in key:
            new_state_dict[key] = current_state_dict[key]
    
    # 載入到 policy
    model.policy.load_state_dict(new_state_dict, strict=True)
    
    if verbose:
        print(f"Loaded {len(policy_keys)} policy parameters")
    
    return model

def scan_experiment_root(root_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    掃描實驗根目錄，自動識別演算法和所有重複實驗
    
    資料夾結構:
    root/
    ├── PBPPO_regul0/
    │   ├── 2025_12_22_15_40_c3_CartPoleSwingUpV1WithAdjustablePole-v0_PBPPO_regul0_rep1/
    │   │   └── *.npz
    │   └── 2025_12_22_15_40_c3_CartPoleSwingUpV1WithAdjustablePole-v0_PBPPO_regul0_rep2/
    │       └── *.npz
    ├── PPO/
    │   ├── ...rep1/
    │   └── ...rep2/
    └── RNPPO/
        └── ...
    
    Args:
        root_dir: 根目錄路徑
    
    Returns:
        Dict[algorithm_name, List[(env_name, npz_path)]]
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    
    algorithm_models = {}
    
    # 遍歷第一層：演算法資料夾
    for alg_dir in root_path.iterdir():
        if not alg_dir.is_dir():
            continue
        
        alg_name = alg_dir.name
        print(f"\n掃描演算法: {alg_name}")
        
        models_list = []
        
        # 遍歷第二層：重複實驗資料夾 (repX)
        for rep_dir in alg_dir.iterdir():
            if not rep_dir.is_dir():
                continue
            
            # 尋找 .npz 文件
            npz_files = list(rep_dir.glob("*.npz"))
            
            if not npz_files:
                print(f"  ⚠ 警告: {rep_dir.name} 中沒有找到 .npz 文件")
                continue
            
            if len(npz_files) > 1:
                print(f"  ⚠ 警告: {rep_dir.name} 中有多個 .npz 文件，使用第一個")
            
            npz_path = str(npz_files[0])
            
            env_name = None
            for env_dir in rep_dir.iterdir():
                if is_valid_env(env_dir.name):
                    env_name = env_dir.name
            
            if not env_name:
                print(f"  ⚠ 警告: 無法從 {rep_dir.name} 解析環境名稱")
                continue
            
            models_list.append((env_name, npz_path, rep_dir.name))
            print(f"  ✓ 找到: {rep_dir.name}")
            print(f"    環境: {env_name}")
            print(f"    模型: {npz_path}")
        
        if models_list:
            algorithm_models[alg_name] = models_list
            print(f"  總計: {len(models_list)} 個重複實驗")
    
    print(f"\n" + "="*70)
    print(f"掃描完成！找到 {len(algorithm_models)} 個演算法")
    for alg_name, models in algorithm_models.items():
        print(f"  - {alg_name}: {len(models)} 個重複實驗")
    print("="*70)
    
    return algorithm_models

def evaluate_algorithms_on_all_indices(
    algorithm_models: Dict[str, List[Tuple[str, str, str]]],
    n_eval_episodes: int = 10,
    output_dir: str = ".",
    output_prefix: str = "evaluation"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    評估所有演算法在不同環境索引上的表現，對相同演算法的多個 rep 取平均
    
    Args:
        algorithm_models: 字典，格式為 {alg_name: [(env_name, npz_path, rep_name), ...]}
        model_class: PPO 模型類別 (PerturbationPPO, RandomNoisePPO, 等)
        n_eval_episodes: 每個環境索引評估的 episode 數量
        output_dir: 輸出目錄
        output_prefix: 輸出文件前綴
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (mean_df, std_df)
    """
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化結果字典
    results_mean = {}
    results_std = {}
    
    print("\n" + "="*80)
    print("開始評估演算法...")
    print("="*80)
    
    # 遍歷所有演算法
    for alg_name, models_list in algorithm_models.items():
        print(f"\n{'='*80}")
        print(f"評估演算法: {alg_name} ({len(models_list)} 個重複實驗)")
        print(f"{'='*80}")
        
        # 儲存每個 rep 的結果
        rep_means_list = []
        rep_stds_list = []
        
        # 評估每個重複實驗
        for rep_idx, (env_name, npz_path, rep_name) in enumerate(models_list, 1):
            print(f"\n[{rep_idx}/{len(models_list)}] 評估: {rep_name}")
            
            rep_means = {}
            rep_stds = {}
            
            # 獲取環境的索引列表
            try:
                init_list = get_init_list(env_name)
                n_indices = len(init_list)
            except:
                n_indices = 5
                print(f"  ⚠ 警告: 無法獲取 {env_name} 的初始化列表，使用默認索引 0-4")
            
            # 評估每個索引
            for index in range(n_indices):
                print(f"  索引 {index}...", end=" ")
                
                try:
                    # 創建評估環境
                    # 使用index 0的vec_normalize.pkl作為標準環境
                    eval_env = make_standard_eval_env(env_name=env_name, npz_path=npz_path, index=index)
                    
                    # 加載模型
                    model = load_params_to_model(eval_env, npz_path, MODEL_DICT[alg_name], verbose=False)
                    
                    # 評估模型
                    mean_reward, std_reward = evaluate_policy(
                        model, 
                        eval_env, 
                        n_eval_episodes=n_eval_episodes,
                        deterministic=True
                    )
                    
                    rep_means[f"Index_{index}"] = mean_reward
                    rep_stds[f"Index_{index}"] = std_reward
                    
                    print(f"Mean: {mean_reward:.2f}, Std: {std_reward:.2f}")
                    
                    # 關閉環境
                    eval_env.close()
                    
                except Exception as e:
                    print(f"失敗: {str(e)}")
                    rep_means[f"Index_{index}"] = np.nan
                    rep_stds[f"Index_{index}"] = np.nan
            
            rep_means_list.append(rep_means)
            rep_stds_list.append(rep_stds)
        
        # 計算該演算法所有 rep 的平均
        if rep_means_list:
            # 轉換為 DataFrame 方便計算
            rep_means_df = pd.DataFrame(rep_means_list)
            rep_stds_df = pd.DataFrame(rep_stds_list)
            
            # 計算平均值
            alg_mean = rep_means_df.mean().to_dict()
            alg_std = rep_stds_df.mean().to_dict()
            
            results_mean[alg_name] = alg_mean
            results_std[alg_name] = alg_std
            
            print(f"\n{alg_name} 平均結果:")
            for key, value in alg_mean.items():
                print(f"  {key}: {value:.2f} ± {alg_std[key]:.2f}")
    
    # 創建 DataFrame
    mean_df = pd.DataFrame.from_dict(results_mean, orient='index')
    std_df = pd.DataFrame.from_dict(results_std, orient='index')
    
    # 確保列的順序
    if not mean_df.empty:
        n_indices = len(mean_df.columns)
        columns_order = [f"Index_{i}" for i in range(n_indices)]
        mean_df = mean_df[columns_order]
        std_df = std_df[columns_order]
    
    # 輸出到 CSV
    mean_csv_path = os.path.join(output_dir, f"{output_prefix}_mean.csv")
    std_csv_path = os.path.join(output_dir, f"{output_prefix}_std.csv")
    
    mean_df.to_csv(mean_csv_path)
    std_df.to_csv(std_csv_path)
    
    print("\n" + "="*80)
    print("評估完成!")
    print(f"Mean rewards 已儲存至: {mean_csv_path}")
    print(f"Std rewards 已儲存至: {std_csv_path}")
    print("="*80)
    
    # 打印結果摘要
    print("\n=== Mean Rewards (averaged across reps) ===")
    print(mean_df.to_string())
    print("\n=== Std Rewards (averaged across reps) ===")
    print(std_df.to_string())
    
    return mean_df, std_df

def record_all_models(
    algorithm_models: Dict[str, List[Tuple[str, str, str]]],
    video_length: int = None,
    episodes: int = 3
):
    """
    錄製所有模型的影片
    
    Args:
        algorithm_models: 字典，格式為 {alg_name: [(env_name, npz_path, rep_name), ...]}
        model_class: PPO 模型類別
        video_length: 影片長度
        episodes: episodes 數量
    """
    print("\n" + "="*80)
    print("開始錄製影片...")
    print("="*80)
    
    for alg_name, models_list in algorithm_models.items():
        print(f"\n錄製演算法: {alg_name}")
        
        for env_name, npz_path, rep_name in models_list:
            print(f"  錄製: {rep_name}")
            
            # 獲取影片保存目錄（在 rep 資料夾內）
            npz_dir = os.path.dirname(npz_path)
            video_dir = os.path.join(npz_dir, "videos")
            
            try:
                # 加載模型
                env = make_standard_eval_env(env_name=env_name, npz_path=npz_path, index=0)
                model = load_params_to_model(env, npz_path, MODEL_DICT[alg_name], verbose=False)
                env.close()
                
                # 錄製所有索引的影片
                init_list = get_init_list(env_name)
                
                for i in range(len(init_list)):
                    record_single_model(
                        env_name=env_name,
                        index=i,
                        model=model,
                        npz_path=npz_path,
                        save_dir=video_dir,
                        video_length=video_length,
                        episodes=episodes
                    )
                
                print(f"    ✓ 影片已儲存至: {video_dir}")
                
            except Exception as e:
                print(f"    ✗ 錄製失敗: {str(e)}")


def record_single_model(
    env_name: str,
    index: int,
    model,
    npz_path: str,
    save_dir: str,
    video_length: int = None,
    episodes: int = 3
):
    """
    錄製單一模型在特定索引的影片
    """
    vec_env = make_standard_eval_env(env_name=env_name, npz_path=npz_path, index=index)
    
    if video_length is None:
        max_steps = vec_env.envs[0].spec.max_episode_steps
        video_length = max_steps * episodes
    
    vec_env = VecVideoRecorder(
        vec_env,
        save_dir,
        record_video_trigger=lambda x: x == 0,
        name_prefix=f"env_index_{index}",
        video_length=video_length
    )
    
    obs = vec_env.reset()
    
    for _ in range(video_length + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = vec_env.step(action)
    
    vec_env.close()

def evaluate_and_record_experiments(
    root_dir: str,
    n_eval_episodes: int = 10,
    output_dir: str = "evaluation_results",
    output_prefix: str = "model_evaluation",
    record_videos: bool = True,
    video_length: int = None,
    episodes: int = 3
):
    """
    完整的實驗評估和錄影流程
    
    Args:
        root_dir: 實驗根目錄
        model_class: PPO 模型類別 (PerturbationPPO, RandomNoisePPO 等)
        n_eval_episodes: 每個索引評估的 episode 數量
        output_dir: CSV 輸出目錄
        output_prefix: 輸出文件前綴
        record_videos: 是否錄製影片
        video_length: 影片長度
        episodes: episodes 數量
    """
    # 1. 掃描實驗目錄
    print("="*80)
    print("步驟 1: 掃描實驗目錄")
    print("="*80)
    algorithm_models = scan_experiment_root(root_dir)
    
    if not algorithm_models:
        print("⚠ 錯誤: 沒有找到任何模型！")
        return
    
    # 2. 評估所有模型
    print("\n" + "="*80)
    print("步驟 2: 評估模型性能")
    print("="*80)
    mean_df, std_df = evaluate_algorithms_on_all_indices(
        algorithm_models=algorithm_models,
        n_eval_episodes=n_eval_episodes,
        output_dir=output_dir,
        output_prefix=output_prefix
    )
    
    # 3. 錄製影片（可選）
    if record_videos:
        print("\n" + "="*80)
        print("步驟 3: 錄製示範影片")
        print("="*80)
        record_all_models(
            algorithm_models=algorithm_models,
            video_length=video_length,
            episodes=episodes
        )
    
    print("\n" + "="*80)
    print("✓ 所有任務完成！")
    print("="*80)
    
    return mean_df, std_df

def paser_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root_dir", help="evaluation root dir", type=str, default ="multiagent/experiment_results")
    parser.add_argument("-s", "--save_dir", help="evaluation save dir", type=str, default = "evaluation_results/") #parser can't pass bool
    parser.add_argument("--record", help="record videos", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":
    args = paser_argument()
    
    # 執行完整評估和錄影
    mean_df, std_df = evaluate_and_record_experiments(
        root_dir=args.root_dir,
        n_eval_episodes=10,
        output_dir=args.save_dir,
        output_prefix="algorithm_comparison",
        record_videos=args.record,
        video_length=None,  # 自動根據環境設定
        episodes=3
    )