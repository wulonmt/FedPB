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
import csv
import pandas as pd

from collections import OrderedDict
import torch as th
import os
from typing import Dict, List, Optional, Tuple
from glob import glob
from pathlib import Path

from utils.init_pos_config import is_valid_env, is_costum_env
from utils.PerturbationPPO import PerturbationPPO

MODEL_DICT = {
    "PPO": PPO,
    "RNPPO": PPO,
    "PBPPO_regul0": PerturbationPPO,
    "PBPPO_regul1": PerturbationPPO,
}

class RewardDisplayWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cumulative_reward = 0
        self.current_reward = 0
        
    def reset(self, **kwargs):
        self.cumulative_reward = 0
        self.current_reward = 0
        return super().reset(**kwargs)
        
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        self.current_reward = reward
        self.cumulative_reward += reward
            
        return obs, reward, done, truncated, info
        
    def render(self):
        img = super().render()
        
        img = np.ascontiguousarray(img, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 準備文字
        text_curr = f'Current Reward: {self.current_reward:.2f}'
        text_cum = f'Accumulated Reward: {self.cumulative_reward:.2f}'
        
        # 獲取文字大小
        (curr_width, curr_height), _ = cv2.getTextSize(text_curr, font, 0.5, 1)
        (cum_width, cum_height), _ = cv2.getTextSize(text_cum, font, 0.5, 1)
        
        # 使用最大寬度
        max_width = max(curr_width, cum_width)
        total_height = curr_height + cum_height + 10
        
        # 添加背景
        overlay = img.copy()
        cv2.rectangle(
            overlay, 
            (5, 5), 
            (10 + max_width, 15 + total_height), 
            (0, 0, 0), 
            -1
        )
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # 添加文字
        cv2.putText(
            img, 
            text_curr, 
            (10, 20), 
            font, 
            0.5, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
        
        cv2.putText(
            img, 
            text_cum, 
            (10, 20 + curr_height + 5),
            font, 
            0.5, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
        
        return img

def make_wrapped_env(env_name, i=0):
    def _init():
        env = gym.make(env_name, render_mode="rgb_array", **get_init_pos(env_name, i))
        return RewardDisplayWrapper(env)
    return _init

def make_eval_env(env_name: str, index: int):
    """
    創建評估用的環境
    
    Args:
        env_name: 環境名稱
        index: 環境索引
    
    Returns:
        gym.Env: 評估環境
    """
    if not is_costum_env(env_name):
        env = gym.make(env_name)
    else:
        env = gym.make(env_name, **get_init_pos(env_name, index))
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

def load_npz_from_folders(folder_list_path: str, base_path: Optional[str] = None) -> Dict[str, Tuple[str, str]]:
    """
    從文本文件中讀取資料夾列表，並加載每個資料夾中的第一個 .npz 文件
    
    Args:
        folder_list_path (str): 包含資料夾名稱列表的文本文件路徑
        base_path (str, optional): 基礎路徑，如果提供，資料夾路徑會相對於這個基礎路徑
    
    Returns:
        Dict[str, Tuple[str, str]]: 以資料夾名為鍵，(環境名, npz路徑) 元組為值的字典
    """
    # 檢查文本文件是否存在
    if not os.path.exists(folder_list_path):
        raise FileNotFoundError(f"找不到文件列表：{folder_list_path}")
    
    # 讀取資料夾列表
    with open(folder_list_path, 'r') as f:
        folders = [line.strip().strip("\"") for line in f if line.strip()]
    
    # 存儲加載的 npz 文件
    loaded_files = {}
    failed_folders = []
    
    # 處理每個資料夾
    for folder in folders:
        # 構建完整路徑
        if base_path:
            full_folder_path = os.path.join(base_path, folder)
        else:
            full_folder_path = folder
            
        try:
            if not os.path.exists(full_folder_path):
                raise FileNotFoundError(f"資料夾不存在：{full_folder_path}")
            
            # 使用 glob 找到資料夾中的所有 .npz 文件
            npz_files = glob(os.path.join(full_folder_path, "*.npz"))
            
            if not npz_files:
                raise FileNotFoundError(f"在資料夾中找不到 .npz 文件：{full_folder_path}")
            
            if len(npz_files) > 1:
                print(f"警告：{folder} 包含多個 .npz 文件，將使用：{os.path.basename(npz_files[0])}")
            
            # 加載第一個找到的 npz 文件
            npz_path = npz_files[0]
            environment = folder.split("_")[-3]
            assert is_valid_env(environment), f"Invalid environment name: {environment}"
            loaded_files[folder] = (environment, npz_path)
            
        except Exception as e:
            print(f"加載失敗 {folder}: {str(e)}")
            failed_folders.append((folder, str(e)))
            continue
    
    # 打印摘要
    print("\n加載摘要:")
    print(f"成功加載: {len(loaded_files)}/{len(folders)} 個文件")
    if failed_folders:
        print("\n加載失敗的資料夾:")
        for folder, error in failed_folders:
            print(f"- {folder}: {error}")
    
    return loaded_files

def evaluate_models_on_all_indices(
    npz_files: Dict[str, Tuple[str, str]],
    n_eval_episodes: int = 10,
    output_dir: str = ".",
    output_prefix: str = "evaluation"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    評估所有模型在不同環境索引上的表現，並輸出 CSV 表格
    
    Args:
        npz_files: 字典，格式為 {folder_name: (env_name, npz_path)}
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
    print("開始評估模型...")
    print("="*80)
    
    # 遍歷所有模型
    for folder_name, (env_name, npz_path) in npz_files.items():
        print(f"\n處理模型: {folder_name}")
        print(f"環境: {env_name}")
        print(f"模型路徑: {npz_path}")
        
        model_means = {}
        model_stds = {}
        
        # 獲取環境的索引列表
        try:
            init_list = get_init_list(env_name)
            n_indices = len(init_list)
        except:
            # 如果獲取失敗，默認使用 0-4
            n_indices = 5
            print(f"警告: 無法獲取 {env_name} 的初始化列表，使用默認索引 0-4")
        
        # 評估每個索引
        for index in range(n_indices):
            print(f"  評估索引 {index}...", end=" ")
            try:
                # 創建評估環境
                eval_env = make_eval_env(env_name, index)
                
                # 加載模型
                model = load_params_to_model(eval_env, npz_path, verbose=False)
                
                # 評估模型
                mean_reward, std_reward = evaluate_policy(
                    model, 
                    eval_env, 
                    n_eval_episodes=n_eval_episodes,
                    deterministic=True
                )
                
                model_means[f"Index_{index}"] = mean_reward
                model_stds[f"Index_{index}"] = std_reward
                
                print(f"Mean: {mean_reward:.2f}, Std: {std_reward:.2f}")
                
                # 關閉環境
                eval_env.close()
                
            except Exception as e:
                print(f"失敗: {str(e)}")
                model_means[f"Index_{index}"] = np.nan
                model_stds[f"Index_{index}"] = np.nan
        
        # 儲存該模型的結果
        results_mean[folder_name] = model_means
        results_std[folder_name] = model_stds
    
    # 創建 DataFrame
    mean_df = pd.DataFrame.from_dict(results_mean, orient='index')
    std_df = pd.DataFrame.from_dict(results_std, orient='index')
    
    # 確保列的順序
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
    print("\n=== Mean Rewards ===")
    print(mean_df.to_string())
    print("\n=== Std Rewards ===")
    print(std_df.to_string())
    
    return mean_df, std_df

def record(env_name, model_path, save_dir, video_length = None, episodes = 3):
    """
    錄製模型在環境中的表現
    """
    _env = make_eval_env(env_name, 0)
    model = load_params_to_model(_env, model_path)
    
    for i in range(len(get_init_list(env_name))):
        vec_env = DummyVecEnv([make_wrapped_env(env_name, i=i)])
        
        if video_length is None:
            max_steps = vec_env.envs[0].spec.max_episode_steps
            video_length = max_steps * episodes
        
        vec_env = VecVideoRecorder(vec_env, save_dir + "\\videos",
                    record_video_trigger=lambda x: x == 0,
                    name_prefix=f"env_index_{i}",
                    video_length=video_length,)
        
        obs = vec_env.reset()
        
        for _ in range(video_length + 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, _, _, _ = vec_env.step(action)
        vec_env.close()

def evaluate_multi_dirs(
    folder_list_path: str = "record_dirs.txt",
    n_eval_episodes: int = 10,
    output_dir: str = "evaluation_results",
    output_prefix: str = "model_evaluation"
):
    """
    評估多個目錄中的模型並輸出 CSV
    
    Args:
        folder_list_path: 包含資料夾列表的文本文件路徑
        n_eval_episodes: 每個環境索引評估的 episode 數量
        output_dir: 輸出目錄
        output_prefix: 輸出文件前綴
    """
    # 加載所有 npz 文件
    npz_files = load_npz_from_folders(folder_list_path=folder_list_path)
    
    # 評估所有模型
    mean_df, std_df = evaluate_models_on_all_indices(
        npz_files=npz_files,
        n_eval_episodes=n_eval_episodes,
        output_dir=output_dir,
        output_prefix=output_prefix
    )
    
    return mean_df, std_df

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
                    eval_env = make_eval_env(env_name, index)
                    
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
                env = make_eval_env(env_name, 0)
                model = load_params_to_model(env, npz_path, MODEL_DICT[alg_name], verbose=False)
                env.close()
                
                # 錄製所有索引的影片
                init_list = get_init_list(env_name)
                
                for i in range(len(init_list)):
                    record_single_model(
                        env_name=env_name,
                        index=i,
                        model=model,
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
    save_dir: str,
    video_length: int = None,
    episodes: int = 3
):
    """
    錄製單一模型在特定索引的影片
    """
    def make_wrapped_env(env_name, i=0):
        def _init():
            env = gym.make(env_name, render_mode="rgb_array", **get_init_pos(env_name, i))
            return RewardDisplayWrapper(env)
        return _init
    
    vec_env = DummyVecEnv([make_wrapped_env(env_name, index)])
    
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


if __name__ == "__main__":
    import sys
    
    # 範例使用
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "multiagent/experiment_results"  # 預設路徑

    default_output_dir = "evaluation_results/" + root_dir.split("/")[-1]
    # 執行完整評估和錄影
    mean_df, std_df = evaluate_and_record_experiments(
        root_dir=root_dir,
        n_eval_episodes=10,
        output_dir=default_output_dir,
        output_prefix="algorithm_comparison",
        record_videos=True,
        video_length=None,  # 自動根據環境設定
        episodes=3
    )