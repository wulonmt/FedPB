import Env
from utils.init_pos_config import is_valid_env, is_costum_env, get_init_pos, get_available_envs
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from pathlib import Path
import os

def get_train_norm_env(env_name: str, index: int = 0, n_cpu: int = 1):
    """
    取得使用VecNormalize的環境
    
    :param env_name: 環境名稱
    :type env_name: str
    :param index: 初始位置索引
    :type index: int
    :param n_cpu: CPU數量
    :type n_cpu: int
    """
    assert is_valid_env(env_name), f"Only environments {', '.join(get_available_envs())} are available"
    if not is_costum_env(env_name):
        env = make_vec_env(env_name, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    else:
        env = make_vec_env(env_name, 
                           n_envs=n_cpu, 
                           vec_env_cls=SubprocVecEnv, 
                           env_kwargs = get_init_pos(env_name, index)
                           )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    return env

def save_train_norm_env(env, path: str):
    assert type(env).__name__ == "VecNormalize", "env must be VecNormalize"
    env.save(Path(path) / Path("vec_normalize.pkl"))

def get_eval_norm_env(env_name: str, 
                      path: str, 
                      render_mode: str = "rgb_array", 
                      index: int = 0, 
                      n_cpu: int = 1, 
                      wrapper_class = None
                      ):
    """
    取得評估用的環境，載入vec_normalize.pkl
    
    :param env_name: 環境名稱
    :type env_name: str
    :param path: vec_normalize.pkl的資料夾路徑
    :type path: str
    :param render_mode: "human" or "rgb_array"
    :type render_mode: str
    :param index: 初始位置索引
    :type index: int
    :param n_cpu: CPU數量
    :type n_cpu: int
    """
    env_kwargs = get_init_pos(env_name, index)
    env_kwargs["render_mode"] = render_mode
    env = make_vec_env(env_name, 
                       n_envs=n_cpu, 
                       vec_env_cls=DummyVecEnv, 
                       env_kwargs=env_kwargs, 
                       wrapper_class=wrapper_class
                       )
    
    norm_path = Path(path) / Path("vec_normalize.pkl")
    assert os.path.isfile(norm_path), f"VecNormalize file not found: {norm_path}"
    env = VecNormalize.load(norm_path, env)
    env.training = False
    env.norm_reward = False
    return env