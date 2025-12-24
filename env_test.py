import gymnasium as gym
import torch as th
from stable_baselines3 import SAC, PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import argparse
from datetime import datetime
import Env
from utils.init_pos_config import get_init_pos, is_valid_env, get_available_envs, is_costum_env
from utils.CustomSAC import CustomSAC
from record import RewardDisplayWrapper
import cv2  # show the RewrdDisplayWrapper render

from utils.PerturbationPPO import PerturbationPPO
from utils.RandomNoisePPO import RandomNoisePPO

MODEL_LIST = ["PBPPO", "PPO", "RNPPO"]
MODEL_STRUCTURE = [256, 256]

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", help="The name of the Gymnasium environment to use.", type=str, default=None)
    parser.add_argument("-m", "--model", help="Model for testing", type=str, default="PBPPO")
    parser.add_argument("-i", "--index", help="Environment index", type=int, default=0)
    parser.add_argument("-r", "--regular", help="using regular in compute kl divergence", action="store_true")
    parser.add_argument("-t", "--time_steps", help="Total Timesteps", type=int, default=2e6)
    return parser.parse_args()


def train(env_name_arg=None, model = "PBPPO", index = 0, using_regular = False, timesteps = 2e6):
    n_cpu = 2
    batch_size = 64
    # You can hardcode the environment name here for testing.
    # If `env_name` is None, the script will use the command-line argument.
    env_name = None
    # env_name = "PendulumFixPos-v0"
    # env_name = "PendulumFixPos-v1"
    # env_name = "MountainCarFixPos-v0"
    # env_name = "MountainCarFixPos-v1"
    # env_name = "CartPoleSwingUpFixInitState-v1"
    # env_name = "CartPoleSwingUpFixInitState-v2"
    # env_name = "CartPoleSwingUpActionScale-v1"
    # env_name = "HopperFixLength-v0"
    # env_name = "HalfCheetahFixLength-v0"
    # env_name = "CrowdedHighway-v0"
    # env_name = "CrowdedHighway-v1"
    # env_name = "CarRacingFixSeed-v0"

    if env_name is None:
        env_name = env_name_arg

    if env_name is None:
        raise ValueError("Environment name must be provided either in the train() function or via command-line argument.")

    assert is_valid_env(env_name), f"Only environments {', '.join(get_available_envs())} are available"
    if not is_costum_env(env_name):
        trained_env = make_vec_env(env_name, n_envs=n_cpu, vec_env_cls=SubprocVecEnv, seed = 1)
        eval_env = gym.make(env_name)
    else:
        trained_env = make_vec_env(env_name, n_envs=n_cpu, vec_env_cls=SubprocVecEnv, seed = 1, env_kwargs = get_init_pos(env_name, index))
        eval_env = gym.make(env_name, **get_init_pos(env_name, index))

    # early stopping setting
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=30, verbose=1)
    eval_callback = EvalCallback(eval_env, eval_freq=10000, callback_after_eval=stop_train_callback, verbose=1)

    tensorboard_log = f"./env_test/{env_name}_id{index}_"
    
    time_str = datetime.now().strftime("%Y%m%d%H%M")

    MODEL = args.model
    assert MODEL in MODEL_LIST, f"Only models {', '.join(MODEL_LIST)} are available"
    tb_log_name = tensorboard_log + time_str + "_" + MODEL

    if MODEL == "PBPPO":
        tb_log_name += f"_{using_regular}"

    policy_kwargs=dict(net_arch=dict(pi=MODEL_STRUCTURE, vf=MODEL_STRUCTURE))

    if env_name == "HalfCheetahFixLength-v0":
        policy_kwargs=dict(activation_fn=nn.Tanh, net_arch=dict(pi=MODEL_STRUCTURE, vf=MODEL_STRUCTURE))
        trained_env = VecNormalize(trained_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    if MODEL == "PPO":
        model = PPO("MlpPolicy",
                    trained_env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=tb_log_name,
                    device = "cpu",
                    batch_size=batch_size,
                    )
        
    elif MODEL == "PBPPO":
        model = PerturbationPPO("MlpPolicy",
                    trained_env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=tb_log_name,
                    device = "cpu",
                    kld_use_regul = using_regular,
                    perturb_hidden_dims = MODEL_STRUCTURE,
                    batch_size=batch_size,
                    )
        
    elif MODEL == "RNPPO":
        model = RandomNoisePPO("MlpPolicy",
                    trained_env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=tb_log_name,
                    device = "cpu",
                    batch_size=batch_size,
                    )

    print(model.policy)
    # breakpoint()
    # Train the agent
    model.learn(total_timesteps=int(timesteps), tb_log_name=time_str)
    # model.learn(total_timesteps=int(3e5), tb_log_name=time_str, callback = eval_callback)
    print("log name: ", tb_log_name)
    model.save(tb_log_name + "/model")

    if env_name == "HalfCheetahFixLength-v0":
        trained_env.save(tb_log_name + "/vec_normalize.pkl")

    ############ evaluation ################

    if not is_costum_env(env_name):
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name, render_mode="human", **get_init_pos(env_name, index))

    if env_name == "HalfCheetahFixLength-v0":
        env_kwargs = get_init_pos(env_name, index)
        env_kwargs["render_mode"] = "human"
        env = make_vec_env(env_name, n_envs=1, env_kwargs=env_kwargs)
        env = VecNormalize.load(tb_log_name + "/vec_normalize.pkl", env)
        env.training = False
        env.norm_reward = False

    while True:
        obs, info = env.reset()
        done = truncated = False
        counter = 0
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            counter += 1
            if counter > 1000:
                break

fps = 30
delay = int(1000 / fps)
def show_reward_frame(window_name, img):
    if img is not None:
        # OpenCV 使用 BGR 格式，而 gymnasium 通常返回 RGB
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, img_bgr)
        
        # 等待 1ms，讓畫面更新，ESC 鍵退出
        key = cv2.waitKey(delay) & 0xFF

def eval(env_name_arg=None): 
    path = "env_test\\HalfCheetahFixLength-v0_id1_202512240250_PPO"
    model = PPO.load(path + "/model")
    # model = PerturbationPPO.load(path + "model")
    env_name = "HalfCheetahFixLength-v0"
    
    if env_name is None:
        env_name = env_name_arg

    if env_name is None:
        raise ValueError("Environment name must be provided either in the train() function or via command-line argument.")

    assert is_valid_env(env_name), f"Only environments {', '.join(get_available_envs(env_name))} are available"
    index = 0
    # env = gym.make(env_name, render_mode="human")
    # env = gym.make(env_name, render_mode="human", **get_init_pos(env_name, index))

    #Show Reward at window
    # env = gym.make(env_name, render_mode="rgb_array", **get_init_pos(env_name, index))
    if not is_costum_env(env_name):
        env = gym.make(env_name, render_mode="rgb_array")
    else:
        env = gym.make(env_name, render_mode="rgb_array", **get_init_pos(env_name, index))

    env = RewardDisplayWrapper(env)
    

    if env_name == "HalfCheetahFixLength-v0":
        env_kwargs = get_init_pos(env_name, index)
        env_kwargs["render_mode"] = "human"
        env = make_vec_env(env_name, n_envs=1, env_kwargs=env_kwargs)
        env = VecNormalize.load(path + "/vec_normalize.pkl", env)
        env.training = False
        env.norm_reward = False

    while True:
        if env_name == "HalfCheetahFixLength-v0":
            obs = env.reset()
        else:
            obs, info = env.reset()
        done = truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs)
            if env_name == "HalfCheetahFixLength-v0":
                obs, reward, done, info = env.step(action)
            else:
                obs, reward, done, truncated, info = env.step(action)
            # env.render()
            img = env.render()
            show_reward_frame(env_name, img)
            if done or truncated:
                print("done")
                cv2.waitKey(1000)  # 暫停 1 秒

if __name__ == "__main__":
    args = parse_arguments()
    # train(env_name_arg=args.environment, model=args.model, index=args.index, using_regular=args.regular, timesteps=args.time_steps)
    eval(env_name_arg=args.environment)