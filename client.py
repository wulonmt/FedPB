import gymnasium as gym
import torch as th
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import argparse
from utils.Ptime import Ptime

import flwr as fl
from collections import OrderedDict
import os
import sys
import Env
from stable_baselines3 import PPO
from utils.PerturbationPPO import PerturbationPPO
from utils.RandomNoisePPO import RandomNoisePPO

from utils.init_pos_config import get_init_pos, assert_alarm, get_init_list
from normalized_env import get_train_norm_env, save_train_norm_env, get_eval_norm_env

MODEL_LIST = ["PBPPO", "PPO", "RNPPO"]
ENV_CONFIG_PATH="env_config.json"

def paser_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_name", help="modified log name", type=str, default ="auto")
    parser.add_argument("-s", "--save_log", help="whether save log or not", type=str, default = "True") #parser can't pass bool
    parser.add_argument("-e", "--environment", help="which my- env been used", type=str, required=True)
    parser.add_argument("-i", "--index", help="client index", type=int, default = 0, required = True)
    parser.add_argument("-p", "--port", help="local port", type=str, default="8080")
    parser.add_argument("-m", "--time_step", help="training time steps", type=int, default=4096)
    parser.add_argument("--log_dir", help="server & client log dir", type=str, default = None)
    parser.add_argument("--n_cpu", help="number of cpu", type=int, default = 1)
    parser.add_argument("--kld_use_regul", help="using regular policy in compute kl divergence", action="store_true")
    parser.add_argument("--model", help="training model", type=str, required=True)
    parser.add_argument("--perturb_scale", help="perturbation scale & random noise scalse", type=float, default = 0.15)
    parser.add_argument("--network_dim", help="network dimension, enter X for [X, X]", type=int, default = 64)


    return parser.parse_args()


class PerturbationPPOClient(fl.client.NumPyClient):
    def __init__(self,
                client_index,
                environment,
                log_dir,
                time_step,
                log_name="auto",
                save_log="True",
                n_cpu=1,
                # Perturbation setting
                kld_use_regul=False,
                MODEL = "PBPPO",
                perturb_scale = 0.15,
                network_dim = 64,
                ):
        batch_size = 64
        n_steps = 512
        self.time_step = time_step

        if os.path.exists(ENV_CONFIG_PATH):
            from utils.env_config_loader import get_config_loader
        
            config_loader = get_config_loader(ENV_CONFIG_PATH)
            env_config = config_loader.get_config(environment)
            
            # 使用配置文件的參數（如果沒有手動指定的話）
            network_dim = env_config['network_dim']
            perturb_scale = env_config['perturbation_scale']
            self.time_step = env_config['local_timesteps']
            
            print(f"\n{'='*60}")
            print(f"[Client {client_index}] Configuration for {environment}")
            print(f"{'='*60}")
            print(f"  Network Dimension: {network_dim}")
            print(f"  Perturbation Scale: {perturb_scale}")
            print(f"  Local Timesteps: {self.time_step}")
            print(f"{'='*60}\n")
        
        init_length = len(get_init_list(environment))
        env_index = client_index if client_index < init_length else np.random.randint(0, init_length)
        self.env = get_train_norm_env(environment, env_index, n_cpu)
        
        self.tensorboard_log = f"{environment}/" if save_log == "True" else None
        time_str = Ptime()
        time_str.set_time_now()
        assert MODEL in MODEL_LIST, f"Only models {', '.join(MODEL_LIST)} are available"
        self.model_name = MODEL
        if save_log == "True":
            self.tensorboard_log = f"multiagent/{time_str.get_time_to_minute()}_{environment}_{MODEL}/{self.tensorboard_log}"
        self.tensorboard_log = log_dir + f"/{environment}/" if log_dir else self.tensorboard_log
        
        trained_env = self.env
        if MODEL == "PPO":
            self.model = PPO("MlpPolicy",
                        trained_env,
                        verbose=1,
                        tensorboard_log=self.tensorboard_log,
                        device="cpu",
                        n_steps=n_steps,
                        policy_kwargs=dict(net_arch=dict(pi=[network_dim, network_dim], vf=[network_dim, network_dim])),
                        )
        elif MODEL == "PBPPO":
            self.model = PerturbationPPO("MlpPolicy",
                        trained_env,
                        verbose=1,
                        tensorboard_log=self.tensorboard_log,
                        device="cpu",
                        n_steps=n_steps,
                        policy_kwargs=dict(net_arch=dict(pi=[network_dim, network_dim], vf=[network_dim, network_dim])),
                        delay_perturb_train=0,
                        kld_use_regul = kld_use_regul,
                        perturb_hidden_dims = [network_dim, network_dim],
                        target_perturb_scale = perturb_scale
                        )
        elif MODEL == "RNPPO":
            self.model = RandomNoisePPO("MlpPolicy",
                        trained_env,
                        n_steps=n_steps,
                        policy_kwargs=dict(net_arch=dict(pi=[network_dim, network_dim], vf=[network_dim, network_dim])),
                        verbose=1,
                        tensorboard_log=self.tensorboard_log,
                        delay_noise_train=0,
                        device = "cpu"
                        )

        self.n_round = int(0)
        
        if save_log == "True":
            description = log_name if log_name != "auto" else \
                        f"PerturbPPO"
            self.log_name = f"{client_index}_{description}"
        else:
            self.log_name = None

        self.save_log = save_log
        self.client_index = client_index
        
        
    def get_parameters(self, config):
        """
        只上傳 policy 的參數（不包含 value network 和 perturbation network）
        """
        # 需要排除的 value network 相關參數：
        # 1. mlp_extractor.value_net.* (MLP中的value分支)
        # 2. value_net.* (最終value輸出層)
        policy_params = []
        excluded_keys = []
        
        for key, value in self.model.policy.state_dict().items():
            # 排除所有 value 相關的參數
            if 'value_net' in key:
                excluded_keys.append(key)
                continue
            
            policy_params.append(value.cpu().numpy())
        
        print(f"[Client {self.client_index}] Uploading {len(policy_params)} policy parameters")
        print(f"[Client {self.client_index}] Excluded {len(excluded_keys)} value parameters: {excluded_keys}")
        return policy_params

    def set_parameters(self, parameters):
        """
        只下載並更新 policy 的參數
        """
        # 獲取當前模型的完整 state_dict
        current_state_dict = self.model.policy.state_dict()
        
        # 只更新 policy 相關的參數（排除所有包含 'value_net' 的參數）
        policy_keys = [key for key in current_state_dict.keys() if 'value_net' not in key]
        
        if len(parameters) != len(policy_keys):
            print(f"[Client {self.client_index}] Warning: Parameter count mismatch!")
            print(f"Expected {len(policy_keys)} parameters, got {len(parameters)}")
            print(f"Policy keys: {policy_keys}")
            return
        
        # 創建新的 state_dict，保留 value_net 的本地參數
        new_state_dict = OrderedDict()
        
        # 更新 policy 參數
        for key, param in zip(policy_keys, parameters):
            new_state_dict[key] = th.tensor(param)
        
        # 保留 value_net 的本地參數（包含 mlp_extractor.value_net 和 value_net）
        for key in current_state_dict.keys():
            if 'value_net' in key:
                new_state_dict[key] = current_state_dict[key]
        
        # 載入到 policy
        self.model.policy.load_state_dict(new_state_dict, strict=True)
        
        if self.model_name == "PBPPO":
            # 更新 regul_policy (全局策略)，用於 Lagrangian perturbation 的約束
            # regul_policy 只需要 policy 部分，不需要 value_net
            regul_state_dict = OrderedDict()
            for key, param in zip(policy_keys, parameters):
                regul_state_dict[key] = th.tensor(param)
            
            # 補齊 regul_policy 需要的 value_net（用當前的即可，因為不會使用）
            for key in current_state_dict.keys():
                if 'value_net' in key and key not in regul_state_dict:
                    regul_state_dict[key] = current_state_dict[key]
            
            self.model.regul_policy.load_state_dict(regul_state_dict, strict=True)
            self.model.regul_policy.set_training_mode(False)  # regul_policy 只用於評估
        
        print(f"[Client {self.client_index}] Updated {len(policy_keys)} policy parameters from server")
        print(f"[Client {self.client_index}] Retained local value_net parameters")


    def fit(self, parameters, config):
        try:
            print(f"[Client {self.client_index}] fit, config: {config}")
            self.n_round += 1
            
            if "server_round" in config.keys():
                self.n_round = config["server_round"]
                self.model.n_rounds = self.n_round
                self.model.num_timesteps = self.time_step * (self.n_round - 1)

            # 更新參數（只更新 policy 部分）
            self.set_parameters(parameters)
            
            if "learning_rate" in config.keys():
                self.model.learning_rate = config["learning_rate"]
            
            print(f"[Client {self.client_index}] Training with learning rate: {self.model.learning_rate}")
            
            # 訓練 agent
            self.model.learn(
                total_timesteps=self.time_step,
                tb_log_name=(self.log_name + f"/round_{self.n_round:0>2d}") if self.log_name is not None else None,
                reset_num_timesteps=False,
            )
            
            # 儲存模型
            if self.save_log == "True":
                save_path = self.tensorboard_log + self.log_name
                print(f"[Client {self.client_index}] Saving model to: {save_path}")
                self.model.save(save_path + "/model")
                save_train_norm_env(self.env, save_path)


            # 使用訓練步數作為權重
            merge_weight = int(max(self.model.num_timesteps, 1))
            
            print(f"[Client {self.client_index}] Merge weight: {merge_weight}")
            
            return self.get_parameters(config={}), merge_weight, {}
            
        except Exception as e:
            import traceback
            print(f"[Client {self.client_index}] Exception during fit: {e}")
            traceback.print_exc()
            return self.get_parameters({}), 0, {}

    def evaluate(self, parameters, config):
        """
        評估模型性能
        """
        print(f"[Client {self.client_index}] Evaluating model")
        self.set_parameters(parameters)

        self.env.training = False    # 停止更新 Observation 的均值和標準差
        self.env.norm_reward = False
        reward_mean, reward_std = evaluate_policy(self.model, self.env)

        self.env.training = True    # 完成驗證後恢復訓練狀態
        self.env.norm_reward = True
        
        print(f"[Client {self.client_index}] Evaluation - Mean: {reward_mean:.2f}, Std: {reward_std:.2f}")
        
        return -reward_mean, 1, {"reward_mean": reward_mean, "reward_std": reward_std}

def main():
    args = paser_argument()
    assert_alarm(args.environment)

    # Start Flower client
    #port = 8080 + args.index
    client = PerturbationPPOClient(client_index=args.index,
                          environment=args.environment,
                          save_log=args.save_log,
                          log_dir=args.log_dir,
                          log_name=args.log_name,
                          time_step=args.time_step,
                          n_cpu=args.n_cpu,
                          kld_use_regul=args.kld_use_regul,
                          MODEL=args.model,
                          perturb_scale=args.perturb_scale,
                          network_dim=args.network_dim
                          )
    fl.client.start_client(
        server_address=f"127.0.0.1:" + args.port,
        client=client.to_client(),
    )
    # sys.exit()

    # if args.index < 4:
    #     env = gym.make(args.environment,
    #                    render_mode="human",
    #                    **get_init_pos(args.environment, args.index)
    #                    )
    # else:
    #     env = gym.make(args.environment, render_mode="human", **get_init_pos(args.environment, 4))
    # # env = gym.make(args.environment, render_mode="human")

    # while True:
    #     obs, info = env.reset()
    #     done = truncated = False
    #     while not (done or truncated):
    #         action, _ = client.model.predict(obs)
    #         obs, reward, done, truncated, info = env.step(action)
    #         env.render()

if __name__ == "__main__":
    main()
    # test()

