import gymnasium as gym
from stable_baselines3 import PPO

import argparse
from datetime import datetime
import Env
from utils.init_pos_config import get_init_pos, is_valid_env, get_available_envs, is_costum_env
from record import RewardDisplayWrapper
import cv2  # show the RewrdDisplayWrapper render
from pathlib import Path

from utils.PerturbationPPO import PerturbationPPO
from utils.RandomNoisePPO import RandomNoisePPO

from normalized_env import get_train_norm_env, save_train_norm_env, get_eval_norm_env

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
    # eval only
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("-p", "--path", help="directory of eval model", type=str, default=None)
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

    training_env = get_train_norm_env(env_name, index, n_cpu)

    tensorboard_log = f"./env_test/{env_name}_id{index}_"
    
    time_str = datetime.now().strftime("%Y%m%d%H%M")

    MODEL = args.model
    assert MODEL in MODEL_LIST, f"Only models {', '.join(MODEL_LIST)} are available"
    tb_log_name = tensorboard_log + time_str + "_" + MODEL

    if MODEL == "PBPPO":
        tb_log_name += f"_{using_regular}"

    policy_kwargs=dict(net_arch=dict(pi=MODEL_STRUCTURE, vf=MODEL_STRUCTURE))

    if MODEL == "PPO":
        model = PPO("MlpPolicy",
                    training_env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=tb_log_name,
                    device = "cpu",
                    batch_size=batch_size,
                    )
        
    elif MODEL == "PBPPO":
        model = PerturbationPPO("MlpPolicy",
                    training_env,
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
                    training_env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=tb_log_name,
                    device = "cpu",
                    batch_size=batch_size,
                    )

    print(model.policy)

    # Train the agent
    model.learn(total_timesteps=int(timesteps), tb_log_name=time_str)
    # model.learn(total_timesteps=int(3e5), tb_log_name=time_str, callback = eval_callback)
    print("log name: ", tb_log_name)
    model.save(tb_log_name + "/model")
    save_train_norm_env(training_env, tb_log_name)

    ############ evaluation ################

    env = get_eval_norm_env(env_name, tb_log_name, index=index, render_mode="human")

    while True:
        obs, info = env.reset()
        done = False
        counter = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
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

def eval(
        env_name_arg=None,
        path=None,
        model_name=None,
        index=0
        ): 
    # path = "env_test\\CartPoleSwingUpV1WithAdjustablePole-v0_id0_202512250353_PPO"
    # model = PPO.load(path + "/model")
    # model = PerturbationPPO.load(path + "model")
    # env_name = "CartPoleSwingUpV1WithAdjustablePole-v0"
    env_name = None
    
    if env_name is None:
        env_name = env_name_arg

    if env_name is None:
        raise ValueError("Environment name must be provided either in the train() function or via command-line argument.")

    if model_name is None:
        model_name = "PPO"

    model_path = Path(path) / Path("model")

    if model_name == "PBPPO":
        model = PerturbationPPO.load(model_path)
    elif model_name == "PPO":
        model = PPO.load(model_path)
    elif model_name == "RNPPO":
        model = RandomNoisePPO.load(model_path)
    else:
        raise ValueError("Model name must be 'PBPPO', 'PPO', or 'RNPPO'")

    #Show Reward at window
    env = get_eval_norm_env(
        env_name, 
        path, 
        render_mode="human", 
        index=index, 
        wrapper_class=RewardDisplayWrapper
        )

    while True:
        obs = env.reset()
        done = False

        while not (done):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # env.render()
            img = env.render()
            show_reward_frame(env_name, img)
            if done:
                print("done")
                cv2.waitKey(1000)  # 暫停 1 秒

if __name__ == "__main__":
    args = parse_arguments()
    if args.eval:
        eval(
            path=args.path, 
            model_name=args.model, 
            index=args.index, 
            env_name_arg=args.environment
            )
    else:
        train(
            env_name_arg=args.environment, 
            model=args.model, 
            index=args.index, 
            using_regular=args.regular, 
            timesteps=args.time_steps
            )