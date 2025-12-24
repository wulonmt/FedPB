from typing import List, Tuple, Dict

import flwr as fl
from flwr.common import Metrics, FitIns
from flwr.common.typing import Parameters
from flwr.server.client_manager import ClientManager
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import GetParametersIns
from flwr.server.utils.tensorboard import tensorboard
from flwr.server.strategy import FedAvg, FedAdam
import argparse

import requests
import sys
import numpy as np
from utils.Ptime import Ptime
import os
from utils.dc_webhook import send_discord_webhook

ENV_CONFIG_PATH="env_config.json"

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", help="local port", type=str, default="8080")
parser.add_argument("-r", "--rounds", help="total rounds", type=int, default=300)
parser.add_argument("-c", "--clients", help="number of clients", type=int, default=2)
parser.add_argument("--log_dir", help="server & client log dir", type=str, default = None)
parser.add_argument("-e", "--environment", help="which my- env been used", type=str, required=True)
args = parser.parse_args()

def save_model(parameters: Parameters, save_dir: str, filename: str = "final_model.npz"):
    """Save the model parameters to a file in the specified directory."""
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the full file path
    file_path = os.path.join(save_dir, filename)
    
    # Convert parameters to numpy arrays
    ndarrays = parameters_to_ndarrays(parameters)
    
    # Save the numpy arrays to a file
    np.savez(file_path, *ndarrays)
    print(f"Model saved to {file_path}")

class SaveModelStrategy(FedAvg):
    def __init__(self, save_dir: str, num_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir
        self.num_rounds = num_rounds
        self.aggregated_log_std = 0.0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Parameters, dict]:
        # Aggregate parameters and metrics using the super class method
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Save the model after the final round
        if server_round == self.num_rounds:
            save_model(parameters, self.save_dir, f"final_model_round_{server_round}.npz")

        return parameters, metrics

def main():
    total_rounds = args.rounds
    clients = args.clients
    time_str = Ptime()
    time_str.set_time_now()
    save_dir = "multiagent/" + time_str.get_time_to_minute()
    save_dir = args.log_dir if args.log_dir else save_dir
    
    if os.path.exists(ENV_CONFIG_PATH):
        from utils.env_config_loader import get_config_loader
    
        config_loader = get_config_loader(ENV_CONFIG_PATH)
        env_config = config_loader.get_config(args.environment)
        
        # 使用配置文件的參數（如果沒有手動指定的話）
        total_rounds = env_config['global_rounds']
        
        print(f"\n{'='*60}")
        print(f"[Server Configuration for {args.environment}")
        print(f"{'='*60}")
        print(f"  Total rounds: {total_rounds}")
        print(f"{'='*60}\n")

    print(f"Starting Server, total rounds {total_rounds}, clients {clients}")

    # Decorated strategy
    strategy = SaveModelStrategy(
        save_dir=save_dir,
        num_rounds=total_rounds,
        min_fit_clients=clients,
        min_evaluate_clients=clients,
        min_available_clients=clients,
    )

    send_discord_webhook(f"Starting Lab: {save_dir}")
    # Start Flower server
    flwr_server = fl.server
    flwr_server.start_server(
        server_address="127.0.0.1:" + args.port,
        config=fl.server.ServerConfig(num_rounds=total_rounds),
        strategy=strategy,
    )
    send_discord_webhook('Experiment done !!!!!')
    sys.exit()
    

if __name__ == "__main__":
    main()
    
