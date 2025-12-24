import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Type
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from gymnasium import spaces


class RandomNoisePPO(PPO):
    """
    PPO with Random Noise Perturbation (對照組).
    在訓練時對 state 添加隨機噪音，計算原始 state 和噪音 state 的 policy gradient.
    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        # Random noise parameters
        noise_scale: float = 0.15,  # 噪音的標準差
        smooth_factor: float = 0.1,  # smoothness loss 的權重
        delay_noise_train: int = 500,  # 延遲開始使用噪音的 update 數
        **kwargs
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            **kwargs
        )
        
        # Random noise parameters
        self.noise_scale = noise_scale
        self.smooth_factor = smooth_factor
        self.delay_noise_train = delay_noise_train
    
    def add_random_noise(self, obs: th.Tensor) -> th.Tensor:
        """
        為觀測添加隨機高斯噪音.
        
        Args:
            obs: 原始觀測 [batch_size, obs_dim]
        
        Returns:
            noisy_obs: 添加噪音後的觀測
        """
        noise = th.randn_like(obs) * self.noise_scale
        noisy_obs = obs + noise
        return noisy_obs
    
    def train(self) -> None:
        """
        Update policy using the current rollout buffer.
        Modified to include random noise perturbation.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        
        # Update optimizer learning rates
        self._update_learning_rate(self.policy.optimizer)
        
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        
        # Lists to track losses
        pg_losses, value_losses, smooth_losses = [], [], []
        noise_norms = []
        clip_fractions = []
        
        continue_training = True
        
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            # Sample from rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                
                # Flatten observations if needed
                if isinstance(rollout_data.observations, dict):
                    obs = rollout_data.observations['observation']
                else:
                    obs = rollout_data.observations
                
                # === Add Random Noise ===
                noisy_obs = self.add_random_noise(obs)
                noise = noisy_obs - obs
                
                # === Policy Loss (Standard PPO) ===
                values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
                values = values.flatten()
                
                # Normalize advantages
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Ratio for clipping
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                
                # Clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                
                # Clip fraction for logging
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                
                # === Value Loss ===
                value_loss = F.mse_loss(rollout_data.returns, values)
                
                # Calculate approximate form of reverse KL Divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                
                # === Training Logic ===
                if self._n_updates < self.delay_noise_train:
                    # 延遲階段：只訓練標準 PPO
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    
                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    
                else:
                    # === Smoothness Loss (L_smooth) with Random Noise ===
                    # Evaluate policy on noisy observations
                    _, log_prob_noisy, _ = self.policy.evaluate_actions(
                        noisy_obs.detach(),  # Don't backprop through noise
                        actions
                    )
                    
                    # Importance sampling ratio: π_θ(a|s+noise) / π_old(a|s)
                    ratio_smooth = th.exp(log_prob_noisy - rollout_data.old_log_prob)
                    
                    # Clipped surrogate loss for smoothness
                    smooth_loss_1 = advantages * ratio_smooth
                    smooth_loss_2 = advantages * th.clamp(ratio_smooth, 1 - clip_range, 1 + clip_range)
                    smooth_loss = -th.min(smooth_loss_1, smooth_loss_2).mean()
                    
                    # === Combined Policy Update ===
                    total_policy_loss = (policy_loss + 
                                       self.smooth_factor * smooth_loss + 
                                       self.ent_coef * entropy_loss + 
                                       self.vf_coef * value_loss)
                    
                    # Optimize policy
                    self.policy.optimizer.zero_grad()
                    total_policy_loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    
                    # Logging for noise training
                    smooth_losses.append(smooth_loss.item())
                    noise_norm = th.norm(noise, dim=-1).mean().item()
                    noise_norms.append(noise_norm)
                
                # === Logging ===
                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
            
            self._n_updates += 1
            if not continue_training:
                break
        
        # Logging
        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        
        if len(smooth_losses) == 0:
            self.logger.record("train/smooth_loss", 0)
            self.logger.record("train/noise_norm", 0)
            self.logger.record("train/noise_scale", self.noise_scale)
        else:
            self.logger.record("train/smooth_loss", np.mean(smooth_losses))
            self.logger.record("train/noise_norm", np.mean(noise_norms))
            self.logger.record("train/noise_scale", self.noise_scale)
        
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/n_updates", self._n_updates)
        
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
