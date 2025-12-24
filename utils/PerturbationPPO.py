import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
from typing import Optional, Type, Union, Dict, Any
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from gymnasium import spaces


class PerturbationNetwork(nn.Module):
    """
    Perturbation Network that generates state perturbations.
    Outputs mean and log_std for Gaussian perturbation.
    """
    def __init__(self, state_dim: int, 
                 hidden_dims: list = [64, 64], 
                 init_perturb_scale=0.001,
                 target_perturb_scale=0.15,
                 log_sigma_max = -1,
                 log_sigma_min = -5,
                 ):
        super().__init__()
        layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, state_dim)
        self.log_sigma_head = nn.Linear(in_dim, state_dim)

        self.scale = init_perturb_scale
        self.init_scale = init_perturb_scale
        self.target_scale = target_perturb_scale

        self.log_sigma_max = log_sigma_max
        self.log_sigma_min = log_sigma_min
        
    def forward(self, state: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Args:
            state: Input state tensor
        Returns:
            mu: Mean of perturbation
            log_sigma: Log standard deviation of perturbation
        """
        features = self.shared(state)
        mu = self.mu_head(features)
        # Force scaling
        mu = th.tanh(mu) * self.scale
        log_sigma = self.log_sigma_head(features)
        # Clamp log_sigma for numerical stability
        log_sigma = th.clamp(log_sigma, self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma
    
    def sample_perturbation(self, state: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Sample perturbation using reparameterization trick.
        Returns:
            perturbed_state: s + delta
            mu: Mean of perturbation
            log_sigma: Log std of perturbation
        """
        mu, log_sigma = self.forward(state)
        epsilon = th.randn_like(mu)
        delta = mu + th.exp(log_sigma) * epsilon
        perturbed_state = state + delta
        return perturbed_state, mu, log_sigma, delta
    
    def update_scale(self, progress: float):
        """
        progress: 0.0 ~ 1.0 的數值 (代表 curriculum 完成度)
        """
        self.scale = min(self.init_scale + self.target_scale * progress, self.target_scale)


class PerturbationPPO(PPO):
    """
    PPO with Lagrangian Perturbation for Federated Learning.
    Implements Algorithm 1 from the paper.
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
        # New parameters for Lagrangian Perturbation
        lag_mul_init: float = 0.1,
        lag_lr: float = 1e-4,
        policy_constraint: float = 0.1,
        perturb_lr: float = 3e-4,
        perturb_hidden_dims: list = [64, 64],
        delay_perturb_train: int = 0,
        smooth_factor: float = 0.1,
        kld_use_regul: bool = False,
        num_mc_samples: int = 10,
        init_perturb_scale: float = 0.001,
        target_perturb_scale: float = 0.15,
        perturb_warmup_steps: int = 900_000,
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
        
        # Initialize Lagrangian multiplier (in log space for positivity)
        init_val = np.log(max(lag_mul_init, 1e-5))
        self.log_lag_mul = th.nn.Parameter(
            th.tensor(init_val, device=self.device, dtype=th.float32),
            requires_grad=True
        )
        self.lag_optimizer = Adam([self.log_lag_mul], lr=lag_lr)
        
        # Policy constraint (eta in the algorithm)
        self.policy_constraint = policy_constraint
        
        # Regularization policy (global policy bar_pi)
        self.regul_policy = deepcopy(self.policy)
        self.regul_policy.set_training_mode(False)
        
        # Perturbation network
        obs_dim = self.observation_space.shape[0]
        self.target_perturb_scale = target_perturb_scale
        self.perturb_net = PerturbationNetwork(
            state_dim=obs_dim,
            hidden_dims=perturb_hidden_dims,
            init_perturb_scale=init_perturb_scale,
            target_perturb_scale=target_perturb_scale,
        ).to(self.device)
        self.perturb_optimizer = Adam(self.perturb_net.parameters(), lr=perturb_lr)

        self.delay_perturb_train = delay_perturb_train

        self.smooth_factor = smooth_factor

        self.kld_use_regul = kld_use_regul

        self.num_mc_samples = num_mc_samples

        self.perturb_warmup_steps = perturb_warmup_steps

    
    # def compute_kl_divergence(
    #     self,
    #     obs: th.Tensor,
    #     perturbed_obs: th.Tensor
    # ) -> th.Tensor:
    #     """
    #     簡潔版，但容易爆衝到0
    #     """
    #     with th.no_grad():
    #         # Get distributions from regularization policy (global policy)
    #         dist_original = self.policy.get_distribution(obs)
    #         dist_perturbed = self.policy.get_distribution(perturbed_obs)
            
    #         # Compute KL divergence
    #         kl_div = th.distributions.kl_divergence(
    #             dist_original.distribution,
    #             dist_perturbed.distribution
    #         )
            
    #     return kl_div.mean(dim=-1) if kl_div.dim() > 1 else kl_div
    
    def compute_kl_divergence(
        self,
        obs: th.Tensor,
        perturbed_obs: th.Tensor,
    ) -> th.Tensor:
        """
        複雜版，但似乎比較準
        """
        if self.kld_use_regul:
            policy = self.regul_policy
        else:
            policy = self.policy
        
        policy.set_training_mode(False)
        # Get mean and log_std directly from the policy network
        # This avoids issues with get_distribution() caching or resampling
        
        # For continuous action spaces (Gaussian policy)
        if hasattr(policy, 'action_net') and hasattr(policy, 'log_std'):
            # Get policy outputs for original states
            latent_pi_original = policy.mlp_extractor.forward_actor(
                policy.extract_features(obs, policy.pi_features_extractor)
            )
            mean_original = policy.action_net(latent_pi_original)
            
            # Get policy outputs for perturbed states
            latent_pi_perturbed = policy.mlp_extractor.forward_actor(
                policy.extract_features(perturbed_obs, policy.pi_features_extractor)
            )
            mean_perturbed = policy.action_net(latent_pi_perturbed)
            
            # Get log_std (might be state-independent or state-dependent)
            if policy.use_sde:
                # For SDE, we need to get the latent std
                log_std_original = policy.get_std_from_latent(latent_pi_original)
                log_std_perturbed = policy.get_std_from_latent(latent_pi_perturbed)
            else:
                # State-independent log_std
                log_std_original = policy.log_std
                log_std_perturbed = policy.log_std
            
            # Compute KL divergence between two Gaussian distributions
            # KL(N(μ1, σ1²) || N(μ2, σ2²)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
            std_original = th.exp(log_std_original)
            std_perturbed = th.exp(log_std_perturbed)
            
            var_original = std_original ** 2
            var_perturbed = std_perturbed ** 2
            
            kl_div = (
                th.log(std_perturbed / std_original) +
                (var_original + (mean_original - mean_perturbed) ** 2) / (2 * var_perturbed) -
                0.5
            )
            
            # Sum over action dimensions, mean over batch
            kl_div = kl_div.sum(dim=-1)
            
        else:
            # Fallback: try to use the distribution objects
            # Make sure to create independent distribution objects
            dist_original = policy.get_distribution(obs)
            dist_perturbed = policy.get_distribution(perturbed_obs)
            
            # Extract parameters and recreate distributions to ensure independence
            if hasattr(dist_original.distribution, 'mean') and hasattr(dist_original.distribution, 'stddev'):
                mean_orig = dist_original.distribution.mean.clone()
                std_orig = dist_original.distribution.stddev.clone()
                mean_pert = dist_perturbed.distribution.mean.clone()
                std_pert = dist_perturbed.distribution.stddev.clone()
                
                # Recreate distributions
                from torch.distributions import Normal
                dist_orig_new = Normal(mean_orig, std_orig)
                dist_pert_new = Normal(mean_pert, std_pert)
                
                kl_div = th.distributions.kl_divergence(dist_orig_new, dist_pert_new)
                kl_div = kl_div.sum(dim=-1)
            else:
                # Last resort: compute from log probs
                kl_div = th.distributions.kl_divergence(
                    dist_original.distribution,
                    dist_perturbed.distribution
                )
                kl_div = kl_div.sum(dim=-1) if kl_div.dim() > 1 else kl_div
            
        return kl_div
    
    def train(self) -> None:
        """
        Update policy using the current rollout buffer.
        Modified to include Lagrangian perturbation.
        """
        # Switch to train mode
        self.policy.set_training_mode(True)
        self.perturb_net.train()
        
        # Update optimizer learning rates
        self._update_learning_rate(self.policy.optimizer)
        
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        
        # Lists to track losses
        pg_losses, value_losses, smooth_losses = [], [], []
        perturb_losses, lag_losses, kl_diffs, delta_norm = [], [], [], []
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
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Check early stopping
                if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to reaching max kl")
                    break

                if self._n_updates < self.delay_perturb_train:
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                     # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                else:
                    # Update perturbation scale
                    if self.num_timesteps <= self.perturb_warmup_steps:
                        self.perturb_net.update_scale(self.num_timesteps / self.perturb_warmup_steps)

                    # === Perturbation Network Forward Pass ===
                    perturbed_obs, _, _, delta = self.perturb_net.sample_perturbation(obs)

                    # === Smoothness Loss (L_smooth) with Importance Sampling ===
                    # Evaluate local policy on perturbed observations
                    _, log_prob_local_perturbed, _ = self.policy.evaluate_actions(
                        perturbed_obs.detach(),  # Don't backprop through perturbation here
                        actions
                    )
                    
                    # Importance sampling ratio: π_θ(a|s̃) / π_old(a|s)
                    # Note: actions were sampled under π_old at original state s
                    ratio_smooth = th.exp(log_prob_local_perturbed - rollout_data.old_log_prob)
                    
                    # Clipped surrogate loss for smoothness
                    smooth_loss_1 = advantages * ratio_smooth
                    smooth_loss_2 = advantages * th.clamp(ratio_smooth, 1 - clip_range, 1 + clip_range)
                    smooth_loss = -th.min(smooth_loss_1, smooth_loss_2).mean()
                    
                    # === Combined Policy Update ===
                    total_policy_loss = policy_loss + self.smooth_factor * smooth_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Optimize policy
                    self.policy.optimizer.zero_grad()
                    total_policy_loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

                    # 取樣多次perturb
                    perturb_loss_sum = 0
                    kl_div_sum = 0

                    for _ in range(self.num_mc_samples):
                        # === Perturbation Network Loss (L_ψ) ===
                        # Recompute perturbation for gradient flow
                        perturbed_obs_grad, _, _, delta = self.perturb_net.sample_perturbation(obs)
                        
                        # Compute KL divergence: D_KL(π̄(·|s) || π̄(·|s̃))
                        kl_div = self.compute_kl_divergence(obs, perturbed_obs_grad)

                        kl_diff = kl_div - self.policy_constraint
                        
                        # Lagrangian multiplier value
                        lambda_val = th.exp(self.log_lag_mul)
                        
                        # Perturbation loss: L(δ, λ) = -||δ||² + λ(KL - η)
                        # We want to maximize ||δ||² (larger perturbations), so minimize -||δ||²
                        delta_norm_sq = th.norm(delta, dim=-1).mean()
                        perturb_loss = -delta_norm_sq + lambda_val.detach() * kl_diff.mean()
                        perturb_loss_sum += perturb_loss
                        kl_div_sum += kl_div.mean().item()

                    perturb_loss = perturb_loss_sum / self.num_mc_samples
                    avg_kl = kl_div_sum / self.num_mc_samples
                    # Optimize perturbation network
                    self.perturb_optimizer.zero_grad()
                    perturb_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.perturb_net.parameters(), self.max_grad_norm)
                    self.perturb_optimizer.step()
                    
                    # === Lagrangian Multiplier Update (L_λ) ===
                    # Loss: -λ * (KL - η) (we want to maximize λ when constraint is violated)
                    # with th.no_grad():
                    #     kl_div_detached = self.compute_kl_divergence(obs, perturbed_obs_grad.detach())
                    #     kl_diff_detached = kl_div_detached - self.policy_constraint
                    
                    kl_diff_detached = avg_kl - self.policy_constraint
                    
                    lag_loss = -(th.exp(self.log_lag_mul) * kl_diff_detached)
                    
                    # Optimize Lagrangian multiplier
                    self.lag_optimizer.zero_grad()
                    lag_loss.backward()
                    self.lag_optimizer.step()
                    
                    # Clamp log_lag_mul to prevent it from going too negative
                    with th.no_grad():
                        self.log_lag_mul.clamp_(min=np.log(1e-2), max=np.log(200))
                    
                    # Perturbation training Logging
                    smooth_losses.append(smooth_loss.item())
                    perturb_losses.append(perturb_loss.item())
                    lag_losses.append(lag_loss.item())
                    kl_diffs.append(kl_diff_detached)
                    delta_norm.append(delta_norm_sq.item())
                
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
            self.logger.record("train/perturb_loss", 0)
            self.logger.record("train/lag_loss", 0)
            self.logger.record("train/kl_difference", 0)
            self.logger.record("train/lambda", 0)
            self.logger.record("train/delta_norm", 0)
        else:
            self.logger.record("train/smooth_loss", np.mean(smooth_losses))
            self.logger.record("train/perturb_loss", np.mean(perturb_losses))
            self.logger.record("train/lag_loss", np.mean(lag_losses))
            self.logger.record("train/kl_difference", np.mean(kl_diffs))
            self.logger.record("train/lambda", th.exp(self.log_lag_mul).item())
            self.logger.record("train/delta_norm", np.mean(delta_norm))
            self.logger.record("train/perturb_scale", self.perturb_net.scale)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/n_updates", self._n_updates)
        
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def save(self, path: str) -> None:
        """
        Save the model including perturbation network and Lagrangian multiplier.
        
        Args:
            path: Path to save the model (without extension)
        
        Example:
            model.save("my_model")
            # 會生成: my_model.zip 和 my_model_perturbation.pth
        """
        # Save the base PPO model
        super().save(path)
        
        # Save additional components
        additional_data = {
            'perturb_net_state_dict': self.perturb_net.state_dict(),
            'perturb_optimizer_state_dict': self.perturb_optimizer.state_dict(),
            'log_lag_mul': self.log_lag_mul.detach().cpu(),
            'lag_optimizer_state_dict': self.lag_optimizer.state_dict(),
            'regul_policy_state_dict': self.regul_policy.state_dict(),
            'policy_constraint': self.policy_constraint,
            'perturb_hidden_dims': getattr(self, 'perturb_hidden_dims', [64, 64]),
            'delay_perturb_train': self.delay_perturb_train,
            'smooth_factor': self.smooth_factor,
            'kld_use_regul': self.kld_use_regul,
        }
        
        # Save to a separate file
        th.save(additional_data, path + "_perturbation.pth")
        print(f"✓ Saved perturbation network to {path}_perturbation.pth")

    @classmethod
    def load(cls, path: str, env: Optional[GymEnv] = None, device: Union[th.device, str] = "auto", **kwargs):
        """
        Load the model including perturbation network and Lagrangian multiplier.
        
        Args:
            path: Path to the saved model (without extension)
            env: Environment (if None, will try to load from saved model)
            device: Device to load the model on
            **kwargs: Additional arguments for PPO initialization
        
        Returns:
            Loaded PerturbationPPO model
        
        Example:
            model = PerturbationPPO.load("my_model", env=env)
            # 會載入: my_model.zip 和 my_model_perturbation.pth
        """
        # Load the base PPO model first
        # Note: Use the parent class's load method
        from stable_baselines3 import PPO
        base_model = PPO.load(path, env=env, device=device, **kwargs)
        
        # Create a new PerturbationPPO instance with the loaded parameters
        # We need to get the constructor arguments from the loaded model
        model = cls.__new__(cls)
        model.__dict__.update(base_model.__dict__)
        
        # Try to load additional components
        perturbation_path = path + "_perturbation.pth"
        
        try:
            additional_data = th.load(perturbation_path, map_location=model.device)
            
            # Store hyperparameters
            model.policy_constraint = additional_data['policy_constraint']
            model.perturb_hidden_dims = additional_data.get('perturb_hidden_dims', [64, 64])
            model.delay_perturb_train = additional_data.get('delay_perturb_train', 500)
            model.smooth_factor = additional_data.get('smooth_factor', 0.1)
            model.kld_use_regul = additional_data.get('kld_use_regul', False)
            model.target_perturb_scale = additional_data.get('target_perturb_scale', 0.15)
            
            # Reconstruct perturbation network
            obs_dim = model.observation_space.shape[0]
            model.perturb_net = PerturbationNetwork(
                state_dim=obs_dim,
                hidden_dims=model.perturb_hidden_dims,
                init_perturb_scale=model.target_perturb_scale,
                target_perturb_scale=model.target_perturb_scale,
            ).to(model.device)
            
            # Load state dicts
            model.perturb_net.load_state_dict(additional_data['perturb_net_state_dict'])
            
            # Recreate optimizers
            model.perturb_optimizer = Adam(model.perturb_net.parameters())
            model.perturb_optimizer.load_state_dict(additional_data['perturb_optimizer_state_dict'])
            
            # Load Lagrangian multiplier
            model.log_lag_mul = th.nn.Parameter(
                additional_data['log_lag_mul'].to(model.device),
                requires_grad=True
            )
            model.lag_optimizer = Adam([model.log_lag_mul])
            model.lag_optimizer.load_state_dict(additional_data['lag_optimizer_state_dict'])
            
            # Load regularization policy
            model.regul_policy = deepcopy(model.policy)
            model.regul_policy.load_state_dict(additional_data['regul_policy_state_dict'])
            model.regul_policy.set_training_mode(False)
            
            print(f"✓ Loaded perturbation network from {perturbation_path}")
            
        except FileNotFoundError:
            print(f"⚠ Warning: Could not find {perturbation_path}")
            print("  Perturbation network will be randomly initialized.")
            # Initialize with default values
            _initialize_perturbation_components(model)
        except Exception as e:
            print(f"⚠ Warning: Error loading perturbation data: {str(e)}")
            print("  Perturbation network will be randomly initialized.")
            _initialize_perturbation_components(model)
        
        return model

def _initialize_perturbation_components(model):
    """
    Helper function to initialize perturbation components when loading fails.
    """
    
    obs_dim = model.observation_space.shape[0]
    
    # Default hyperparameters
    model.policy_constraint = getattr(model, 'policy_constraint', 0.1)
    model.perturb_hidden_dims = getattr(model, 'perturb_hidden_dims', [64, 64])
    model.delay_perturb_train = getattr(model, 'delay_perturb_train', 500)
    model.smooth_factor = getattr(model, 'smooth_factor', 0.1)
    model.kld_use_regul = getattr(model, 'kld_use_regul', False)
    model.target_perturb_scale = getattr('target_perturb_scale', 0.15)
    
    # Initialize perturbation network
    model.perturb_net = PerturbationNetwork(
        state_dim=obs_dim,
        hidden_dims=model.perturb_hidden_dims,
        init_perturb_scale=model.target_perturb_scale,
        target_perturb_scale=model.target_perturb_scale
    ).to(model.device)
    model.perturb_optimizer = Adam(model.perturb_net.parameters(), lr=1e-4)
    
    # Initialize Lagrangian multiplier
    model.log_lag_mul = th.nn.Parameter(
        th.tensor(np.log(1e-2), device=model.device, dtype=th.float32),
        requires_grad=True
    )
    model.lag_optimizer = Adam([model.log_lag_mul], lr=1e-4)
    
    # Initialize regularization policy
    model.regul_policy = deepcopy(model.policy)
    model.regul_policy.set_training_mode(False)
