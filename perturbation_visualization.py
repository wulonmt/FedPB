import cv2
import numpy as np
import gymnasium as gym
import torch as th
from typing import Optional
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from utils.init_pos_config import get_init_pos, is_costum_env
import Env
from normalized_env import get_eval_norm_env


class PerturbationVisualizerWrapper(gym.Wrapper):
    """
    Gym Wrapper for visualizing perturbations in "Ghost Mode".
    Displays both real state (solid) and perturbed state (transparent ghost).
    """
    
    def __init__(
        self, 
        env: gym.Env,
        perturb_net: Optional[th.nn.Module],
        device: str = "cpu",
        # Rendering parameters (for CartPole)
        screen_width: int = 600,
        screen_height: int = 400,
        world_width: float = 4.8,
        cart_y_offset: float = 100,  # pixels from bottom
        pole_length_pixels: float = 100,  # visual pole length
        ghost_alpha: float = 0.4,  # transparency of ghost
        show_velocity_arrows: bool = True,
        arrow_scale: float = 20.0,  # scale for velocity arrows
        action_arrow_scale: float = 150.0,
        enable_perturbation: bool = True,  # 是否啟用擾動
        debug_mode: bool = False,  # 調試模式
    ):
        super().__init__(env)
        
        self.perturb_net = perturb_net
        self.device = device
        self.enable_perturbation = enable_perturbation
        self.debug_mode = debug_mode
        
        # Rendering parameters
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.world_width = world_width
        self.scale = screen_width / world_width
        self.cart_y_offset = cart_y_offset
        self.pole_length_pixels = pole_length_pixels
        self.ghost_alpha = ghost_alpha
        self.show_velocity_arrows = show_velocity_arrows
        self.arrow_scale = arrow_scale
        self.action_arrow_scale = action_arrow_scale
        
        # Tracking variables
        self.cumulative_reward = 0
        self.current_reward = 0
        self.current_obs = None
        self.perturbed_obs = None
        
        # Set perturbation network to eval mode
        if self.perturb_net is not None:
            self.perturb_net.eval()
    
    def reset(self, **kwargs):
        self.cumulative_reward = 0
        self.current_reward = 0
        obs, info = super().reset(**kwargs)
        self.current_obs = obs
        self.current_action = 0
        
        # Compute perturbed observation
        if self.perturb_net is not None:
            self.perturbed_obs, self.perturbed_std = self._compute_perturbed_obs(obs)
        else:
            self.perturbed_obs, self.perturbed_std = obs.copy(), 0
        
        return obs, info
    
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        
        self.current_reward = reward
        self.cumulative_reward += reward
        self.current_obs = obs
        self.current_action = action
        
        # Compute perturbed observation
        if self.perturb_net is not None and self.enable_perturbation:
            self.perturbed_obs, self.perturbed_std = self._compute_perturbed_obs(obs)
        else:
            self.perturbed_obs, self.perturbed_std = obs.copy(), 0
        
        return obs, reward, done, truncated, info
    
    def _compute_perturbed_obs(self, obs: np.ndarray) -> np.ndarray:
        """Compute perturbed observation using perturbation network"""
        if self.perturb_net is None:
            return obs.copy()
            
        with th.no_grad():
            obs_tensor = th.tensor(obs, dtype=th.float32, device=self.device).unsqueeze(0)
            
            # 確保 perturbation network 在 eval 模式
            self.perturb_net.eval()
            
            # 獲取擾動
            perturbed_obs_tensor, mu, log_sigma, delta = self.perturb_net.sample_perturbation(obs_tensor)
            # perturbed_obs = perturbed_obs_tensor.squeeze(0).cpu().numpy()
            perturbed_mu = (obs_tensor + mu).squeeze(0).cpu().numpy()
            std = th.exp(log_sigma).squeeze(0).cpu().numpy()
            
            # # 調試信息：檢查擾動是否過大
            # delta_np = delta.squeeze(0).cpu().numpy()
            # delta_norm = np.linalg.norm(delta_np)
            
            # # 如果擾動異常大，打印警告
            # if delta_norm > 2.0:  # 根據你的環境調整這個閾值
            #     print(f"Warning: Large perturbation detected! ||delta|| = {delta_norm:.3f}")
            #     print(f"  Original obs: {obs}")
            #     print(f"  Delta: {delta_np}")
            #     print(f"  Perturbed obs: {perturbed_obs}")
            
        return perturbed_mu, std
    
    def render(self):
        """Override render to add ghost mode visualization"""
        img = super().render()
        
        if img is None:
            return None
        
        img = np.ascontiguousarray(img, dtype=np.uint8)
        
        # Draw ghost cart and pole if we have perturbation data
        if self.current_obs is not None and self.perturbed_obs is not None:
            img = self._draw_ghost_mode(img)
        
        # Draw reward information (keep original functionality)
        img = self._draw_reward_info(img)
        
        return img
    
    def _draw_ghost_mode(self, img: np.ndarray) -> np.ndarray:
        """
        Draw ghost cart and pole on the image.
        Assumes CartPole-like environment with state: [x, x_dot, theta, theta_dot]
        or [x, x_dot, cos(theta), sin(theta), theta_dot]
        """
        # Parse states
        real_state = self.current_obs
        ghost_state = self.perturbed_obs
        ghost_shift = self.perturbed_std
        action = self.current_action
        
        # Determine state format (4 or 5 dimensions)
        if len(real_state) == 4:
            # [x, x_dot, theta, theta_dot]
            real_x, real_x_dot, real_theta, real_theta_dot = real_state
            ghost_x, ghost_x_dot, ghost_theta, ghost_theta_dot = ghost_state
        elif len(real_state) == 5:
            # [x, x_dot, cos(theta), sin(theta), theta_dot]
            real_x, real_x_dot, cos_theta, sin_theta, real_theta_dot = real_state
            real_theta = np.arctan2(sin_theta, cos_theta)
            
            ghost_x, ghost_x_dot, g_cos_theta, g_sin_theta, ghost_theta_dot = ghost_state
            ghost_shift_x, ghost_shift_x_dot, g_shift_cos_theta, g_shift_sin_theta, ghost_shift_theta_dot = ghost_shift
            
            def refine_theta(sin, cos):
                # cos、sin不一致，要先還原成圓再算
                r = np.sqrt(sin**2 + cos**2) + 1e-8
                sin_n = sin / r
                cos_n = cos / r
                return np.arctan2(sin_n, cos_n)

            ghost_theta = refine_theta(g_sin_theta, g_cos_theta)
            shift_angle_list = [
                refine_theta(g_sin_theta + g_shift_sin_theta, g_cos_theta + g_shift_cos_theta),
                refine_theta(g_sin_theta + g_shift_sin_theta, g_cos_theta - g_shift_cos_theta),
                refine_theta(g_sin_theta - g_shift_sin_theta, g_cos_theta + g_shift_cos_theta),
                refine_theta(g_sin_theta - g_shift_sin_theta, g_cos_theta - g_shift_cos_theta)
            ]
            ghost_shift_theta_max = max(shift_angle_list)
            ghost_shift_theta_min = min(shift_angle_list)

        else:
            # Unknown format, skip ghost drawing
            return img
        
        # Create overlay for transparent drawing
        overlay = img.copy()
        
        # Draw ghost cart and pole
        self._draw_cart_and_pole(
            overlay, 
            ghost_x, 
            ghost_theta,
            color=(0, 0, 255),  # Blue color for ghost
            shift=ghost_shift_x,
            shift_theta_max=ghost_shift_theta_max,
            shift_theta_min=ghost_shift_theta_min
        )
        
        # Blend overlay with original image
        img = cv2.addWeighted(overlay, self.ghost_alpha, img, 1 - self.ghost_alpha, 0)
        
        # Draw velocity arrows if enabled
        if self.show_velocity_arrows:
            img = self._draw_velocity_arrows(
                img, 
                real_x, real_x_dot, real_theta, real_theta_dot,
                ghost_x, ghost_x_dot, ghost_theta, ghost_theta_dot,
                ghost_shift_x_dot, ghost_shift_theta_dot,
                action,
            )
        
        # Draw perturbation magnitude info (移到右上角避免重疊)
        delta = ghost_state - real_state
        delta_norm = np.linalg.norm(delta)
        
        # 添加半透明背景
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (self.screen_width // 2 - 55, 5),
            (self.screen_width - 5, 90),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        cv2.putText(
            img,
            f'Perturbation: ||delta|| = {delta_norm:.3f}',
            (self.screen_width - 245, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 100),  # 淺黃色，更清楚
            1,
            cv2.LINE_AA
        )
        reduce_arr = lambda x: np.array2string(x, precision=2, suppress_small=True)
        cv2.putText(
            img,
            f'mu = {reduce_arr(ghost_state - real_state)}',
            (self.screen_width // 2 - 50, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 100),  # 淺黃色，更清楚
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            img,
            f'std = {reduce_arr(ghost_shift)}',
            (self.screen_width // 2 - 50, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 100),  # 淺黃色，更清楚
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            img,
            f'      [x, x_dot, cos, sin, theta_dot]',
            (self.screen_width // 2 - 50, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 100),  # 淺黃色，更清楚
            1,
            cv2.LINE_AA
        )
        
        return img
    
    def _draw_cart_and_pole(
        self, 
        img: np.ndarray, 
        x: float, 
        theta: float,
        color: tuple = (0, 0, 255),
        shift: float = 0,
        shift_theta_max: float = 0,
        shift_theta_min: float = 0,
    ):
        """
        Draw cart and pole at given position.
        """
        # Convert world coordinates to pixel coordinates
        cart_x_pixel = int(x * self.scale + self.screen_width / 2)
        cart_y_pixel = self.screen_height - self.cart_y_offset
        
        # Draw cart (rectangle)
        cart_width = 50
        cart_height = 30
        cv2.rectangle(
            img,
            (cart_x_pixel - cart_width // 2, cart_y_pixel - cart_height // 2),
            (cart_x_pixel + cart_width // 2, cart_y_pixel + cart_height // 2),
            color,
            -1  # thickness (hollow)
        )

        # Draw shift range (rectangle)
        cart_shift_width = 50 + 2 * shift * self.scale
        cart_shift_width = int(min(cart_shift_width, self.screen_width))
        cv2.rectangle(
            img,
            (cart_x_pixel - cart_shift_width // 2, cart_y_pixel - cart_height // 2),
            (cart_x_pixel + cart_shift_width // 2, cart_y_pixel + cart_height // 2),
            (100, 100, 255),
            2  # thickness (hollow)
        )
        
        # Draw pole (line)
        pole_end_x = cart_x_pixel - int(self.pole_length_pixels * np.sin(theta))
        pole_end_y = cart_y_pixel - int(self.pole_length_pixels * np.cos(theta))
        
        cv2.line(
            img,
            (cart_x_pixel, cart_y_pixel),
            (pole_end_x, pole_end_y),
            (0, 255, 0),
            3  # thickness
        )

        shift_theta_1 = shift_theta_max
        shift_theta_2 = shift_theta_min
        
        # Draw shift pole (line)
        sh1_pole_end_x = cart_x_pixel - int(self.pole_length_pixels * np.sin(shift_theta_1))
        sh1_pole_end_y = cart_y_pixel - int(self.pole_length_pixels * np.cos(shift_theta_1))
        sh2_pole_end_x = cart_x_pixel - int(self.pole_length_pixels * np.sin(shift_theta_2))
        sh2_pole_end_y = cart_y_pixel - int(self.pole_length_pixels * np.cos(shift_theta_2))

        cv2.line(
            img,
            (cart_x_pixel, cart_y_pixel),
            (sh1_pole_end_x, sh1_pole_end_y),
            (255, 255, 0), # yellow
            3  # thickness
        )
        cv2.line(
            img,
            (cart_x_pixel, cart_y_pixel),
            (sh2_pole_end_x, sh2_pole_end_y),
            (255, 128, 0), # orange
            3  # thickness
        )
        
        # Draw pole tip (circle)
        cv2.circle(
            img,
            (pole_end_x, pole_end_y),
            8,
            color,
            2
        )
    
    def _draw_velocity_arrows(
        self,
        img: np.ndarray,
        real_x: float, real_x_dot: float, real_theta: float, real_theta_dot: float,
        ghost_x: float, ghost_x_dot: float, ghost_theta: float, ghost_theta_dot: float,
        ghost_shift_x_dot: float, ghost_shift_theta_dot: float,
        action: float,
    ) -> np.ndarray:
        """
        Draw velocity arrows for both real and ghost states.
        """
        cart_y_pixel = self.screen_height - self.cart_y_offset
        
        # Real state velocity arrow (黃色) - 調整到車子上方
        real_cart_x = int(real_x * self.scale + self.screen_width / 2)
        real_arrow_end_x = int(real_cart_x + real_x_dot * self.arrow_scale)
        
        cv2.arrowedLine(
            img,
            (real_cart_x, cart_y_pixel - 60),  # 改到車子上方 60 像素
            (real_arrow_end_x, cart_y_pixel - 60),
            (0, 255, 255),  # 亮靛青色 (BGR)
            3,  # 加粗
            tipLength=0.3
        )
        
        # 添加標籤
        cv2.putText(
            img,
            'Real v',
            (real_cart_x - 30, cart_y_pixel - 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        # Ghost state velocity arrow (紅色) - 調整位置
        ghost_cart_x = real_cart_x
        ghost_arrow_end_x = int(ghost_cart_x + ghost_x_dot * self.arrow_scale)
        
        # Ghost velocity 加shift疊在底下
        shift = ghost_shift_x_dot if ghost_arrow_end_x > ghost_cart_x else -ghost_shift_x_dot
        shift_start = int(ghost_cart_x - shift*self.arrow_scale)
        shift_end = int(ghost_arrow_end_x + shift*self.arrow_scale)
        cv2.line(
            img,
            (shift_start, cart_y_pixel - 40),
            (shift_end, cart_y_pixel - 40),
            (255, 0, 255),
            2,
        )

        cv2.arrowedLine(
            img,
            (ghost_cart_x, cart_y_pixel - 40),  # 改到車子上方 40 像素
            (ghost_arrow_end_x, cart_y_pixel - 40),
            (0, 0, 255),
            3,  # 加粗
            tipLength=0.3
        )

        
        # 添加標籤
        cv2.putText(
            img,
            'Ghost v',
            (ghost_cart_x - 35, cart_y_pixel - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )
        
        # Angular velocity arrows at pole tip (只畫真實的，避免混亂)
        real_pole_end_x = real_cart_x - int(self.pole_length_pixels * np.sin(real_theta))
        real_pole_end_y = cart_y_pixel - int(self.pole_length_pixels * np.cos(real_theta))
        
        # Tangent direction for angular velocity
        real_tangent_x = int(real_pole_end_x - real_theta_dot * self.arrow_scale * np.cos(real_theta))
        real_tangent_y = int(real_pole_end_y + real_theta_dot * self.arrow_scale * np.sin(real_theta))
        
        cv2.arrowedLine(
            img,
            (real_pole_end_x, real_pole_end_y),
            (real_tangent_x, real_tangent_y),
            (0, 255, 0),  # 綠色表示角速度
            2,
            tipLength=0.4
        )

        # 添加標籤
        cv2.putText(
            img,
            'Real w',
            (real_pole_end_x, real_pole_end_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        # Ghost Angular velocity arrows at pole tip
        ghost_pole_end_x = real_cart_x - int((self.pole_length_pixels + 20) * np.sin(real_theta))
        ghost_pole_end_y = cart_y_pixel - int((self.pole_length_pixels + 20) * np.cos(real_theta))
        
        # Ghost Tangent direction for angular velocity
        ghost_tangent_x = int(ghost_pole_end_x - ghost_theta_dot * self.arrow_scale * np.cos(real_theta))
        ghost_tangent_y = int(ghost_pole_end_y + ghost_theta_dot * self.arrow_scale * np.sin(real_theta))

        cv2.arrowedLine(
            img,
            (ghost_pole_end_x, ghost_pole_end_y),
            (ghost_tangent_x, ghost_tangent_y),
            (0, 100, 0),  # 綠色表示角速度
            2,
            tipLength=0.4
        )

        # 添加標籤
        cv2.putText(
            img,
            'Ghost w',
            (ghost_pole_end_x, ghost_pole_end_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 100, 0),
            1,
            cv2.LINE_AA
        )

        # Angular velocity shift arrows at pole tip
        dot_shift_pos = ghost_shift_theta_dot + ghost_shift_theta_dot
        ghost_shift_pole_end_x = ghost_tangent_x - int(dot_shift_pos * self.arrow_scale * np.sin(real_theta))
        ghost_shift_pole_end_y = ghost_tangent_y - int(dot_shift_pos * self.arrow_scale * np.cos(real_theta))
        
        # Tangent direction shift for angular velocity
        dot_shift_neg = ghost_shift_theta_dot - ghost_shift_theta_dot
        ghost_shift_tangent_x = ghost_tangent_x - int(dot_shift_neg * self.arrow_scale * np.sin(real_theta))
        ghost_shift_tangent_y = ghost_tangent_y - int(dot_shift_neg * self.arrow_scale * np.cos(real_theta))
        
        # cv2.line(
        #     img,
        #     (ghost_shift_pole_end_x, ghost_shift_pole_end_y),
        #     (ghost_shift_tangent_x, ghost_shift_tangent_y),
        #     (125, 102, 39),
        #     2,
        # )

        # Action arrow
        action_x = self.screen_width // 2
        action_y = self.screen_height // 4
        # CartPole action is only 1-dimension
        action_x_end = action_x + int(action * self.action_arrow_scale)
        
        cv2.arrowedLine(
            img,
            (action_x, action_y),
            (action_x_end, action_y),
            (140, 0, 0),
            3,
            tipLength=0.3
        )

        cv2.putText(
            img,
            'Action',
            (action_x, action_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (140, 0, 0),
            1,
            cv2.LINE_AA
        )
        

        return img
    
    def _draw_reward_info(self, img: np.ndarray) -> np.ndarray:
        """
        Draw reward information (keep original functionality).
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Prepare text
        text_curr = f'Current Reward: {self.current_reward:.2f}'
        text_cum = f'Accumulated Reward: {self.cumulative_reward:.2f}'
        
        # Get text size
        (curr_width, curr_height), _ = cv2.getTextSize(text_curr, font, 0.5, 1)
        (cum_width, cum_height), _ = cv2.getTextSize(text_cum, font, 0.5, 1)
        
        # Use maximum width
        max_width = max(curr_width, cum_width)
        total_height = curr_height + cum_height + 10
        
        # Add background
        overlay = img.copy()
        cv2.rectangle(
            overlay, 
            (5, 5), 
            (10 + max_width, 15 + total_height), 
            (0, 0, 0), 
            -1
        )
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)  # 加深背景
        
        # Add text
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
        
        # Add legend (移到左下角，分開顯示)
        legend_y_start = self.screen_height - 80
        
        # 添加背景
        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (5, legend_y_start - 5),
            (200, self.screen_height - 5),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        # Legend 內容
        cv2.putText(img, 'Legend:', (10, legend_y_start + 15), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Real state (黑色車子)
        cv2.rectangle(img, (15, legend_y_start + 25), (30, legend_y_start + 35), (0, 0, 0), -1)
        cv2.putText(img, 'Real State', (35, legend_y_start + 35), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Ghost state (紅色輪廓)
        cv2.rectangle(img, (15, legend_y_start + 45), (30, legend_y_start + 55), (0, 0, 255), 2)
        cv2.putText(img, 'Ghost State', (35, legend_y_start + 55), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        
        return img


# ============================================================================
# Recording Script
# ============================================================================

def record_perturbation_demo(
    env,
    model,
    perturb_net: th.nn.Module,
    save_dir: str = "./perturbation_videos",
    video_length: int = 1000,
    episodes: int = 3,
    device: str = "cpu"
):
    """
    Record a video demonstrating perturbation robustness with ghost mode.
    
    Args:
        env_name: Name of the gym environment
        model: Trained PPO model
        perturb_net: Trained perturbation network
        save_dir: Directory to save videos
        video_length: Total frames to record
        episodes: Number of episodes
        device: Device for perturbation network
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # # Create environment with ghost mode wrapper
    # def make_env():
    #     env = get_env(env_name)
    #     env = PerturbationVisualizerWrapper(
    #         env, 
    #         perturb_net=perturb_net,
    #         device=device,
    #         ghost_alpha=0.9,
    #         show_velocity_arrows=True
    #     )
    #     return env
    
    # # Create vectorized environment
    # vec_env = DummyVecEnv([make_env])
    
    # Wrap with video recorder
    vec_env = VecVideoRecorder(
        env, 
        save_dir,
        record_video_trigger=lambda x: x == 0,
        name_prefix="robust_agent_ghost_mode",
        video_length=video_length
    )
    
    print(f"Recording {video_length} frames ({episodes} episodes)...")
    
    # Run episodes
    obs = vec_env.reset()
    for step in range(video_length + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        
        if step % 100 == 0:
            print(f"  Frame {step}/{video_length}")
    
    vec_env.close()
    print(f"✓ Video saved to: {save_dir}")

def get_env(env_name: str, index: int = 0, mode:str = "rgb_array"):
    if not is_costum_env(env_name):
        env = gym.make(env_name, render_mode=mode)
    else:
        env = gym.make(env_name, render_mode=mode, **get_init_pos(env_name, index))
    return env

# ============================================================================
# Complete Usage Example
# ============================================================================

if __name__ == "__main__":
    import gymnasium as gym
    
    # Example: Load trained model and perturbation network
    env_name = "CartPoleSwingUpV1WithAdjustablePole-v0"
    
    print("="*70)
    print("1. Loading environment and model...")
    print("="*70)
    
    # Load trained PerturbationPPO model
    # Replace with your actual loading code
    from utils.PerturbationPPO import PerturbationPPO
    
    path = "multiagent\\2025_12_25_19_35_CartPoleSwingUpV1WithAdjustablePole-v0_c3\\PBPPO_regul0\\rep1"
    norm_path = path + f"/{env_name}/0_PerturbPPO"
    env = get_eval_norm_env(env_name, 
                            path=norm_path, 
                            render_mode="rgb_array", 
                            index=0,
                            )
    model_path = norm_path + "/model"
    model = PerturbationPPO.load(model_path, env=env, device="cpu")
    perturb_net = model.perturb_net
    print("✓ Model and perturbation network loaded")
    
    # ========== Option 1: Record video with ghost mode ==========
    print("\n" + "="*70)
    print("2. Recording demonstration video...")
    print("="*70)

    env = get_eval_norm_env(env_name, 
                            path=norm_path, 
                            render_mode="rgb_array", 
                            index=0, 
                            wrapper_class=PerturbationVisualizerWrapper,
                            wrapper_kwargs={
                                "perturb_net": perturb_net,
                                }
                            )
    
    record_perturbation_demo(
        env=env,
        model=model,
        perturb_net=perturb_net,
        save_dir="./perturbation_demos",
        video_length=1000,
        episodes=3,
        device="cpu"
    )
    
    # ========== Option 2: Test wrapper manually ==========
    # print("\n" + "="*70)
    # print("3. Testing wrapper manually...")
    # print("="*70)
    
    # test_env = get_env(env_name, mode="human")
    # test_env = PerturbationVisualizerWrapper(
    #     test_env,
    #     perturb_net=perturb_net,
    #     device="cpu"
    # )
    
    # obs, _ = test_env.reset()
    # for _ in range(500):
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done, truncated, info = test_env.step(action)
    #     test_env.render()
        
    #     if done or truncated:
    #         obs, _ = test_env.reset()
    
    # test_env.close()
    # print("✓ Manual test completed")