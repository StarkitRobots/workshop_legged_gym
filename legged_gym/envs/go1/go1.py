import torch
from legged_gym.utils.isaacgym_utils import get_euler_xyz
from legged_gym.envs import LeggedRobot
from isaacgym import gymtorch

class Go1(LeggedRobot):
    def compute_observations(self):
        """ Computes observations
        """
        # <YOUR CODE>
        # соберите вектор наблюдений
        # для примера используйте compute_observations из envs/base/legged_robot
        # вектор наблюдений почти такой же как в envs/base/legged_robot
        # но в нем не должно быть линейной скорости base
        # и команд на скорость
        pass
        
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[18:30] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[30:42] = 0. # previous actions
        return noise_vec

    def _reward_tracking_pitch(self):
        # Tracking
        base_quat = self.root_states[:, 3:7] # 0-2: position x, y, z; 3-7: orientation in quaternion
        euler = get_euler_xyz(base_quat) # euler - orientation roll pitch yaw
        # <YOUR CODE>
        # compute reward for tracking pitch command
        # для примера возьмите _reward_tracking_lin_vel из envs/base/legged_robot
        pass
    
    def _reward_hip_pos(self):
        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)