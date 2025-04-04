from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go1RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        #<YOUR CODE>
        #calculate num_observations for Go1 
        episode_length_s = 10

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0,  # [rad]
            'RL_hip_joint': 0,  # [rad]
            'FR_hip_joint': -0,  # [rad]
            'RR_hip_joint': -0,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 0.8,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 0.8,  # [rad]

            'FL_calf_joint': -1.3,  # [rad]
            'RL_calf_joint': -1.3,  # [rad]
            'FR_calf_joint': -1.3,  # [rad]
            'RR_calf_joint': -1.3,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = []
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        tracking_sigma = 0.75
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            lin_vel_z = 0.
            ang_vel_xy = 0.
            feet_air_time = 0.
            tracking_pitch = 1.5
            hip_pos = -0.5
            feet_drag = -0.
            collision = 0.
    
    # YOUR CODE 
    # add parameters for new rewards

    # YOUR CODE
    # add domain_rand (LeggedRobotCfg.domain_rand)

class Go1RoughCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.5
        
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'go1'
        max_iterations = 500
