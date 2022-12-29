import numpy as np

variants = {
    "default": (2, 300, 0, -0.1, 10, -3, -3, 0, "push"), # default has no obstacles
    "two": (2, 300, 0, -0.1, 10, -3, -3, 0, "two"), # default has no obstacles
}
# control_freq, var_horizon, num_obstacles, standard_reward, target_reward, obstacle_reward, out_of_bounds_reward, mode

    # TARGET_HALFSIZE = 0.05  # half of side length of block
    # PUCK_RADIUS = 0.025  # radius of goal circle
    # PADDLE_HANDWIDTH = 0.025
    # PADDLE_HALFRADIUS = 0.06
    # PADDLE_HEIGHT = 0.1
    # GRIPPER_BOUNDS = np.array([
    #     [-0.2, 0],  # x
    #     [-0.1, 0.1],  # y
    #     [0, 0.2],  # z
    # ])
    # PADDLE_SPAWN_AREA = np.array([
    #     [-0.2, -0.08],  # x
    #     [-0.1, 0.1],  # y
    # ])
    # TARGET_SPAWN_AREA = np.array([
    #     [0.12, 0.13],  # x
    #     [-0.1, 0.1],  # y
    # ])
    # PUCK_SPAWN_AREA = np.array([
    #     [-0.1, 0.07],  # x
    #     [-0.1, 0.1],  # y
    # ])
    # SPAWN_AREA_SIZE = 0.15
    # OBSTACLE_GRID_RESOLUTION = 5  # side length of obstacle grid
    # OBSTACLE_HEIGHT = 0.1
    # OBSTACLE_HALF_SIDELENGTH = SPAWN_AREA_SIZE / OBSTACLE_GRID_RESOLUTION



ranges_fixed = {
    "Action": [np.array([-1,-1,-1]).astype(np.float64), np.array([1,1,1]).astype(np.float64)],
    "Gripper": [np.array([-0.3, -0.31, .80]).astype(np.float64), np.array([0.2, 0.21, 1]).astype(np.float64)],
    "Paddle": [np.array([-0.3, -0.31, .80]).astype(np.float64), np.array([0.2, 0.21, 1]).astype(np.float64)],
    "Puck": [np.array([-0.3, -0.31, .80]).astype(np.float64), np.array([0.2, 0.21, 1]).astype(np.float64)],
    "Obstacle": [np.array([-0.3, -0.31, .80]).astype(np.float64), np.array([0.2, 0.21, 1]).astype(np.float64)],
    "Target": [np.array([-0.3, -0.31, .80]).astype(np.float64), np.array([0.5, 0.21, 1]).astype(np.float64)],
    "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
    "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
}

dynamics_fixed = {
    "Action": [np.array([-2,-2,-2]).astype(np.float64), np.array([2,2,2]).astype(np.float64)],
    "Gripper": [np.array([-0.05, -0.05, -0.05]).astype(np.float64), np.array([0.05, 0.05, 0.05]).astype(np.float64)],
    "Paddle": [np.array([-0.05, -0.05, -.05]).astype(np.float64), np.array([0.05, 0.05, 0.05]).astype(np.float64)],
    "Puck": [np.array([-0.05, -0.05, -.05]).astype(np.float64), np.array([0.05, 0.05, 0.05]).astype(np.float64)],
    "Obstacle": [np.array([-0.05, -0.05, -.05]).astype(np.float64), np.array([0.05, 0.05, 0.05]).astype(np.float64)],
    "Target": [np.array([-0.05, -0.05, -.05]).astype(np.float64), np.array([0.05, 0.05, 0.05]).astype(np.float64)],
    "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
    "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
}


ranges = {
    "Action": [np.array([-1,-1,-1]).astype(np.float64), np.array([1,1,1]).astype(np.float64)],
    "Gripper": [np.array([-0.3, -0.2, .831]).astype(np.float64), np.array([0.1, 0.2, 1]).astype(np.float64)],
    "Puck": [np.array([-0.2, -0.31, .82]).astype(np.float64), np.array([0.2, 0.21, .84]).astype(np.float64)],
    "Paddle": [np.array([-0.2, -0.31, .82]).astype(np.float64), np.array([0.2, 0.21, .84]).astype(np.float64)],
    "Obstacle": [np.array([-0.22, -0.12, .80]).astype(np.float64), np.array([0.02, 0.12, 0.81]).astype(np.float64)],
    "Target": [np.array([-0.25, -0.15, .80]).astype(np.float64), np.array([0.05, 0.15, 0.81]).astype(np.float64)],
    "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
    "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
}

dynamics = {
    "Action": [np.array([-2,-2,-2]).astype(np.float64), np.array([2,2,2]).astype(np.float64)],
    "Gripper": [np.array([-0.05, -0.05, -0.03]).astype(np.float64), np.array([0.05, 0.05, 0.03]).astype(np.float64)],
    "Paddle": [np.array([-0.05, -0.05, -.05]).astype(np.float64), np.array([0.05, 0.05, 0.05]).astype(np.float64)],
    "Puck": [np.array([-0.05, -0.05, -.05]).astype(np.float64), np.array([0.05, 0.05, 0.05]).astype(np.float64)],
    "Obstacle": [np.array([0, 0, 0]).astype(np.float64), np.array([0, 0, 0]).astype(np.float64)],
    "Target": [np.array([0, 0, 0]).astype(np.float64), np.array([0, 0, 0]).astype(np.float64)],
    "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
    "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
}


position_masks = {
    "Action": np.array([0,0,0]),
    "Gripper":np.array([1,1,1]),
    "Paddle": np.array([1,1,1]),
    "Puck": np.array([1,1,1]),
    "Obstacle": np.array([1,1,1]),
    "Target": np.array([1,1,1]),
    "Done": np.array([0]),
    "Reward": np.array([0]),
}

instanced = {
    "Action": 1,
    "Gripper":1,
    "Puck": 1,
    "Paddle": 1,
    "Obstacle": 20,
    "Target": 1,
    "Done": 1,
    "Reward": 1
}