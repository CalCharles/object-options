variants = {
    "no_obstacles": (2, 0, -0.1, 10, -3, -3, 0),
    "obstacles": (2, 10, -0.1, 10, -3, -3, 0),
    "obstacles_many": (2, 15, -0.1, 10, -3, -3, 0)
    "joint": (2, 15, -0.1, 10, -3, -3, 1)
    "hard": (2, 15, -1, 10, -3, -3, 2)
    "planar": (2, 15, -0.1, 10, -3, -3, 3)
}

ranges = {
    "Action": [np.array([-1,-1,-1]).astype(np.float64), np.array([1,1,1]).astype(np.float64)],
    "Gripper": [np.array([-0.2, -0.31, .83]).astype(np.float64), np.array([0.2, 0.21, 0.915]).astype(np.float64)],
    "Block": [np.array([-0.2, -0.31, .81]).astype(np.float64), np.array([0.2, 0.21, .84]).astype(np.float64)],
    "Obstacle": [np.array([-0.27, -0.26, .82]).astype(np.float64), np.array([0.13, 0.26, 0.83]).astype(np.float64)],
    "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
    "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
}

dynamics = {
    "Action": [np.array([-2,-2,-2]).astype(np.float64), np.array([2,2,2]).astype(np.float64)],
    "Gripper": [np.array([-0.05, -0.05, -0.03]).astype(np.float64), np.array([0.05, 0.05, 0.03]).astype(np.float64)],
    "Block": [np.array([-0.05, -0.05, -.002]).astype(np.float64), np.array([0.05, 0.05, 0.002]).astype(np.float64)],
    "Obstacle": [np.array([0, 0, 0]).astype(np.float64), np.array([0, 0, 0]).astype(np.float64)],
    "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
    "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
}


position_masks = {
    "Action": np.array([0,0,0]),
    "Gripper":np.array([1,1,1]),
    "Block": np.array([1,1,1]),
    "Obstacle": np.array([1,1,1]),
    "Done": np.array([0]),
    "Reward": np.array([0]),
}

instanced = {
    "Action": False,
    "Gripper":False,
    "Block": False,
    "Obstacle": True,
    "Done": False,
    "Reward": False
}