import os

breakout_configs = [os.path.join("/hdd", "datasets", "object_data", "breakout", "random"), os.path.join("configs", "Breakout", "action_paddle_interaction.yaml"), os.path.join("configs", "Breakout", "action_paddle_mask.yaml"), os.path.join("configs", "Breakout", "action_paddle_option.yaml"), os.path.join("configs", "Breakout", "paddle_ball_interaction.yaml"), os.path.join("configs", "Breakout", "paddle_ball_mask.yaml"), os.path.join("configs", "Breakout", "paddle_ball_option.yaml")]
breakout_variant_configs = [os.path.join("configs", "Breakout", "center.yaml"), os.path.join("configs", "Breakout", "single.yaml"), os.path.join("configs", "Breakout", "neg.yaml"), os.path.join("configs", "Breakout", "prox.yaml"), os.path.join("configs", "Breakout", "big.yaml"), os.path.join("configs", "Breakout", "hard.yaml")]
robopushing_configs = [os.path.join("/nfs", "data", "object_data", "robopush", "random"), os.path.join("configs", "RoboPushing", "action_gripper_interaction.yaml"), os.path.join("configs", "RoboPushing", "action_gripper_mask.yaml"), os.path.join("configs", "RoboPushing", "action_gripper_option.yaml"), os.path.join("configs", "RoboPushing", "gripper_block_interaction.yaml"), os.path.join("configs", "RoboPushing", "gripper_block_mask.yaml"), os.path.join("configs", "RoboPushing", "gripper_block_option.yaml")]
obstacle_config = os.path.join("configs", "RoboPushing", "obstacles.yaml")
