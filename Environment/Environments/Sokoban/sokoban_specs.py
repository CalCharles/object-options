import numpy as np
RNG = 30

def generate_specs_fixed(rows, columns, num_obstacles, num_blocks, num_targets):
    ranges = { # TODO: we might need to normalize by RNG rather the rows, columns if we want to be bound agnostic
        "Action": [np.array([0]).astype(np.float64), np.array([3]).astype(np.float64)],
        "Pusher": [np.array([0, 0]).astype(np.float64), np.array([RNG,RNG]).astype(np.float64)],
        "Obstacle": [np.array([0, 0]).astype(np.float64), np.array([RNG,RNG]).astype(np.float64)],
        "Block": [np.array([0, 0]).astype(np.float64), np.array([RNG,RNG]).astype(np.float64)],
        "Target": [np.array([0, 0, 0]).astype(np.float64), np.array([RNG,RNG, 1]).astype(np.float64)],
        "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
        "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
    }

    dynamics = {
        "Action": [np.array([-3]).astype(np.float64), np.array([3]).astype(np.float64)],
        "Pusher": [np.array([-1, -1]).astype(np.float64), np.array([1, 1]).astype(np.float64)],
        "Obstacle": [np.array([-1, -1]).astype(np.float64), np.array([1, 1]).astype(np.float64)],
        "Block": [np.array([-1, -1]).astype(np.float64), np.array([1, 1]).astype(np.float64)],
        "Target": [np.array([-1, -1, -1]).astype(np.float64), np.array([1, 1, 1]).astype(np.float64)],
        "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
        "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
    }


    position_masks = {
        "Action": np.array([0]),
        "Pusher": np.array([1,1]),
        "Obstacle": np.array([1,1]),
        "Block": np.array([1,1]),
        "Target": np.array([1,1,0]),
        "Done": np.array([0]),
        "Reward": np.array([0]),
    }

    instanced = {
        "Action": 1,
        "Pusher": 1,
        "Obstacle": num_obstacles,
        "Block": num_blocks,
        "Target": num_targets,
        "Done": 1,
        "Reward": 1,
    }
    return ranges, dynamics, position_masks, instanced


def generate_specs(rows, columns, num_obstacles, num_blocks, num_targets):
    ranges = { # TODO: we might need to normalize by RNG rather the rows, columns if we want to be bound agnostic
    	"Action": [np.array([0]).astype(np.float64), np.array([3]).astype(np.float64)],
    	"Pusher": [np.array([0, 0]).astype(np.float64), np.array([rows - 1, columns - 1]).astype(np.float64)],
    	"Obstacle": [np.array([0, 0]).astype(np.float64), np.array([rows - 1, columns - 1]).astype(np.float64)],
        "Block": [np.array([0, 0]).astype(np.float64), np.array([rows - 1, columns - 1]).astype(np.float64)],
        "Target": [np.array([0, 0, 0]).astype(np.float64), np.array([rows - 1, columns - 1, 1]).astype(np.float64)],
    	"Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
    	"Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
    }

    dynamics = {
        "Action": [np.array([-3]).astype(np.float64), np.array([3]).astype(np.float64)],
        "Pusher": [np.array([-1, -1]).astype(np.float64), np.array([1, 1]).astype(np.float64)],
        "Obstacle": [np.array([-1, -1]).astype(np.float64), np.array([1, 1]).astype(np.float64)],
        "Block": [np.array([-1, -1]).astype(np.float64), np.array([1, 1]).astype(np.float64)],
        "Target": [np.array([0, 0, -1]).astype(np.float64), np.array([0.01, 0.01, 1]).astype(np.float64)],
        "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
        "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
    }


    position_masks = {
        "Action": np.array([0]),
        "Pusher": np.array([1,1]),
        "Obstacle": np.array([1,1]),
        "Block": np.array([1,1]),
        "Target": np.array([1,1,0]),
        "Done": np.array([0]),
        "Reward": np.array([0]),
    }

    instanced = {
        "Action": 1,
        "Pusher": 1,
        "Obstacle": num_obstacles,
        "Block": num_blocks,
        "Target": num_targets,
        "Done": 1,
        "Reward": 1,
    }
    return ranges, dynamics, position_masks, instanced


# num_rows, num_columns, num_blocks, num_obstacles, num_targets, step_limit, preset (load directory for human-created puzzles)
sokoban_variants = {"single": (10, 10, 1, 10, 1, 100, ""), 
                    "default":(12, 12, 5, 20, 5, 200, ""),
                    "small_single":(5, 5, 1, 1, 1, 50, ""),
                    "small":(5, 5, 2, 2, 2, 50, ""),
                    "small_obs":(10, 10, 1, 20, 1, 50, ""),
                    "few_obs":(10, 10, 1, 10, 1, 50, ""),
                    "no_obs":(10, 10, 1, 0, 1, 50, "")  }
