import numpy as np

ranges = {
	"Action": [np.array([0]).astype(np.float64), np.array([3]).astype(np.float64)],
	"Paddle": [np.array([71.9, 7.5, 0, 0, 0.9]).astype(np.float64), np.array([72.1, 75.5, 0, 0, 1.1]).astype(np.float64)],
	"Ball": [np.array([0, 0, -2, -1, 0.9]).astype(np.float64), np.array([84, 84, 2, 1, 1.1]).astype(np.float64)],
	"Block": [np.array([10, 0, 0, 0, -1]).astype(np.float64), np.array([58, 84, 0, 0, 1]).astype(np.float64)],
	"Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
	"Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
}

dynamics = {
    "Action": [np.array([-3]).astype(np.float64), np.array([3]).astype(np.float64)],
    "Paddle": [np.array([0, -2, 0, 0, 0]).astype(np.float64), np.array([0, 2, 0, 0, 0]).astype(np.float64)],
    "Ball": [np.array([-2, -1, -4, -2, 0]).astype(np.float64), np.array([2, 1, 4, 2, 0]).astype(np.float64)],
    "Block": [np.array([0, 0, 0, 0, -1]).astype(np.float64), np.array([0, 0, 0, 0, 1]).astype(np.float64)],
    "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
    "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
}

ranges_fixed = {
    "Action": [np.array([0]).astype(np.float64), np.array([3]).astype(np.float64)],
    "Paddle": [np.array([0, 0, -2, -1, -1]).astype(np.float64), np.array([84, 84, 2, 1, 1]).astype(np.float64)],
    "Ball": [np.array([0, 0, -2, -1, -1]).astype(np.float64), np.array([84, 84, 2, 1, 1]).astype(np.float64)],
    "Block": [np.array([0, 0, -2, -1, -1]).astype(np.float64), np.array([84, 84, 2, 1, 1]).astype(np.float64)],
    "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
    "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
}

dynamics_fixed = {
    "Action": [np.array([-3]).astype(np.float64), np.array([3]).astype(np.float64)],
    "Paddle": [np.array([-2, -2, -4, -2, -1]).astype(np.float64), np.array([2, 2, 4, 2, 1]).astype(np.float64)],
    "Ball": [np.array([-2, -2, -4, -2, -1]).astype(np.float64), np.array([2, 2, 4, 2, 1]).astype(np.float64)],
    "Block": [np.array([-2, -2, -4, -2, -1]).astype(np.float64), np.array([2, 2, 4, 2, 1]).astype(np.float64)],
    "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
    "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
}



position_masks = {
    "Action": np.array([0]),
    "Paddle": np.array([1,1,0,0,0]),
    "Ball": np.array([1,1,0,0,0]),
    "Block": np.array([1,1,0,0,0]),
    "Done": np.array([0]),
    "Reward": np.array([0]),
}

def get_instanced(num_rows, num_columns, random_exist, is_big_block):
    instanced = {
        "Action": 1,
        "Paddle": 1,
        "Ball": 1,
        "Block": 1 if is_big_block else (num_rows * num_columns if random_exist <= 0 else random_exist),
        "Done": 1,
        "Reward": 1,
    }
    return instanced


# var_form, num_rows, num_columns, hit_reset, negative_mode, bounce_cost, bounce_reset, completion_reward, timeout_penalty, drop_stopping
# var_form, num_rows, num_columns, max_block_height, hit_reset, negative_mode, random_exist, bounce_cost, bounce_reset, completion_reward, timeout_penalty, drop_stopping
breakout_variants = {"default": (0,5, 20,4, -1, "", -1, 0,0, 0,-10, False),
                     "drop_stopping": (0,5, 20,4, -1, "", -1, 0,0, 0,-10, True),
                     "row":  (0,1,10,4,-1,"", -1, 0,0, 0,-10, False),
                     "small": (0,2,10,4,-1,"", -1, 0,0, 0,-10, False), 
                    "row_nobreak": (0,1,10,4,10,"", -1, 0,0, 0,-1, False), 
                    "small_nobreak": (0,2,10,4,15,"", -1, 0,0, 0,-1, False),
                    "full_nobreak": (0,5,20,4,115,"", -1, 0,0, 0,-1, False),
                    "big_block": (1,1,1,20,-1,"", -1,0,0, 10,-10, True),
                    "single_block": (1,1,1,4,-1,"", -1,-1,0, 0,-10, True),
                    "negative_split_full": (0,5,20,4,75,"side", -1,0,0, 0,-20, False),
                    "negative_split_small": (0,2,10,4,15,"side", -1,0,0, 0,-20, False),
                    "negative_split_row": (0,1,10,4,5,"side", -1,0,0, 0,-20, False),
                    "negative_center_full": (0,5,20,4,75,"center", -1,0,0, 0,-20, False),
                    "negative_center_small": (0,2,10,4,15,"center", -1,0,0, 0,-10, False),
                    "negative_center_row": (0,1,10,4,10,"center", -1,0,0, 0,-10, False),
                    "negative_edge_full": (0,5,20,4,75,"edge", -1,0,0, 0,-10, False),
                    "negative_edge_small": (0,2,10,4,15,"edge", -1,0,0, 0,-10, False),
                    "negative_edge_row": (0,1,10,4,10,"edge", -1,0,0, 0,-10, False),
                    "negative_checker_row": (0,1,10,4,10,"checker", -1,0,0, 0,-10, False),
                    "negative_rand_row": (0,1,10,4,5,"rand", -1,0,0, 0, -10, False),
                    "negative_double": (1,1,1,4,-1,"rand", -1,-1,0, 0, -10, False),
                    "negative_multi": (1,1,1,4,-1,"rand", -1,-1,0, 0, -10, False),
                    "negative_top_full": (0,5,20,4,40,"top", -1,0,0,0, -120, False),
                    "negative_top_small": (0,2,10,4,7,"top", -1,0,0,0, -30, False),
                    "breakout_priority_small": (0,2,10,4,-1,"", -1,-2,0, 1, -30, False),
                    "breakout_priority_medium": (0,3,10,4,-1,"", -1,-2,0, 5, -75, False),
                    "breakout_priority_large": (0,4,15,4,-1,"", -1,-1,0, 20, -100, False),
                    "breakout_priority_full": (0,5,20,4,-1,"", -1,-2,0, 20, -120, False),
                    "edges_full": (2,5,20,4,-1,"", -1,-1,0, 20, -120, False),
                    "edges_small": (2,2,10,4,-1,"", -1,-1,0, 1, -30, False),
                    "center_small": (3,2,10,4,-1,"", -1,-1,0, 1,-30, False),
                    "center_medium": (3,3,15,4,-1,"", -1,-1,0, 5,-75, False),
                    "center_large": (3,4,15,4,-1,"", -1,-1, 0,20,-100, False),
                    "center_full": (3,5,20,4,-1,"", -1,-2, 0,20,-120, False),
                    "harden_single": (4,5,12,4,-1,"", -1,-1,10,0,-10, True),
                    "proximity": (0,4,15,60,0,"", -1, 0,0, 0,-10, True)}
def adjacent(i,j):
    return [(i-1,j-1), (i, j-1), (i, j+1), (i-1, j), (i-1,j+1),
            (i, j-2), (i-1, j-2), (i-2, j-2), (i-2, j-1), (i-2, j), (i-2, j+1), (i-2, j+2), (i-1, j+2), (i-2, j+2)]

ball_vels = [np.array([-1.,-1.]).astype(int), np.array([-2.,-1.]).astype(int), np.array([-2.,1.]).astype(int), np.array([-1.,1.]).astype(int)]
