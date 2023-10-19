import numpy as np

ranges = {
	"Action": [np.array([-0.1,-0.1,-0.1,-0.1,0]).astype(np.float64), np.array([0.1,0.1,0.1,0.1,3]).astype(np.float64)],
	"Paddle": [np.array([71.9, 7.5, 0, 0, 0.9]).astype(np.float64), np.array([72.1, 75.5, 0, 0, 1.1]).astype(np.float64)],
	"Ball": [np.array([0, 0, -2, -1, 0.9]).astype(np.float64), np.array([84, 84, 2, 1, 1.1]).astype(np.float64)],
	"Block": [np.array([10, 0, 0, 0, -1]).astype(np.float64), np.array([58, 84, 0, 0, 1]).astype(np.float64)],
	"Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
	"Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
}

dynamics = {
    "Action": [np.array([-0.1,-0.1,-0.1,-0.1,-3]).astype(np.float64), np.array([0.1,0.1,0.1,0.1,3]).astype(np.float64)],
    "Paddle": [np.array([0, -2, 0, 0, 0]).astype(np.float64), np.array([0, 2, 0, 0, 0]).astype(np.float64)],
    "Ball": [np.array([-2, -1, -4, -2, 0]).astype(np.float64), np.array([2, 1, 4, 2, 0]).astype(np.float64)],
    "Block": [np.array([0, 0, 0, 0, -1]).astype(np.float64), np.array([0, 0, 0, 0, 1]).astype(np.float64)],
    "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
    "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
}

ranges_fixed = {
    "Action": [np.array([-0.1,-0.1,-0.1,-0.1,0]).astype(np.float64), np.array([0.1,0.1,0.1,0.1,3]).astype(np.float64)],
    "Paddle": [np.array([0, 0, -2, -1, -1]).astype(np.float64), np.array([84, 84, 2, 1, 1]).astype(np.float64)],
    "Ball": [np.array([0, 0, -2, -1, -1]).astype(np.float64), np.array([84, 84, 2, 1, 1]).astype(np.float64)],
    "Block": [np.array([0, 0, -2, -1, -1]).astype(np.float64), np.array([84, 84, 2, 1, 1]).astype(np.float64)],
    "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
    "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
}

dynamics_fixed = {
    "Action": [np.array([-0.1,-0.1,-0.1,-0.1,-3]).astype(np.float64), np.array([0.1,0.1,0.1,0.1,3]).astype(np.float64)],
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
breakout_variants = {"default": (0,5, 20,4,-1, -1, "", -1, 0,0, 0,0, False, False),
                     "drop_stopping": (0,5, 20,4,-1, -1, "", -1, 0,0, 0,-10, True, False),
                     "drop_stopping_no_penalty": (0,5, 20,4,-1, -1, "", -1, 0,0, 0,0, True, False),
                     "row":  (0,1,10,4,-1,-1,"", -1, 0,0, 0,-10, False, False),
                     "small": (0,2,10,4,-1,-1,"", -1, 0,0, 0,-10, False, False), 
                    "row_nobreak": (0,1,10,4,-1,10,"", -1, 0,0, 0,-1, False, False), 
                    "small_nobreak": (0,2,10,4,-1,15,"", -1, 0,0, 0,-1, False, False),
                    "full_nobreak": (0,5,20,4,-1,115,"", -1, 0,0, 0,-1, False, False),
                    "big_block": (1,1,1,20,-1,-1,"", -1,0,0, 0,-10, True, False),
                    "single_block": (1,1,1,4,-1,-1,"", -1,-1,0, 0,-10, True, False),
                    "negative_split_full": (0,5,20,4,-1,75,"side", -1,0,0, 0,-20, False, False),
                    "negative_split_small": (0,2,10,4,-1,15,"side", -1,0,0, 0,-20, False, False),
                    "negative_split_row": (0,1,10,4,-1,5,"side", -1,0,0, 0,-20, False, False),
                    "negative_center_full": (0,5,20,4,-1,75,"center", -1,0,0, 0,-20, False, False),
                    "negative_center_small": (0,2,10,4,-1,15,"center", -1,0,0, 0,-10, False, False),
                    "negative_center_row": (0,1,10,4,-1,10,"center", -1,0,0, 0,-10, False, False),
                    "negative_edge_full": (0,5,20,4,-1,75,"edge", -1,0,0, 0,-10, False, False),
                    "negative_edge_small": (0,2,10,4,-1,15,"edge", -1,0,0, 0,-10, False, False),
                    "negative_edge_row": (0,1,10,4,-1,10,"edge", -1,0,0, 0,-10, False, False),
                    "negative_checker_row": (0,1,10,4,-1,10,"checker", -1,0,0, 0,-10, False, False),
                    "negative_rand_row": (0,1,10,4,-1,5,"rand", -1,0,0, 0, -10, False, False),
                    "negative_double": (1,1,1,4,-1,-1,"rand", -1,-1,0, 0, -10, False, False),
                    "negative_multi": (1,1,1,4,-1,-1,"rand", -1,-1,0, 0, -10, False, False),
                    "negative_top_full": (0,5,20,4,-1,40,"top", -1,0,0,0, -120, False, False),
                    "negative_top_small": (0,2,10,4,-1,7,"top", -1,0,0,0, -30, False, False),
                    "breakout_priority_small": (0,2,10,4,-1,-1,"", -1,-2,0, 1, -30, False, False),
                    "breakout_priority_medium": (0,3,10,4,-1,-1,"", -1,-2,0, 5, -75, False, False),
                    "breakout_priority_large": (0,4,15,4,-1,-1,"", -1,-1,0, 20, -100, False, False),
                    "breakout_priority_full": (0,5,20,4,-1,-1,"", -1,-2,0, 20, -120, False, False),
                    "edges_full": (2,5,20,4,-1,-1,"", -1,-1,0, 20, -120, False, False),
                    "edges_small": (2,2,10,4,-1,-1,"", -1,-1,0, 1, -30, False, False),
                    "center_small": (3,2,10,4,-1,-1,"", -1,-1,0, 1,-30, False, False),
                    "center_medium": (3,3,15,4,-1,-1,"", -1,-1,0, 5,-10, False, False),
                    "center_large": (3,4,15,4,-1,-1,"", -1,-1, 0,20,-100, False, False),
                    "center_full": (3,5,20,4,-1,-1,"", -1,-2, 0,20,-120, False, False),
                    "harden_single": (4,5,12,4,-1,-1,"", -1,-1,10,0,-2, True, False),
                    "rand_small": (0,10,12,4,2,-1,"", 10,0,0,0,0, True, True),
                    "proximity": (0,4,15,20,-1,60,"", -1, 0,0, 0,-10, True, False)}

# var_form, num_rows, num_columns, max_block_height, min_block_height, hit_reset, 
# negative_mode, random_exist, bounce_cost, bounce_reset, completion_reward, timeout_penalty, drop_stopping, drop_reset

def adjacent(i,j):
    return [(i-1,j-1), (i, j-1), (i, j+1), (i-1, j), (i-1,j+1),
            (i, j-2), (i-1, j-2), (i-2, j-2), (i-2, j-1), (i-2, j), (i-2, j+1), (i-2, j+2), (i-1, j+2), (i-2, j+2)]

ball_vels = [np.array([-1.,-1.]).astype(int), np.array([-2.,-1.]).astype(int), np.array([-2.,1.]).astype(int), np.array([-1.,1.]).astype(int)]
