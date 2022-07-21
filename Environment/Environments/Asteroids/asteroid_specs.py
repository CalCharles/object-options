import numpy as np

def generate_specs(asteroid_speed, ship_speed, laser_speed, asteroid_size, asteroid_size_variance, num_asteroids):
    ranges = {
    	"Action": [np.array([0]).astype(np.float64), np.array([5]).astype(np.float64)],
    	"Ship": [np.array([0, 0, -1, -1]).astype(np.float64), np.array([84, 84, 1, 1]).astype(np.float64)],
    	"Laser": [np.array([0, 0, -laser_speed, -laser_speed, 0]).astype(np.float64), np.array([84, 84, laser_speed, laser_speed, 1]).astype(np.float64)],
    	"Asteroid": [np.array([0, 0, -asteroid_speed, -asteroid_speed, asteroid_size - asteroid_size_variance, 0]).astype(np.float64), np.array([84, 84, asteroid_speed, asteroid_speed, asteroid_size + asteroid_size_variance, 1]).astype(np.float64)],
    	"Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
    	"Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
    }

    dynamics = {
        "Action": [np.array([-5]).astype(np.float64), np.array([5]).astype(np.float64)],
        "Ship": [np.array([-ship_speed, -ship_speed, -0.5, -0.5]).astype(np.float64), np.array([ship_speed, ship_speed, 0.5, 0.5]).astype(np.float64)],
        "Laser": [np.array([-laser_speed, -laser_speed, 0, 0, -1]).astype(np.float64), np.array([laser_speed, laser_speed, 0, 0, 1]).astype(np.float64)],
        "Asteroid": [np.array([-asteroid_speed, -asteroid_speed, -asteroid_speed, -asteroid_speed, 0, 0]).astype(np.float64), np.array([asteroid_speed, asteroid_speed, asteroid_speed, asteroid_speed, 0.01, 0.01]).astype(np.float64)],
        "Done": [np.array([0]).astype(np.float64), np.array([1]).astype(np.float64)],
        "Reward": [np.array([-100]).astype(np.float64), np.array([100]).astype(np.float64)]
    }


    position_masks = {
        "Action": np.array([0]),
        "Ship": np.array([1,1,0,0]),
        "Laser": np.array([1,1,0,0,0]),
        "Asteroid": np.array([1,1,0,0,0,0]),
        "Done": np.array([0]),
        "Reward": np.array([0]),
    }

    instanced = {
        "Action": 1,
        "Ship": 1,
        "Laser": 1,
        "Asteroid": num_asteroids,
        "Done": 1,
        "Reward": 1,
    }
    return ranges, dynamics, position_masks, instanced


# num_asteroids, asteroid_size, asteroid_speed, asteroid_size_variance asteroid_variance, ship_variance, ship_speed, movement_type, laser_speed, hit reward, shot penalty, crash penalty, completion reward

asteroid_variants = {"single": (1,5,2,1,1,1,(2, np.pi/8),"angle", 4,2,-1,-10,10),
                    "default":(10,4,4,3,1,1,(2,np.pi/8),"angle", 7,2,-1,-10,10) }
