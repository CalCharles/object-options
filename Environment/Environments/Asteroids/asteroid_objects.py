import numpy as np

class Object:
    def __init__(self):
        self.state = None
        self.interaction_trace = list()

class Action(Object):
    def __init__(self):
        super().__init__()
        self.name = "Action"
        self.action = 0
        self.interaction_trace = list()

    def step(self, action):
        self.action = action

    def get_state(self):
        return np.array([self.action])

def rotation_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

cardinal_angles = np.array([0, np.pi/2, np.pi, 3* np.pi / 2])

class Ship(Object):
    def __init__(self, pos, angle, speed, movement_type, size=3):
        super().__init__()
        self.name="Ship"
        self.pos = pos
        self.angle = angle
        self.size = size # the length from the center to the tips
        self.rotation = np.array([np.sin(angle), np.cos(angle)])
        self.speed, self.rotation_speed = speed
        self.movement_type = movement_type
        self.pos_change = np.zeros(pos.shape)
        self.angle_change = np.zeros((1,))
        self.fire_attempt = 0
        self.laser_exist = 0
        self.update_tips()

    def get_coord_change(self, speed, angle = [[1], [0]]):
        return np.dot(rotation_matrix(self.angle), np.array(angle) * speed).squeeze()

    def intersect(self, asteroid):
        return (intersect_general(asteroid.pos, asteroid.size, self.tip, end=self.right)
        or intersect_general(asteroid.pos, asteroid.size, self.right, end=self.left)
        or intersect_general(asteroid.pos, asteroid.size, self.left, end=self.tip))

    def step(self, action):
        self.pos_change = np.zeros(self.pos.shape)
        self.angle_change = np.zeros(self.angle.shape)
        action.interaction_trace = [self.name]
        self.interaction_trace = [action.name]
        if self.movement_type == "coordinate":
            if action.action == 1: self.pos_change[0] -= self.speed
            if action.action == 2: self.pos_change[0] += self.speed
            if action.action == 3: self.pos_change[1] -= self.speed
            if action.action == 4: self.pos_change[1] += self.speed
        if self.movement_type == "coordinate_turn":
            self.angle_change = self.angle
            if action.action == 1: 
                self.pos_change[1] += self.speed
                self.angle_change = cardinal_angles[3]
            if action.action == 2: 
                self.pos_change[1] -= self.speed
                self.angle_change = cardinal_angles[1]
            if action.action == 3: 
                self.pos_change[0] -= self.speed
                self.angle_change = cardinal_angles[2]
            if action.action == 4: 
                self.pos_change[0] += self.speed
                self.angle_change = cardinal_angles[0]
        elif self.movement_type == "angle":
            if action.action == 1: self.pos_change += self.get_coord_change(-self.speed)
            if action.action == 2: self.pos_change -= self.get_coord_change(-self.speed)
            if action.action == 3: self.angle_change += self.rotation_speed
            if action.action == 4: self.angle_change -= self.rotation_speed
        elif self.movement_type == "row":
            if action.action == 3: self.pos_change[1] -= self.speed
            if action.action == 4: self.pos_change[1] += self.speed
        self.fire_attempt = 1 if action.action == 5 else 0 

    def update_tips(self):
        self.tip = self.pos + self.get_coord_change(self.size).squeeze()
        self.right = self.pos + self.get_coord_change(self.size, angle=[[-1], [0.5]]).squeeze()
        self.left = self.pos + self.get_coord_change(self.size, angle=[[-1], [-0.5]]).squeeze()

    def update(self, laser):
        self.laser_exist = float(self.fire_attempt and laser.exist == 0)
        self.pos = np.clip(self.pos + self.pos_change, 0, 84)
        if self.movement_type == "coordinate_turn": newangle = self.angle_change
        else:
            newangle = self.angle + self.angle_change
            if newangle.squeeze() > 2 * np.pi: 
                newangle = newangle - 2 * np.pi
            if newangle.squeeze() < 0:
                newangle = 2 * np.pi + newangle
        self.angle = newangle
        self.update_tips()

    def get_state(self):
        # if self.movement_type == "coordinate_turn": return np.array(self.pos.tolist() + [self.angle])
        return np.array(self.pos.tolist() + [np.sin(self.angle)] + [np.cos(self.angle)] + [self.laser_exist])

def intersect_general(center,size, start, vel=None, end=None):
    if vel is None:
        vel = end - start
    if np.linalg.norm(center - start) > size + np.linalg.norm(vel): 
        return False

    # check if endpoints in circle
    A = np.array([center[0], center[1]])
    if np.linalg.norm(start - A) < size:
        return True
    A = np.array([center[0] + vel[0], center[1] + vel[1]])
    if np.linalg.norm(start - A) < size:
        return True


    # check if line segment intersects circle
    A = vel[0] * vel[0] + vel[1] * vel[1]
    B = 2 * (vel[1] * (start[1] - center[1]) + vel[0] * (start[0] - center[0]))
    C = (start[1] - center[1]) ** 2 + (start[0] - center[0]) ** 2 - size ** 2
    
    det = (B ** 2 - 4 * A * C)
    if ((A <= 0.0000001) or (det < 0)):
        return False
    elif (det == 0):
        t = -B / (2 * A)
        return 0< t < 1
    else:
        t1 = (-B + np.sqrt(det)) / (2 * A)
        t2 = (-B - np.sqrt(det)) / (2 * A)
        return 0 < t1 < 1 or 0 < t2 < 1

class Laser(Object):
    def __init__(self, pos, speed, exist, size = 3):
        super().__init__()
        self.name="Laser"
        self.pos = pos
        self.vel = np.zeros((2,))
        self.exist = exist
        self.pos_change = np.zeros((2,))
        self.vel_change = np.zeros((2,)) # sets vel  to the value in vel_change
        self.exist_change = 0
        self.speed = speed
        self.size = size
        self.update_bottom_top()


    def step(self, ship, action):
        self.pos_change = np.zeros((2,))
        self.vel_change = np.zeros((2,)) # sets vel  to the value in vel_change
        self.exist_change = self.exist
        if action.action == 5 and self.exist == 0: # firing the laser 
            self.interaction_trace = [ship.name, action.name]
            ship.interaction_trace += [self.name]
            action.interaction_trace += [self.name]
            self.exist_change = 1
            self.vel_change = ship.get_coord_change(self.speed)
            self.pos_change = ship.tip
            return 1
        else:
            self.pos_change = self.pos + self.vel
            self.vel_change = self.vel
            self.exist_change = self.exist
            return 0

    def update(self):
        self.pos = self.pos_change
        self.vel = self.vel_change
        self.exist = self.exist_change
        if 0 > self.pos[0] or self.pos[0] > 84 or 0 > self.pos[1] or self.pos[1] > 84:
            self.exist = 0
            self.pos = np.zeros((2,))
            self.vel = np.zeros((2,))
        self.update_bottom_top()

    def update_bottom_top(self):
        self.bottom = self.pos.copy().squeeze()
        self.top = self.pos + (self.vel / np.linalg.norm(self.vel) * self.size).squeeze() if np.linalg.norm(self.vel) > 0 else self.pos.squeeze()

    def intersect(self, asteroid):
        return intersect_general(asteroid.pos, asteroid.size, self.pos, self.vel)

    def get_state(self):
        return np.array(self.pos.tolist() + self.vel.tolist() + [self.exist])

class Asteroid(Object):
    def __init__(self, pos, vel, exist, size, idx):
        super().__init__()
        self.name = "Asteroid"+ str(idx)
        self.pos = pos
        self.vel = vel
        self.exist = exist
        self.pos_change = np.zeros((2,))
        self.vel_change = np.zeros((2,)) # sets vel  to the value in vel_change
        self.exist_change = self.exist
        self.size = size

    def step(self, laser):
        self.pos_change = np.zeros((2,))
        self.vel_change = np.zeros((2,)) # sets vel  to the value in vel_change
        self.exist_change = self.exist
        intersect = laser.intersect(self) if laser.exist else False
        if self.exist == 0 or intersect: # asteroids hit or nonexisten are stationary
            if intersect and self.exist != 0: 
                self.interaction_trace = [laser.name]
                laser.interaction_trace += [self.name]
                laser.exist_change = 0
            self.vel_change = np.zeros((2,))
            self.pos_change = self.pos
            self.exist_change = 0
        else:
            self.pos_change = self.pos + self.vel
            self.vel_change = self.vel
            self.exist_change = self.exist
            # wall bouncing behavior
            if 0 > self.pos_change[0] or self.pos_change[0] > 84:
                self.vel_change[0] = - self.vel[0]
            if 0 > self.pos_change[1] or self.pos_change[1] > 84:
                self.vel_change[1] = - self.vel[1]

    def update(self):
        self.pos = self.pos_change
        self.vel = self.vel_change
        hit =  1 if self.exist_change == 0 and self.exist == 1 else 0
        self.exist = self.exist_change
        return hit

    def get_state(self):
        return np.array(self.pos.tolist() + self.vel.tolist() + [self.size] + [self.exist])