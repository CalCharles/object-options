# Objects

import numpy as np

paddle_width = 7
paddle_velocity = 2

class Object():

    def __init__(self, pos, attribute):
        self.pos = pos
        self.vel = np.zeros(pos.shape).astype(np.int64)
        self.width = 0
        self.height = 0
        self.attribute = attribute
        self.interaction_trace = list()

    def getBB(self):
        return [self.pos[0], self.pos[1], self.pos[0] + self.height, self.pos[1] + self.width]

    def getMidpoint(self):
        return [self.pos[0] + (self.height / 2), self.pos[1]  + (self.width/2)]

    def getPos(self, mid):
        return [int(mid[0] - (self.height / 2)), int(mid[1]  - (self.width/2))]

    def getAttribute(self):
        return self.attribute

    def getState(self):
        return self.getMidpoint() + [self.attribute]

    def interact(self, other):
        pass

class animateObject(Object):

    def __init__(self, pos, attribute, vel):
        super(animateObject, self).__init__(pos, attribute)
        self.animate = True
        self.vel = vel
        self.apply_move = True
        self.zero_vel = False

    def move(self):
        # print (self.name, self.pos, self.vel)
        if self.apply_move:
            self.pos += self.vel
        else:
            self.apply_move = True
            self.pos += self.vel
        if self.zero_vel:
            self.vel = np.zeros(self.vel.shape).astype(np.int64)

def intersection(a, b):
    midax, miday = (a.next_pos[1] * 2 + a.width)/ 2, (a.next_pos[0] * 2 + a.height)/ 2
    midbx, midby = (b.pos[1] * 2 + b.width)/ 2, (b.pos[0] * 2 + b.height)/ 2
    # print (midax, miday, midbx, midby)
    return (abs(midax - midbx) * 2 < (a.width + b.width)) and (abs(miday - midby) * 2 < (a.height + b.height))

class Ball(animateObject):
    def __init__(self, pos, attribute, vel, top_reset = False, hard_mode=False):
        super(Ball, self).__init__(pos, attribute, vel)
        self.width = 2
        self.height = 2
        self.name = "Ball"
        self.losses = 0
        self.paddlehits = 0
        self.reset_seed = -1
        self.top_reset = top_reset
        self.needs_ball_reset = False
        self.hard_mode = hard_mode
        self.hit_trace = list()

        # indicators for if the ball hit a particular object
        self.flipped = False
        self.bottom_wall = False
        self.top_wall = False
        self.block = False
        self.block_id = None
        self.paddle = False

        # MODE 1 # only odd lengths valid, prefer 7,11,15, etc. 
        self.paddle_interact = dict()
        angles = [np.array([-1, -1]), np.array([-2, -1]), np.array([-2, 1]), np.array([-1, 1])]
        total_number = paddle_width+3
        if total_number % 4 == 0:
            num_per = total_number // 4
        else:
            num_per = (total_number - 2) // 4
        for i in range(-2,paddle_width+1):
            if i == -2 and total_number % 4 != 0:
                self.paddle_interact[i] = angles[0]
            elif i == paddle_width and total_number % 4 != 0:
                self.paddle_interact[i] = angles[3]
            else:
                self.paddle_interact[i] = angles[(i + 1) // num_per]

        self.reset_pos()

    def clear_hits(self):
        self.paddle, self.bottom_wall, self.top_wall, self.block, self.flipped = False, False, False, False, False
        self.hit_trace = list()

    def reset_pos(self):
        self.vel = np.array([np.random.randint(1,2), np.random.choice([-1,1])])
        self.pos = np.array([np.random.randint(43, 46), np.random.randint(14, 70)])

    def interact(self, other):
        '''
        interaction computed before movement
        '''
        self.next_pos = self.pos + self.vel
        # print(self.apply_move, self.vel)
        if intersection(self, other) and self.apply_move:
            if other.name == "Paddle":
                rel_x = self.next_pos[1] - other.pos[1]
                self.vel = self.paddle_interact[int(rel_x)].copy()
                self.apply_move = False
                self.paddlehits += 1
                self.paddle = True
            elif other.name.find("SideWall") != -1:
                self.vel = np.array([self.vel[0], -self.vel[1]])
                self.apply_move = False
            elif other.name.find("TopWall") != -1:
                self.top_wall = True
                if self.top_reset: # basically only the big block domain
                    print(self.pos, self.vel, "topdrop", intersection(self,other))
                    self.losses += 1
                else:
                    self.vel = np.array([-self.vel[0], self.vel[1]])
                    self.apply_move = False
            elif other.name.find("BottomWall") != -1:
                self.bottom_wall = True
                self.losses += 1
                print(self.pos, self.vel, "dropped", intersection(self,other))
                # position reset handled by breakout screen
            elif other.name.find("Block") != -1 and other.attribute != 0:
                self.block = True
                self.block_id = other
                rel_x = self.pos[1] - other.pos[1]
                rel_y = self.pos[0] - other.pos[0]
                print(rel_x, rel_y, self.vel, self.pos[0], other.pos[0], other.height, other.name, intersection(self, other))
                if not self.hard_mode or other.attribute != -1: other.attribute = 0
                next_vel = self.vel
                # if (rel_y == -2 or rel_y == -3 or rel_y == 3 or rel_y == 2) and not self.flipped:
                # if rel_y == -2 or rel_y == -3 or rel_y == 3 or rel_y == 2:
                if rel_y <= -1 or 1 <= rel_y:
                    next_vel[0] = - self.vel[0]
                    self.flipped = True
                if other.attribute == -1 and self.hard_mode:
                    if -1 > rel_y > -self.height or other.height > rel_y > 1:
                        next_vel[1] = - self.vel[1]

                if other.attribute == -1 and self.hard_mode:
                    if len(self.hit_trace) > 0:
                        print("push_out", np.sign(next_vel[1]), np.sign(rel_x))
                        trace_rel = self.pos[0] - self.hit_trace[0].pos[0]
                        if trace_rel != rel_y:
                            if np.sign(next_vel[1]) == np.sign(rel_x):
                                next_vel[1] = - self.vel[1]
                self.vel = np.array(next_vel)
                self.apply_move = False
                self.hit_trace.append(other)
                other.interaction_trace.append(self.name)
                # self.nohit_delay = 2
            self.interaction_trace.append(other.name)

class Paddle(animateObject):
    def __init__(self, pos, attribute, vel):
        super(Paddle, self).__init__(pos, attribute, vel)
        self.width = paddle_width
        self.height = 2
        self.name = "Paddle"
        self.nowall = False
        self.zero_vel = True

    def interact(self, other):
        if other.name == "Action":
            self.interaction_trace.append(other.name)
            other.interaction_trace.append(self.name) # action.interaction(paddle) will never be called because action is not an animateobject
            if other.attribute == 0 or other.attribute == 1:
                self.vel = np.array([0,0], dtype=np.int64)
            elif other.attribute == 2:
                if self.pos[1] == 4:
                    if self.nowall:
                        self.pos = np.array([0,68])
                    self.vel = np.array([0,0])
                    self.interaction_trace.append("LeftSideWall")
                else:
                    self.vel = np.array([0,-paddle_velocity])
                # self.vel = np.array([0,-2])
            elif other.attribute == 3:
                if self.pos[1] >= 72:
                    if self.nowall:
                        self.pos = np.array([0,8])
                    self.interaction_trace.append("RightSideWall")
                    self.vel = np.array([0,0])
                else:
                    self.vel = np.array([0,paddle_velocity])
                # self.vel = np.array([0,2])


class Wall(Object):
    def __init__(self, pos, attribute, side):
        super(Wall, self).__init__(pos, attribute)
        if side == "Top":
            self.width = 84
            self.height = 4
        elif side == "RightSide":
            self.width = 4
            self.height = 84
        elif side == "LeftSide":
            self.width = 4
            self.height = 84
        elif side == "Bottom":
            self.width = 84
            self.height = 4
        self.name = side + "Wall"

class Block(Object):
    def __init__(self, pos, attribute, index, index2d, width= 3, height=2,size=1):
        super(Block, self).__init__(pos, attribute)
        self.width = width * size
        self.height = height * size
        if index >= 0:
            self.name = "Block" + str(index)
        else:
            self.name = "Block"
        self.index2D = index2d

    # def interact(self, other):
    #     if other.name == "Ball":
    #         if intersection(other, self):
    #             print(self.name, self.pos, other.pos)
    #             self.attribute = 0

class Action(Object):
    def __init__(self, pos, attribute):
        super(Action, self).__init__(pos, attribute)
        self.width = 0
        self.height = 0
        self.name = "Action"

    def take_action(self, action):
        self.attribute = action

    def interact (self, other):
        if other.name == "Paddle": self.interaction_trace.append(other.name)