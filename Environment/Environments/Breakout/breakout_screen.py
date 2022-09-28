# Screen
import sys, cv2
import numpy as np
from Environment.Environments.Breakout.breakout_objects import *
import imageio as imio
import os, copy
from Environment.environment import Environment
from Environment.Environments.Breakout.breakout_specs import *
from Record.file_management import numpy_factored
from gym import spaces


# default settings for normal variants, args in order: 
# target_mode (1)/edges(2)/center(3), scatter (4), num rows, num_columns, no_breakout (value for hit_reset), negative mode, bounce_count

class Breakout(Environment):
    def __init__(self, frameskip = 1, breakout_variant="default"):
        super(Breakout, self).__init__()
        # breakout specialized parameters are stored in the variant
        self.variant = breakout_variant
        self.self_reset = True

        # environment properties
        self.num_actions = 4 # this must be defined, -1 for continuous. Only needed for primitive actions
        self.name = "Breakout" # required for an environment 
        self.discrete_actions = True
        self.frameskip = 1 # no frameskip

        # spaces
        self.action_shape = (1,)
        self.action_space = spaces.Discrete(self.num_actions) # gym.spaces
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8) # raw space, gym.spaces
        self.seed_counter = -1

        # state components
        self.frame = None # the image generated by the environment
        self.reward = 0
        self.done = False
        self.action = np.zeros(self.action_shape)
        self.extracted_state = None

        # running values
        self.itr = 0
        self.total_score = 0

        # proximity components
        self.position_masks = position_masks

        # factorized state properties
        self.object_names = ["Action", "Paddle", "Ball", "Block", 'Done', "Reward"]
        self.object_sizes = {"Action": 1, "Paddle": 5, "Ball": 5, "Block": 5, 'Done': 1, "Reward": 1}
        self.object_name_dict = dict() # initialized in reset
        self.object_range = ranges
        self.object_dynamics = dynamics

        # asign variant values
        var_form, num_rows, num_columns, max_block_height, hit_reset, negative_mode, random_exist, bounce_cost, bounce_reset, completion_reward, timeout_penalty, drop_stopping = breakout_variants[breakout_variant]
        self.object_instanced = get_instanced(num_rows, num_columns, random_exist, self.variant == "big_block")
        self.target_mode = (var_form == 1)
        if var_form == 2: negative_mode = "hardedge"
        elif var_form == 3: negative_mode = "hardcenter"
        elif var_form == 4: negative_mode = "hardscatter"
        self.negative_mode = negative_mode
        self.no_breakout = hit_reset > 0
        self.hit_reset = hit_reset
        self.timeout_penalty = timeout_penalty
        self.completion_reward = completion_reward
        self.assessment_stat = 0 if self.variant != "proximity" else (0,0)# a measure of performance specific to the variant
        self.drop_stopping = drop_stopping
        self.top_dropping = self.variant == "big_block"
        self.bounce_cost = bounce_cost
        self.bounce_reset = bounce_reset
        self.num_rows = num_rows # must be a factor of 10
        self.num_columns = num_columns # must be a factor of 60
        self.max_block_height = max_block_height
        self.random_exist = random_exist

        # assign dependent values
        self.default_reward = 1
        if self.target_mode: self.num_blocks = 1
        elif negative_mode == "hardscatter": self.num_blocks = 6 # hardcoded for now
        else: self.num_blocks = num_rows * num_columns
        self.block_height = min(self.max_block_height, 10 // self.num_rows)
        self.block_width = 60 // self.num_columns
        self.hard_mode = self.negative_mode[:4] == "hard"

        # reset counters
        self.hit_counter = 0
        self.bounce_counter = 0
        self.since_last_bounce = 0
        self.choices = list()
        self.reset()
        self.safe_distance = self.ball.height + max(self.paddle.height, self.block_height) + 4
        self.low_block = self.block_height * self.num_rows + 22 + self.safe_distance
        self.paddle_height = self.paddle.pos[0]
        self.num_remove = self.get_num(True)
        self.all_names = ["Action", "Paddle", "Ball"] + [b.name for b in self.blocks] + ['Done', "Reward"]
        self.instance_length = len(self.all_names)

    def assign_assessment_stat(self):
        if self.dropped and self.variant != "proximity":
            self.assessment_stat += -1000
        elif self.dropped:
            self.assessment_stat = -1000            
        elif self.variant == "big_block":
            if self.ball.block: self.assessment_stat = 1
        elif self.variant == "default":
            if self.ball.block: self.assessment_stat += 1
        elif self.variant == "negative_rand_row":
            if self.reward > 0: self.assessment_stat += 1
        elif self.variant == "center_large":
            if self.ball.paddle: self.assessment_stat -= 1
        elif self.variant == "breakout_priority_large":
            if self.ball.paddle: self.assessment_stat -= 1
        elif self.variant == "harden_single": 
            if self.ball.paddle: self.assessment_stat -= 1
        elif self.variant == "single_block": 
            if self.ball.paddle: self.assessment_stat -= 1
        elif self.variant == "proximity":
            if self.ball.block:
                # the assessment stat for proximity is (number of hits, distance of hits), when combined this gives average distance of hit
                if type(self.assessment_stat) == tuple: self.assessment_stat = (self.assessment_stat[0] + 1, self.assessment_stat[1] + np.linalg.norm(self.sampler.param[:2] - self.ball.block_id.getMidpoint(), ord=1))
                print("hit at l1", np.linalg.norm(self.sampler.param[:2] - self.ball.block_id.getMidpoint(), ord=1), self.ball.block_id.getMidpoint(), self.sampler.param[:2])
            if self.done:
                if type(self.assessment_stat) == tuple:
                    self.assessment_stat = self.assessment_stat[1] / self.assessment_stat[0]

    def ball_reset(self):
        self.ball.pos = [41, np.random.randint(20, 52)]
        # self.ball.pos = [np.random.randint(38, 45), np.random.randint(14, 70)]
        self.ball.vel = np.array([np.random.randint(1,2), np.random.choice([-1,1])])

    def assign_attribute(self, nmode, block, atrv):
        if nmode == "side":
            if block.pos[1] < 42:
                block.attribute = atrv
        elif nmode == "top":
            if block.pos[0] < 22 + self.block_height * self.num_rows / 2:
                block.attribute = atrv
        elif nmode == "edge":
            if block.pos[1] < 28 or block.pos[1] > 56:
                block.attribute = atrv
        elif nmode == "center":
            if 28 < block.pos[1] < 56:
                block.attribute = atrv
        elif nmode == "checker":
            block.attribute = -1 + (int(block.name[5:]) % 2) * 2

    def get_nmode_atrv(self):
        atrv = -1
        nmode = self.negative_mode
        if self.negative_mode[:4] == "zero":
            atrv = 0
            nmode = self.negative_mode[4:]
        if self.negative_mode[:4] == "hard":
            atrv = -1
            nmode = self.negative_mode[4:]
        return atrv, nmode

    def assign_attributes(self):
        '''
        assigns negative/hard blocks
        '''
        atrv, nmode = self.get_nmode_atrv()
        if nmode == "scatter":
            newblocks = list()
            pos = list(range(len(self.blocks)))
            self.target = np.random.choice(pos, size=1, replace=False)[0]
            pos.pop(self.target)
            self.choices = np.random.choice(pos, size=10, replace=False) # Hardcoded at the moment
            for choice in self.choices:
                self.blocks[choice].attribute = atrv
                newblocks.append(self.blocks[choice])
            self.blocks[self.target].attribute = 1
            newblocks.append(self.blocks[self.target])
            for i, block in enumerate(newblocks):
                block.name = "Block" + str(i)
            self.blocks = newblocks
        elif nmode == "rand":
            self.choices = np.random.choice(list(range(len(self.blocks))), size=len(self.blocks) // 2, replace=False)
            for choice in self.choices:
                self.blocks[choice].attribute = atrv
        else:
            for block in self.blocks:
                self.assign_attribute(nmode, block, atrv)

    def reset_target_mode(self):
        '''
        handles block resetting for single block as a target domains
        '''
        if self.bounce_cost != 0: # target mode with small blocks
            pos_block = Block(np.array([int(17 + np.random.rand() * 20),int(15 + np.random.rand() * 51)]), 1, -1, (0,0), size = 2)
            self.block_width = pos_block.width
            self.block_height = pos_block.height
            self.blocks = [pos_block]
            if len(self.negative_mode) > 0:
                while True:
                    neg_pos = np.array([int(17 + np.random.rand() * 20),int(15 + np.random.rand() * 51)])
                    if abs(neg_pos[0] - pos_block.pos[0]) > 4 or abs(neg_pos[1] - pos_block.pos[1]) > 6:
                        break
                neg_block = Block(neg_pos, -1, -1, (0,0), size = 2)
                self.blocks.append(neg_block)
        else:
            self.blocks = [Block(np.array([17,15 + np.random.randint(4) * 15]), 1, -1, (0,0), size = 6)]
            self.block_height = self.blocks[0].height
            self.block_width = self.blocks[0].width
            self.num_rows = 1

    def reset_default(self): 
        '''
        handles block resetting for multiple blocks as targets domains
        '''
        blockheight = min(self.max_block_height, 10 // self.num_rows)
        blockwidth = 60 // self.num_columns
        for i in range(self.num_rows):
            block2D_row = list()
            for j in range(self.num_columns):
                block = Block(np.array([22 + i * blockheight,12 + j * blockwidth]), 1, i * self.num_columns + j, (i,j), width=blockwidth, height=blockheight)
                self.blocks.append(block)
                # self.blocks.append(Block(np.array([32 + i * 2,12 + j * 3]), 1, i * 20 + j))
                block2D_row.append(block)
        if self.random_exist > 0:
            choices = np.random.choice(list(range(len(self.blocks))), replace=False, size=self.random_exist)
            new_blocks = list()
            for i, choice in enumerate(choices):
                newb = self.blocks[choice]
                newb.name = "Block" + str(i)
                new_blocks.append(newb)
            self.blocks = new_blocks

    def reset(self):
        vel= np.array([np.random.randint(1,2), np.random.choice([-1,1])])
        self.ball = Ball(np.array([np.random.randint(38, 45), np.random.randint(14, 70)]), 1, vel, top_reset=self.target_mode and self.bounce_cost == 0, hard_mode=self.hard_mode)
        self.ball.reset_pos()
        self.paddle = Paddle(np.array([71, 84//2]), 1, np.zeros((2,), dtype = np.int64))
        self.actions = Action(np.zeros((2,), dtype = np.int64), 0)
        self.animate = [self.paddle, self.ball]

        # assign blocks
        self.blocks = []
        if self.target_mode:
            self.reset_target_mode()
        else:
            self.reset_default()
        self.assign_attributes()

        # assign object names, used for traces, TODO: Reward and Done not handled
        self.object_name_dict = {**{"Action": self.actions, "Paddle": self.paddle, "Ball": self.ball}, **{"Block" + str(i): self.blocks[i] for i in range(len(self.blocks))}}
        # assign walls
        self.walls = []
        self.walls.append(Wall(np.array([4, 4]), 1, "Top"))
        self.walls.append(Wall(np.array([80, 4]), 1, "Bottom"))
        self.walls.append(Wall(np.array([0, 8]), 1, "LeftSide"))
        self.walls.append(Wall(np.array([0, 72]), 1, "RightSide"))
        self.objects = [self.actions, self.paddle, self.ball] + self.blocks + self.walls

        # zero out relevant values
        self.assessment_stat = 0 if self.variant != "proximity" else (0,0)
        self.needs_ball_reset = False
        self.hit_counter = 0
        self.bounce_counter = 0
        self.since_last_bounce = 0
        self.total_score = 0
        self.render()
        if self.variant == "proximity" and hasattr(self, "sampler"): self.sampler.sample(self.get_state())
        return self.get_state()

    def render(self):
        self.frame = np.zeros((84,84), dtype = 'uint8')
        for block in self.blocks:
            if block.attribute != 0:
                self.frame[block.pos[0]:block.pos[0]+block.height, block.pos[1]:block.pos[1]+block.width] = .5 * 255
            if block.attribute == -1:
                self.frame[block.pos[0]:block.pos[0]+block.height, block.pos[1]:block.pos[1]+block.width] = .2 * 255
            if block.attribute == 2:
                self.frame[block.pos[0]:block.pos[0]+block.height, block.pos[1]:block.pos[1]+block.width] = .8 * 255
        for wall in self.walls:
            self.frame[wall.pos[0]:wall.pos[0]+wall.height, wall.pos[1]:wall.pos[1]+wall.width] = .3 * 255
        ball, paddle = self.ball, self.paddle
        self.frame[ball.pos[0]:ball.pos[0]+ball.height, ball.pos[1]:ball.pos[1]+ball.width] = 1 * 255
        self.frame[paddle.pos[0]:paddle.pos[0]+paddle.height, paddle.pos[1]:paddle.pos[1]+paddle.width] = .75 * 255
        return self.frame

    def get_num(self, live=1):
        total = 0
        for block in self.blocks:
            if block.attribute == live:
                total += 1
        return total

    def get_state(self, render=False):
        if render: self.render()
        state =  {"raw_state": self.frame, "factored_state": numpy_factored({**{obj.name: obj.getMidpoint() + obj.vel.tolist() + [obj.getAttribute()] for obj in self.objects}, **{'Done': [self.done], 'Reward': [self.reward]}})}
        return state

    def get_info(self):
        return {"lives": 5-self.ball.losses, "TimeLimit.truncated": False, "assessment": self.assessment_stat, "total_score": self.total_score}

    def clear_interactions(self):
        self.ball.clear_hits()
        for o in self.objects:
            o.interaction_trace = list()

    def interaction_effects(self, obj1, obj2):
        '''
        if obj2 is a block and obj1 is a ball and the ball hit the block, then the block attribute should change
        and the reward will be computed accordingly
        other interaction effects include the action moving the paddle, or the ball bouncing off of the paddle
        or the walls affecting the ball
        '''
        preattr = obj2.attribute
        obj1.interact(obj2)
        hit = False
        if preattr != obj2.attribute:
            if self.variant == "proximity":
                rew = self.compute_proximity_reward(self.sampler.param, np.concatenate([np.array(obj2.getMidpoint()), np.array([0,0,0])]))
                self.reward += rew
            else:
                self.reward += preattr * self.default_reward
                self.total_score += preattr * self.default_reward
            hit = True
        return hit

    def step(self, action, render=True, angle=-1): # TODO: remove render as an input variable
        self.done = False
        self.reward = 0

        hit = False
        needs_reset = False
        needs_ball_reset = False
        self.clear_interactions()
        for i in range(self.frameskip):
            if self.no_breakout: # fixes the blocks that got hit so breakouts do not occur
                atrv, nmode = self.get_nmode_atrv() 
                for choice in self.choices:
                    self.blocks[choice].attribute = atrv
                for block in self.blocks:
                    if block.attribute == 0:
                        block.attribute = 1
            # perform the object dynamics, updating block reward accordingly
            self.actions.take_action(action)
            self.paddle.interact(self.actions)
            for wall in self.walls:
                self.ball.interact(wall)
            # print("ball wall", self.ball.pos, self.paddle.pos, self.ball.vel, self.paddle.vel)
            self.ball.interact(self.paddle)
            # print("ball", self.ball.pos, self.paddle.pos, self.ball.vel, self.paddle.vel)
            if self.ball.pos[0] < self.low_block:
                for block in self.blocks:
                    hit += self.interaction_effects(self.ball, block)
                    # print(block.pos)
            # # old logic
            # for obj1 in self.animate:
            #     for obj2 in self.objects:
            #         if obj1.name == obj2.name:
            #             continue
            #         else:
            #             hit += self.interaction_effects(obj1, obj2)
            for ani_obj in self.animate:
                ani_obj.move()

            # render prior to resets
            if render: self.render()

            # penalize bouncing the ball off the paddle, if necessary
            if self.ball.paddle and self.bounce_cost < 0: self.reward += self.bounce_cost # the bounce count is a penalty for paddle bounces only if negative
            self.dropped = self.ball.bottom_wall or self.top_dropping and self.ball.top_wall
            if self.dropped:
                self.reward += self.timeout_penalty # negative reward for dropping the ball since done is not triggered
                self.total_score += self.timeout_penalty
                needs_ball_reset = True
                self.dropped = True
                self.since_last_bounce = 0

            # end of episode by dropping
            if self.ball.losses == 5 or (self.dropped and self.drop_stopping):
                self.done = True
                needs_reset = self.ball.losses == 5
                print("eoe", self.total_score)
                break

            # record hit information (end of episode)
            if hit:
                self.hit_counter += 1
                # end of episode by hitting
                if ((self.get_num(True) == 0 and self.hit_reset <= 0) # removed as many blocks as necessary (all the positive blocks in negative/hard block domains, all the blocks in other domains)
                    or (self.no_breakout and self.hit_reset > 0 and self.hit_counter == self.hit_reset) # reset after a fixed number of hits
                    or self.target_mode): # only one block to hit
                    print(self.hit_counter, self.hit_reset, self.target_mode)
                    needs_reset = True
                    self.reward += self.completion_reward * self.default_reward
                    self.done = True
                    print("eoe", self.total_score)
                    break

            # reset because the ball is stuck
            self.since_last_bounce += 1
            if self.since_last_bounce > 1000:
                needs_reset = True
                break

            # record paddle bounce information (end of episode)
            if self.ball.paddle:
                self.since_last_bounce = 0
                if self.bounce_reset > 0:
                    self.bounce_counter += 1
                    if self.bounce_counter == self.bounce_reset:
                        needs_reset = True
                        self.done = True
                        break

        # record state information before any resets
        self.itr += 1
        full_state = self.get_state(render)
        frame, extracted_state = full_state['raw_state'], full_state['factored_state']
        lives = 5-self.ball.losses

        # get assessment values
        self.assign_assessment_stat() # TODO: bugs may occur if using frame skipping
        assessment_stat = self.assessment_stat
        info = {"lives": lives, "TimeLimit.truncated": False, "assessment": self.assessment_stat, "total_score": self.total_score} # treats drops as truncations
        
        # perform resets
        if needs_ball_reset: self.ball.reset_pos()
        if needs_reset: self.reset()
        if hit and self.variant == "proximity": self.sampler.sample(full_state)
        return full_state, self.reward, self.done, info

    def compute_proximity_reward(self, target_block, block):
        dist = np.linalg.norm(target_block[:2] - block[:2], ord=1)
        return (np.exp(-dist/10) - .2) * 2

    def set_from_factored_state(self, factored_state, seed_counter=-1, render=False):
        '''
        TODO: only sets the active elements, and not the score, reward and other features. This could be an issue in the future.
        '''
        if seed_counter > 0:
            self.seed_counter = seed_counter
            self.ball.reset_seed = seed_counter
        if "Ball" in factored_state:
            self.ball.pos = self.ball.getPos(np.array(factored_state["Ball"]).squeeze()[:2])
            self.ball.vel = np.array(factored_state["Ball"]).squeeze()[2:4].astype(int)
            self.ball.losses = 0 # ensures that no weirdness happens since ball losses are not stored, though that might be something to keep in attribute...
        if "Paddle" in factored_state:
            self.paddle.pos = self.paddle.getPos(np.array(factored_state["Paddle"]).squeeze()[:2])
            self.paddle.vel = np.array(factored_state["Paddle"]).squeeze()[2:4].astype(int)
        if "Action" in factored_state:
            self.actions.attribute = factored_state["Action"][-1]
        if "Block" in factored_state:
            for i in range(self.num_blocks):
                self.blocks[i].attribute = float(np.array(factored_state["Block"]).squeeze()[i*5+4])
        if "Block0" in factored_state:
            i=0
            while "Block" + str(i) in factored_state:
                self.blocks[i].attribute = float(np.array(factored_state["Block" + str(i)]).squeeze()[-1])
                i += 1
        if render: self.render_frame()
        # TODO: set the info from the factored state as well

    def demonstrate(self):
        action = 0
        frame = self.render()
        frame = cv2.resize(frame, (frame.shape[0] * 5, frame.shape[1] * 5), interpolation = cv2.INTER_NEAREST)
        cv2.imshow('frame',frame)
        key = cv2.waitKey(100)
        if key == ord('q'):
            action = -1
        elif key == ord('a'):
            action = 2
        elif key == ord('w'):
            action = 1
        elif key == ord('s'):
            action = 0
        elif key == ord('d'):
            action = 3
        return action

if __name__ == '__main__':
    screen = Screen()
    # policy = RandomPolicy(4)
    # policy = RotatePolicy(4, 9)
    # policy = BouncePolicy(4)
    screen.run(policy, render=True, iterations = 1000, duplicate_actions=1, save_path=sys.argv[1])
    # demonstrate(sys.argv[1], 1000)
