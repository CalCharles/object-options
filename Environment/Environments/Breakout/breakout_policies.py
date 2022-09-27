import numpy as np
import copy
from Environment.Environments.Breakout.breakout_screen import Breakout

class Policy():
    def act(self, screen):
        print ("not implemented")

    def get_angle(self, screen):
        return 0

class RandomPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, screen, angle=0):
        return np.random.randint(self.action_space)

class RandomConsistentPolicy(Policy):
    def __init__(self, action_space, change_prob):
        self.action_space = action_space
        self.change_prob = change_prob
        self.current_action = np.random.randint(self.action_space)

    def act(self, screen, angle=0):
        if np.random.rand() < self.change_prob:
            self.current_action = np.random.randint(self.action_space)
        return self.current_action

class RotatePolicy(Policy):
    def __init__(self, action_space, hold_count):
        self.action_space = action_space
        self.hold_count = hold_count
        self.current_action = 0
        self.current_count = 0

    def act(self, screen, angle=0):
        self.current_count += 1
        if self.current_count >= self.hold_count:
            self.current_action = np.random.randint(self.action_space)
            # self.current_action = (self.current_action+1) % self.action_space
            self.current_count = 0
        return self.current_action

class BouncePolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space
        self.internal_screen = Screen(angle_mode = False)
        self.objective_location = 84//2
        self.last_paddlehits = -1

    def act(self, screen, angle=0):
        # print(screen.ball.paddlehits, screen.ball.losses, self.last_paddlehits)
        if screen.ball.paddlehits + screen.ball.losses > self.last_paddlehits or (screen.ball.paddlehits + screen.ball.losses == 0 and self.last_paddlehits != 0):
            if (screen.ball.paddlehits + screen.ball.losses == 0 and self.last_paddlehits != 0):
                self.last_paddlehits = 0
            self.internal_screen = copy.deepcopy(screen)
            self.internal_screen.angle_mode = False
            self.internal_screen.save_path = ""
            # print(self.internal_screen.ball.pos, screen.ball.pos, self.last_paddlehits)

            while self.internal_screen.ball.pos[0] < 71 and not self.internal_screen.done:
                # print("running internal")
                self.internal_screen.step(0)
            # print("completed")
            # print(self.internal_screen.ball.pos, screen.ball.pos, self.last_paddlehits)
            self.objective_location = self.internal_screen.ball.pos[1] + np.random.choice([-1, 0, 1])
            self.last_paddlehits += 1
        if self.objective_location > screen.paddle.getMidpoint()[1]:
            return 3
        elif self.objective_location < screen.paddle.getMidpoint()[1]:
            return 2
        else:
            return 0

class AnglePolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space
        self.internal_screen = Breakout()
        self.objective_location = 84//2
        self.last_paddlehits = -1
        self.counter = 0

    def reset_screen(self, screen):
        self.internal_screen = copy.deepcopy(screen)
        self.internal_screen.angle_mode = False
        self.internal_screen.save_path = ""

    def pick_action(self, objective_location, xpoint):
        if objective_location > xpoint:
            return 3
        elif objective_location < xpoint:
            return 2
        else:
            return 0

    def act(self, screen, angle=0, force=False):
        if screen.ball.vel[0] > 0 and 46 <= screen.ball.pos[0] <= 47 or screen.ball.vel[0] < 0 and 67 <= screen.ball.pos[0] <= 68 or force:
            if (screen.ball.paddlehits + screen.ball.losses == 0 and self.last_paddlehits != 0):
                self.last_paddlehits = 0
            self.reset_screen(screen)
            # print(self.internal_screen.ball.pos, screen.ball.pos, self.last_paddlehits)

            while self.internal_screen.ball.pos[0] < 69 and not self.internal_screen.done:
                # print("running internal")
                self.internal_screen.step(0)
            # print("completed")
            # print(self.internal_screen.ball.pos, screen.ball.pos, self.last_paddlehits)
            base_location = self.internal_screen.ball.pos[1]
            sv = self.internal_screen.ball.vel[1]
            if angle == 0:
                self.objective_location = base_location + sv * 1
            elif angle == 1:
                self.objective_location = base_location - 2 + sv * 1
            elif angle == 2:
                self.objective_location = base_location - 4 + sv * 1
            elif angle == 3:
                self.objective_location = base_location - 6 + sv * 1
            self.objective_location += self.objective_location % 2
        return self.pick_action(self.objective_location, screen.paddle.pos[1])

def DemonstratorPolicy(Policy):
    def act(self, screen, angle=0):
        action = 0
        frame = screen.render_frame()
        cv2.imshow('frame',frame)
        key = cv2.waitKey(500)
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


def demonstrate(save_dir, num_frames):
    domain = Screen()
    quit = False
    domain.set_save(0, save_dir, 0, True)
    for i in range(num_frames):
        frame = domain.render_frame()
        cv2.imshow('frame',frame)
        action = 0
        key = cv2.waitKey(500)
        if key == ord('q'):
            quit = True
        elif key == ord('a'):
            action = 2
        elif key == ord('w'):
            action = 1
        elif key == ord('s'):
            action = 0
        elif key == ord('d'):
            action = 3
        domain.step(action)
        if quit:
            break
