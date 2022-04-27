    def terminate_reward_chain(self, full_state, next_full_state, param, chain, mask, mask_chain, environment_model=None):
        # recursively get all of the dones and rewards
        true_inter = 0 # true_inter is the actual interaction, used as a base case, 0 if not used
        if self.next_option is not None: # lower levels should have masks the same as the active mask( fully trained)
            if type(self.next_option) != PrimitiveOption:
                next_param = self.next_option.sampler.convert_param(chain[-1]) # mapped actions need to be expanded to fit param dimensions
                next_mask = mask_chain[-2]
            else:
                next_param = chain[-1]
                next_mask = mask_chain[-1]
            last_done, last_rewards, last_termination, last_ext_term, _, _ = self.next_option.terminate_reward_chain(full_state, next_full_state, next_param, chain[:len(chain)-1], next_mask, mask_chain[:len(mask_chain)-1])
            if self.terminate_reward.true_interaction: true_inter = self.dataset_model.check_current_trace(self.next_option.name, self.name)
        termination, reward, inter, time_cutoff = self.terminate_reward.check(full_state, next_full_state, param, mask, true_inter=true_inter)
        ext_term = self.temporal_extension_manager.get_extension(termination, last_termination[-1])
        done = self.done_model.check(termination, self.state_extractor.get_true_done(next_full_state))
        rewards, terminations, ext_term = last_rewards + [reward], last_termination + [termination], last_ext_term + [ext_term]
        mid_term = False
        for i in range(1, len(ext_term) + 1):
            mid_term = ext_term[len(ext_term) - i] or mid_term
            ext_term[len(ext_term) - i] = mid_term
        return done, rewards, terminations, ext_term, inter, time_cutoff

class Reward():
    def __init__(self, **kwargs):
        pass
        
    def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
        return 1

class BinaryParameterizedReward(Reward):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.use_diff = kwargs['use_diff'] # compare the parameter with the diff, or with the outcome
        # self.use_both = kwargs['use_both'] # supercedes use_diff
        self.epsilon_close = kwargs['epsilon_close']
        self.norm_p = kwargs['param_norm']

    def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
        # NOTE: input state is from the current state, state, param are from the next state
        # if self.use_both:
        #   if len(diff.shape) == 1 and len(param.shape) == 1:
        #       s = torch.cat((state, diff), dim=0)
        #       return ((s - param).norm(p=1) <= self.epsilon).float()
        #   else:
        #       s = torch.cat((state, diff), dim=1)
        #       return ((s - param).norm(p=1, dim=1) <= self.epsilon).float()
        # elif self.use_diff:
        #   if len(diff.shape) == 1 and len(param.shape) == 1:
        #       return ((diff - param).norm(p=1) <= self.epsilon).float()
        #   else:
        #       return ((diff - param).norm(p=1, dim=1) <= self.epsilon).float()
        # else:
        if len(state.shape) == 1:
            return (np.linalg.norm((state - param) * mask, ord = 1) <= self.epsilon_close).astype(float)
        else:
            return (np.linalg.norm((state - param) * mask, ord = 1, axis=1) <= self.epsilon_close).astype(float)

class Termination():
    def __init__(self, **kwargs):
        pass

    def check_interaction(self, inter):
        return True

    def check(self, input_state, state, param, mask, true_done=0):
        return True

class ParameterizedStateTermination(Termination):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        dataset_model = kwargs["dataset_model"]
        self.epsilon_close = kwargs['epsilon_close']
        self.name = kwargs['name']
        self.inter = 1
        self.inter_pred = dataset_model.interaction_prediction

    def check_interaction(self, inter):
        return inter > (1-self.inter_pred)


    def check(self, input_state, state, param, mask, true_done=0): # handling diff/both outside
        # NOTE: input state is from the current state, state, param are from the next state
        # param = self.convert_param(param)
        # if self.use_both:
            # if len(diff.shape) == 1:
            #   s = torch.cat((state, diff), dim=0)
            #   return (s - param).norm(p=1) <= self.epsilon
            # else:
                # s = torch.cat((state, diff), dim=1)
                # return (s - param).norm(p=1, dim=1) <= self.epsilon
        # elif self.use_diff:
        #   if len(diff.shape) == 1:
        #       return (diff - param).norm(p=1) <= self.epsilon
        #   else:
        #       return (diff - param).norm(p=1, dim=1) <= self.epsilon
        # else:
        if len(state.shape) == 1:
            # print("dist", np.linalg.norm((state - param) * mask, ord  = 1), self.epsilon_close, state, param)
            return np.linalg.norm((state - param) * mask, ord  = 1) <= self.epsilon_close
        else:
            return np.linalg.norm((state - param) * mask, ord =1, axis=1 ) <= self.epsilon_close

class DoneModel():
    def __init__(self, **kwargs):
        self.use_termination = kwargs["use_termination"]
        self.use_timer = kwargs["time_cutoff"]
        self.use_true_done = kwargs["true_done_stopping"]
        self.timer= 0

    def update(self, done):
        self.timer += 1
        if done:
            self.timer = 0


    def check(self, termination, true_done):
        term, tim, tru = self.done_check(termination, true_done)
        # print(term, tim, tru)
        if term:
            # print("term done")
            return term
        elif tim:
            # print("timer done")
            return tim
        elif tru:
            # print("true done")
            return tru
        return term or tim or tru

    def done_check(self, termination, true_done):
        if type(termination) == np.ndarray: termination = termination.squeeze() # troublesome line
        term = (termination * self.use_termination)
        tim = (self.timer == self.use_timer)
        if type(true_done) == np.ndarray: true_done = true_done.squeeze() # troublesome line
        tru = (self.use_true_done * true_done)
        return term, tim, tru