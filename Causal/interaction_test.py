    
class InteractionTesting:
    def __init__(self, interaction_params)
        self.interaction_prediction, self.forward_threshold, self.passive_threshold, self.difference_threshold = interaction_params

    def compute_binary(self, forward, passive):
        '''computes an interaction binary, which defines if the active prediction is high likelihood enough
        the passive is low likelihood enough, and the difference is sufficiently large
        TODO: there should probably be a mechanism that incorporates variance explicitly
        TODO: there should be a way of discounting components of state that are ALWAYS predicted with high probability
        '''
        
        # values based on log probability
        active_prediction = forward < self.forward_threshold # the prediction must be good enough (negative log likelihood)
        not_passive = passive > self.passive_threshold # the passive prediction must be bad enough
        difference = forward - passive < self.difference_threshold # the difference between the two must be large enough
        return ((not_passive) * (active_prediction) * (difference)).float() #(active_prediction+not_passive > 1).float()

    def __call__(self, interactions):
        return interactions > self.interaction_prediction