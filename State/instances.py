    def split_instances(self, delta_state, obj_dim=-1):
        # split up a state or batch of states into instances
        if obj_dim < 0:
            obj_dim = self.object_dim
        nobj = delta_state.shape[-1] // obj_dim
        if len(delta_state.shape) == 1:
            delta_state = delta_state.reshape(nobj, obj_dim)
        elif len(delta_state.shape) == 2:
            delta_state = delta_state.reshape(-1, nobj, obj_dim)
        return delta_state

    def flat_instances(self, delta_state):
        # change an instanced state into a flat state
        if len(delta_state.shape) == 2:
            delta_state = delta_state.flatten()
        elif len(delta_state.shape) == 3:
            batch_size = delta_state.shape[0]
            delta_state = delta_state.reshape(batch_size, delta_state.shape[1] * delta_state.shape[2])
        return delta_state
