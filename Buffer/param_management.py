# param management

    def add_param(self, batch, indices = None):
        orig_obs, orig_next = None, None
        if self.parameterized:
            orig_obs, orig_next = batch.obs, batch.obs_next
            if self.param_process is None:
                param_process = lambda x,y: np.concatenate((x,y), axis=1) # default to concatenate
            else:
                param_process = self.param_process
            if indices is None:
                batch['obs'] = param_process(batch['obs'], batch['param'])
                if type(batch['obs_next']) == np.ndarray: batch['obs_next'] = param_process(batch['obs_next'], batch['param']) # relies on batch defaulting to Batch, and np.ndarray for all other state representations
            else: # indices indicates that it is handling a buffer
                batch.obs[indices] = param_process(batch.obs[indices], batch.param[indices])
                if type(batch.obs_next[indices]) == np.ndarray: batch.obs_next[indices] = param_process(batch.obs_next[indices], batch.param[indices])                
                # print(batch.obs[indices].shape, batch.obs_next.shape)
        return orig_obs, orig_next


    def restore_obs(self, batch, orig_obs, orig_next):
        if self.parameterized:
            batch['obs'], batch['obs_next'] = orig_obs, orig_next

    def restore_buffer(self, buffer, orig_obs, orig_next, rew, done, idices):
        if self.parameterized:
            buffer.obs[idices], buffer.obs_next[idices], buffer.rew[idices], buffer.done[idices] = orig_obs, orig_next, rew, done
