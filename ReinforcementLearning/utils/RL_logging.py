from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from Network.network_utils import pytorch_model

def collect_test_trials(logger, option, test_collector, term_step, i, trials, random):
    '''
    collect trials with the test collector
    the environment is reset before starting these trials
    most of the inputs are used for printing out the results of these trials 
    term_step is a tuple of the max number of episodes, and max number of steps for a testing episode
    '''
    option.toggle_test(True) # switches option to testing mode
    test_collector.reset()
    print("starting trials")
    results = list()
    for j in range(trials):
        result = test_collector.collect(n_episode=1, n_term=None if term_step[0] <=0 else term_step[0], n_step=term_step[1], random=random, new_param=True)
        result['n/tr'] = max(1, result['n/tr']) # at least one (virtual) epsiode occurs before the end, for testing purposes
        logger.log_results(result)
        results.append(result)
    option.toggle_test(False) # switched option back to training mode
    return results
    
def buffer_printouts(args, train_collector, option):
    # Buffer debugging printouts
    print("itr, idx, done, term, trunc, reward, inter")
    print("acts, action, mapped, q")
    print("tar, time, param, target, next target, inter state")
    print("obs, obs, rvobs, next obs, rvnext obs")
    rv = option.state_extractor.reverse_obs_norm
    buf = train_collector.buffer
    for j in range(100):
        idx = (train_collector.at + (j - 100)) % args.collect.buffer_len
        d, info, tm, r, bi, a, ma, ti, p, t, nt, itr, obs, obs_n, mask = buf[idx].done, buf[idx].info, buf[idx].terminate, buf[idx].rew, buf[idx].inter, buf[idx].act, buf[idx].mapped_act, buf[idx].time, buf[idx].param, buf[idx].target, buf[idx].next_target, buf[idx].inter_state, buf[idx].obs, buf[idx].obs_next, buf[idx].mask
        # print(obs.shape, rv_variance.shape, rv_mean.shape)
        
        print(j, idx, d, tm, info["TimeLimit.truncated"], r, bi, 
            "\nacts", a, ma, pytorch_model.unwrap(option.policy.compute_Q(Batch(obs=obs, obs_next = obs_n,info=info, act=a), False)), 
            "\ntar", ti, p, t, nt, itr, 
            "\nobs", obs, rv(obs, mask.squeeze()), obs_n, rv(obs_n, mask.squeeze()))

    if args.hindsight.use_her:
        print("itr, idx, done, term, trunc, reward, inter")
        print("acts, action, mapped, q")
        print("tar, time, param, target, next target, inter state")
        print("obs, obs, rvobs, next obs, rvnext obs")
        hrb = train_collector.her_buffer
        if len(hrb) > 10:
            print("hindsight buffer", len(hrb))
            for j in range(100):
                idx = (train_collector.her_at + (j - 100)) % args.collect.buffer_len
                dh, infoh, tmh, rh, ih, ah, mah, tih, ph, th, nth, itrh, obsh, obs_nh, mask_h = hrb[idx].done, hrb[idx].info, hrb[idx].terminate, hrb[idx].rew, hrb[idx].inter, hrb[idx].act, hrb[idx].mapped_act, hrb[idx].time, hrb[idx].param, hrb[idx].target, hrb[idx].next_target, hrb[idx].inter_state, hrb[idx].obs, hrb[idx].obs_next, hrb[idx].mask
                print(j, idx, dh, tmh, infoh["TimeLimit.truncated"], rh, ih, 
                    "\nacts", ah, mah, pytorch_model.unwrap(option.policy.compute_Q(Batch(obs=obsh, obs_next = obs_nh,info=infoh, act=ah), False)),
                    "\ntar",tih,  ph,  th, nth, itrh, 
                    "\nobs", obsh, rv(obsh, mask_h.squeeze()), obs_nh, rv(obs_nh, mask_h.squeeze()))