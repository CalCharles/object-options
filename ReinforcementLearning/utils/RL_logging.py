
def add_assessment(env, result, assessment, drops, timeout):
    if timeout:
        assessment.append(env.timeout_penalty)
    for assesses in result["assessment"]:
        if assesses <= -1000:
            while assesses <= -1000:
                drops.append(1)
                assesses = assesses + 1000
            assessment.append(assesses)
        elif assesses > -900:
            assessment.append(assesses)
            drops.append(0)
        else:
            drops.append(1)
    # print("adding assessment", assessment, result["assessment"])

def _collect_test_trials(args, test_collector, i, total_steps, total_epsiodes, test_perf, suc, hit_miss, hit_miss_train, assessment_test, assessment_train, drops, train_drops, random=False, option=None, tensorboard_logger=None):
    '''
    collect trials with the test collector
    the environment is reset before starting these trials
    most of the inputs are used for printing out the results of these trials 
    '''
    test_collector.reset()
    trials = args.test_trials
    # print(test_collector.data)
    if random:
        trials = args.pretest_trials
    #     trials = args.test_trials * 10
    eps = option.policy.epsilon
    option.set_epsilon(.02) # small epsilon
    if args.object == "Block" and args.env == "SelfBreakout" and not args.target_mode:
        orig_env_model = test_collector.option.sampler.current_environment_model
        test_collector.option.sampler.current_environment_model = test_collector.environment_model
    for j in range(trials):
        result = test_collector.collect(n_episode=1, n_term=None if args.test_episode else 1, n_step=args.max_steps, random=random, visualize_param=args.visualize_param)
        test_perf.append(result["rews"].mean())
        suc.append(float(result["terminate"] and args.max_steps != result['n/st']))
        hit_miss.append(result['n/h'])
        add_assessment(args.environment, result, assessment_test, drops, args.max_steps == result['n/st'])
    option.set_epsilon(eps)
    if random:
        print("Initial trials: ", trials)
    else:
        print("Iters: ", i, "Steps: ", total_steps, "Episodes: ", total_epsiodes)
    mean_perf, mean_suc, mean_hit, mean_assessment = np.array(test_perf).mean(), np.array(suc).mean(), sum(hit_miss)/ max(1, len(hit_miss)), np.array(assessment_test).mean()
    total_drops = np.sum(np.array(drops))
    hmt = 0.0
    if len(list(hit_miss_train)) > 0:
        hmt = np.sum(np.array(list(hit_miss_train)), axis=0)
        hmt = hmt[0] / (hmt[0] + hmt[1])
    mean_train_assess = 0
    if len(list(assessment_train)) > 0:
        mean_train_assess = np.array(assessment_train)
        mean_train_assess = mean_train_assess.mean()
    train_drops_num = 0
    if len(list(train_drops)) > 0:
        train_drops = np.array(train_drops)
        train_drops_num = np.sum(train_drops)
    print(f'Test mean returns: {mean_perf}', f"Success: {mean_suc}", 
        f"Hit Miss: {mean_hit}", f"Hit Miss train: {hmt}", 
        f"Assess: {mean_assessment}", f"Assess Train: {mean_train_assess}", 
        f"Drops: {total_drops}", f"Train drops: {train_drops_num}")
    if tensorboard_logger is not None:
        tensorboard_logger.add_scalar("Return", mean_perf, i)
        tensorboard_logger.add_scalar("Hit Miss/Test", mean_hit, i)
        tensorboard_logger.add_scalar("Hit Miss/Train", hmt, i)
        tensorboard_logger.add_scalar("Assessment/Test", mean_assessment, i)
        tensorboard_logger.add_scalar("Assessment/Train", mean_train_assess, i)
        tensorboard_logger.flush()
    if args.object == "Block" and args.env == "SelfBreakout"  and not args.target_mode:
        test_collector.option.sampler.current_environment_model = orig_env_model
    return mean_perf, mean_suc, mean_hit 

def full_save(args, option, graph):
    option.save(args.save_dir)
    graph.save_graph(args.save_graph, [args.object], args.environment_model, cuda=args.cuda)


    test_perf, suc, assessment_test, assessment_train, drops = deque(maxlen=2000), deque(maxlen=2000), deque(maxlen=100), deque(maxlen=100), deque(maxlen=200)



def save_pretrain()
    if len(args.save_pretrain) > 0:
        her_at = option.policy.learning_algorithm.at if option.policy.learning_algorithm is not None else 0
        her_buffer = option.policy.learning_algorithm.replay_buffer if option.policy.learning_algorithm is not None else None
        buffer_wrapper = BufferWrapper(train_collector.at, train_collector.buffer, 
                                        train_collector.full_at, train_collector.full_buffer,
                                        her_at, her_buffer)
        save_to_pickle(os.path.join(args.save_pretrain,"pretrain_collector.pkl"), buffer_wrapper)


def record_stats():
        total_steps, total_episodes = collect_result['n/st'] + total_steps, collect_result['n/tep'] + total_episodes
        # once if the collected episodes' mean returns reach the threshold,
        # or every 1000 steps, we test it on test_collector
        add_assessment(environment, collect_result, assessment_train, train_drops, False)
        # print("episodes", collect_result['n/ep'], collect_result['assessment'], np.mean(assessment_train))
        hit_miss_queue_train.append([collect_result['n/h'], collect_result['n/m']])
        cumul_losses.append(losses)
        for lk in losses.keys():
            tensorboard_logger.add_scalar(lk, losses[lk], i)
        tensorboard_logger.flush()

def log_stats():
            total_losses = copy.deepcopy(cumul_losses[0])
            for j in range(len(cumul_losses) - 1):
                l = cumul_losses[j]
                for k in l.keys():
                    if k not in total_losses:
                        total_losses[k] = l[k]
                    total_losses[k] += l[k]
            for k in total_losses.keys():
                total_losses[k] = total_losses[k] / len(cumul_losses)
            print("losses", total_losses)
            option.print_epsilons()
            print("FPS: ", ((i+1) *args.num_steps + args.pretrain_iters) /(time.time() - start))
            # print("epsilons", epsilon, interaction, epsilon_close)



def buffer_printouts():
        # Buffer debugging printouts
        if i % args.log_interval == 0 and args.print_buffer:
            buf = train_collector.buffer
            print("main buffer", len(buf), train_collector.get_buffer_idx())
            rv = lambda x: ""
            if args.env == "RoboPushing" and args.object == 'Block':
                rv_mean = [-.105,-.05,.8725, -.105,-.05,.824, -.105,-.05,.824, 0,0,0, 0,0,0.03]
                rv_variance = [.2,.26,.0425, .2,.26,.001, .2,.26,.001, .2,.26,.0425, .2,.26,.0425]

                rv = lambda x: (x * rv_variance) + rv_mean
            print(args.env, args.object, args.target_mode)
            
            if args.env == "SelfBreakout" and args.object == 'Block':
                if args.target_mode:
                    rv_mean = [84 // 2, 84 // 2, 0,0,1, 32, 84 // 2, 0,0,0]
                    rv_variance = [84 // 2, 84 // 2, 2,1,1, 10, 84 // 2, 2,1,1]
                    rv = lambda x: (x * rv_variance) + rv_mean
                else: 
                    print(args.num_instance)
                    num_instances = args.num_instance
                    param_mean = [32, 84 // 2, 0,0,0]
                    param_variance = [10, 84 // 2, 2,1,1]
                    if len(args.breakout_variant) > 0 and args.breakout_variant not in ["harden_single", "proximity"]:
                        param_mean = list()
                        param_variance = list()
                    rv_mean = np.array(param_mean + [84 // 2, 84 // 2, 0,0,1] + [32, 84 // 2, 0,0,0] * args.num_instance)
                    rv_variance = np.array(param_variance + [84 // 2, 84 // 2, 2,1,1] + [10, 84 // 2, 2,1,1] * args.num_instance)
                    rv = lambda x: (x * rv_variance) + rv_mean
            if not option.policy.sample_HER:
                print("itr, idx, done, term, trunc, reward, inter")
                print("acts, action, mapped, q")
                print("tar, time, param, target, next target, inter state")
                print("obs, obs, rvobs, next obs, rvnext obs")
                for j in range(50):
                    idx = (train_collector.get_buffer_idx() + (j - 100)) % args.buffer_len
                    d, info, tm, r, bi, a, ma, ti, p, t, nt, itr, obs, obs_n = buf[idx].done, buf[idx].info, buf[idx].terminate, buf[idx].rew, buf[idx].inter, buf[idx].act, buf[idx].mapped_act, buf[idx].time, buf[idx].param, buf[idx].target, buf[idx].next_target, buf[idx].inter_state, buf[idx].obs, buf[idx].obs_next
                    # print(obs.shape, rv_variance.shape, rv_mean.shape)
                    
                    print(j, idx, d, tm, info["TimeLimit.truncated"], r, bi, 
                        "\nacts", a, ma, option.policy.compute_Q(Batch(obs=obs, obs_next = obs_n,info=info, act=a), False), 
                        "\ntar", ti, p, t, nt, itr, 
                        "\nobs", obs, rv(obs), obs_n, rv(obs_n))

            if option.policy.is_her:
                print("itr, idx, done, term, trunc, reward, inter")
                print("acts, action, mapped, q")
                print("tar, time, param, target, next target, inter state")
                print("obs, obs, rvobs, next obs, rvnext obs")
                hrb = option.policy.learning_algorithm.replay_buffer
                if len(hrb) > 10:
                    print("hindsight buffer", len(hrb), option.policy.learning_algorithm.get_buffer_idx())
                    for j in range(50):
                        idx = (option.policy.learning_algorithm.get_buffer_idx() + (j - 100)) % args.buffer_len
                        dh, infoh, tmh, rh, ih, ah, mah, tih, ph, th, nth, itrh, obsh, obs_nh = hrb[idx].done, hrb[idx].info, hrb[idx].terminate, hrb[idx].rew, hrb[idx].inter, hrb[idx].act, hrb[idx].mapped_act, hrb[idx].time, hrb[idx].param, hrb[idx].target, hrb[idx].next_target, hrb[idx].inter_state, hrb[idx].obs, hrb[idx].obs_next
                        print(j, idx, dh, tmh, infoh["TimeLimit.truncated"], rh, ih, 
                            "\nacts", ah, mah, option.policy.compute_Q(Batch(obs=obsh, obs_next = obs_nh,info=infoh, act=ah), False),
                            "\ntar",tih,  ph,  th, nth, itrh, 
                            "\nobs", obsh, rv(obsh), obs_nh, rv(obs_nh))