# graph read files
import numpy as np
import re
import os


TeR = "test_reward: "

def read_ts_format(filename):
    file = open(filename, 'r')
    steps = list()
    values = list()
    for line in file.readlines():
        if line.find(TeR) != -1:
            # print(line[line.find(TeR):].split(" "))
            # print(line[line.find("Epoch"):].split(" "))
            steps.append(int(line[line.find("Epoch"):].split(" ")[1][1:-1]) * 1000) 
            values.append(float(line[line.find(TeR):].split(" ")[1]))
    print(steps, values)
    return steps, {"scores": values}


paddle_start = "observation: ['param', 'target', 'param_relative']"
paddle_end = "performance comparison"
ball_start = "observation: ['param', 'parent', 'target', 'parent_relative']"
gripper_start = "Gripper_train"
gripper_end = "performance comparison"
block_start = "Block_train"

keys = ["Steps", "Hit", "train", "test"]
S = "Steps"
H = "Hit"
TR = "train"
TE = "test"

def read_iterations(filename, hitmiss=False, mode = ""):
    # reads a file for the performance the the iterations
    file = open(filename, 'r')
    test_at = list()
    test_vals = {"scores": list()}
    train_mode = False
    started = True if len(mode) == 0 else False
    for line in file.readlines():
        if mode == "Paddle":
            if line.find(paddle_start) != -1:
                started = True
            if line.find(paddle_end) != -1:
                started = False
            if not started:
                continue
        elif mode == "Ball":
            if line.find(ball_start) != -1:
                started = True
            if not started:
                continue
        elif mode == "Gripper":
            if line.find(gripper_start) != -1:
                started = True
            if line.find(gripper_end) != -1:
                started = False
            if not started:
                continue
        elif mode == "Block":
            if line.find(block_start) != -1:
                started = True
            if not started:
                continue
        if line.find(TR) != -1:
            train_mode = True
        if line.find(TE) != -1:
            train_mode = False
        # print(filename, started, line[:5])

        if not train_mode: 
            # print(line, S, line.find(S), line.find(H))
            if line.find(H) != -1:
                if hitmiss:
                    val = float(line.split(":")[1].split(", ")[3])
                else:
                    val = float(line.split(", ")[-1])
                test_vals["scores"].append(val)
                # print("add", val)
        else:
            if line.find(S) != -1:
                at = int(line[line.find(S)+7:].split(", ")[0])
                # test_at.append(at * (1 + 2 * (1-np.exp(-at/2000000))) + 10000)
                test_at.append(at)
    print(test_at, test_vals)
    return test_at, test_vals

IS = "init_stage"
BE = "Breakout episode score"
RE = "end of episode"
TE = "test"

def read_iterations_cdl(filename):
    # reads a file for the performance the the iterations
    file = open(filename, 'r')
    test_at = list()
    test_vals = {"scores": list()}
    train_mode = False
    for line in file.readlines():
        if line.find(IS) != -1:
            itr_at = int(line.split(", ")[0].split("/")[0])
        if line.find(RE) != -1:
            score = float(line.split(",")[0].split(":")[1].strip())
            test_at.append(itr_at)
            test_vals["scores"].append(score)
        if line.find(BE) != -1:
            score = float(line.split(":")[1].strip())
            test_at.append(itr_at)
            test_vals["scores"].append(score)
    return test_at, test_vals

FAM = "flat_average_miss"
IAM = "inter_average_miss"
AAT = "active_"
AT = "at "
PAT = "passive_"
IAT = "interaction at "
FERR = "flat error:"
TRTE = "trace rate:"
SFP = "soft FP:"
SFN = "soft FN:"

def read_full_inter(filename):
    # reads a file for the performance the the iterations
    file = open(filename, 'r')
    test_at = list()
    vals = {"fam": list(), "aat": list(), "pat": list(), "iat": list()}
    for line in file.readlines():
        try:
            if line.find(FAM) != -1:
                score = float(line.split(" ")[-1])
                vals["fam"].append(score)
            if line.find(AAT) != -1 and line.find(AT) != -1:
                itr_at = int(line.split(", ")[0].split(" ")[-1])
                test_at.append(itr_at)
                score = float(line.split(", ")[1].split(": ")[-1])
                vals["aat"].append(score)
            if line.find(PAT) != -1 and line.find(AT) != -1:
                score = float(line.split(", ")[1].split(": ")[-1])
                vals["pat"].append(score)
            if line.find(IAT) != -1:
                score = float(line.split(", ")[1].split(": ")[-1])
                vals["iat"].append(score)
            if line.find(FERR) != -1:
                rex = re.compile(r'\s+')
                nline = rex.sub(' ', line)
                ferrs = nline.split(' ')
                # print("ferr", line, ferrs, nline)
                for i, v in enumerate(ferrs[2:]):
                    try:
                        if FERR + str(i) in vals: vals[FERR + str(i)].append(float(v.strip("[]")))
                        else: vals[FERR + str(i)] = [float(v.strip("[]"))]
                    except ValueError as e:
                        continue
            if line.find(TRTE) != -1:
                rex = re.compile(r'\s+')
                nline = rex.sub(' ', line)
                trtes = nline.split(' ')
                # print("ferr", line, ferrs, nline)
                for i, v in enumerate(trtes[2:]):
                    try:
                        if TRTE + str(i) in vals: vals[TRTE + str(i)].append(float(v.strip("[]")))
                        else: vals[TRTE + str(i)] = [float(v.strip("[]"))]
                    except ValueError as e:
                        continue
            if line.find(SFP) != -1:
                rex = re.compile(r'\s+')
                nline = rex.sub(' ', line)
                sfps = nline.split(' ')
                # print("false positive", line, sfps, nline)
                for i, v in enumerate(sfps[2:]):
                    try:
                        if SFP + str(i) in vals: vals[SFP + str(i)].append(float(v.strip("[]")))
                        else: vals[SFP + str(i)] = [float(v.strip("[]"))]
                    except ValueError as e:
                        continue
            if line.find(SFN) != -1:
                rex = re.compile(r'\s+')
                nline = rex.sub(' ', line)
                sfns = nline.split(' ')
                for i, v in enumerate(sfns[2:]):
                    try:
                        if SFN + str(i) in vals: vals[SFN + str(i)].append(float(v.strip("[]")))
                        else: vals[SFN + str(i)] = [float(v.strip("[]"))]
                    except ValueError as e:
                        continue
        except ValueError as e:
            continue

    return test_at, vals

TRBASE = "train_baseline_"
TESTBASE = "test_baseline_"
ACL = "active loss:"
SP = "soft Positive:"
SN = "soft Negative:"
TRW = "trace weight"


def read_full_acbase(filename):
    # reads a file for the performance the the iterations
    file = open(filename, 'r')
    test_at = list()
    vals = {"act": list()}
    training_mode = False
    test_mode = False
    for line in file.readlines():
        try:
            if line.find(TRBASE) != -1:
                training_mode = True
                test_mode = False
            if line.find(TESTBASE) != -1:
                test_mode = True
                training_mode = False
            if training_mode:
                if line.find(ACL) != -1 and line.find(IAT) != -1:
                    itr_at = int(line.split(", ")[0].split(" ")[-1])
                    test_at.append(itr_at)
                    score = float(line.split(", ")[2].split(": ")[-1])
                    vals["act"].append(score)
                if line.find(FERR) != -1:
                    rex = re.compile(r'\s+')
                    nline = rex.sub(' ', line)
                    ferrs = nline.split(' ')
                    for i, v in enumerate(ferrs[2:]):
                        try:
                            if FERR + str(i) in vals: vals[FERR + str(i)].append(float(v.strip("[]")))
                            else: vals[FERR + str(i)] = [float(v.strip("[]"))]
                        except ValueError as e:
                            pass
                if line.find(SP) != -1:
                    rex = re.compile(r'\s+')
                    nline = rex.sub(' ', line)
                    sfps = nline.split(' ')
                    for i, v in enumerate(sfps[2:]):
                        try:
                            if SP + str(i) in vals: vals[SP + str(i)].append(float(v.strip("[]")))
                            else: vals[SP + str(i)] = [float(v.strip("[]"))]
                        except ValueError as e:
                            continue
                if line.find(SN) != -1:
                    rex = re.compile(r'\s+')
                    nline = rex.sub(' ', line)
                    sfns = nline.split(' ')
                    for i, v in enumerate(sfns[2:]):
                        try:
                            if SN + str(i) in vals: vals[SN + str(i)].append(float(v.strip("[]")))
                            else: vals[SN + str(i)] = [float(v.strip("[]"))]
                        except ValueError as e:
                            continue
            if test_mode:
                if line.find(TRW) != -1:
                    rex = re.compile(r'\s+')
                    nline = rex.sub(' ', line)
                    trtes = nline.split(' ')
                    for i, v in enumerate(trtes[2:]):
                        try:
                            if TRW + str(i) in vals: vals[TRW + str(i)].append(float(v.strip("[]")))
                            else: vals[TRW + str(i)] = [float(v.strip("[]"))]
                        except ValueError as e:
                            continue
        except ValueError as e:
            continue

    return test_at, vals



def group_assess(read_fn, folder):
    results = list()
    aggregated = dict()
    file_keys = list()
    for path, subdirs, files in os.walk(folder):
        for name in files:
            file_path = os.path.join(path, name)
            if read_fn.find("stack") != -1:
                if read_fn.find("gripper") != -1:
                    mode = "Gripper"
                if read_fn.find("paddle") != -1:
                    mode = "Paddle"
                if read_fn.find("block") != -1:
                    mode = "Block"
                if read_fn.find("ball") != -1:
                    mode = "Ball"
                result = read_iterations(file_path, hitmiss=True, mode=mode)
            elif read_fn.find("ride") != -1:
                result = read_ts_format(file_path)
            elif read_fn.find("cdl") != -1:
                result = read_iterations_cdl(file_path)
            elif read_fn.find("full") != -1:
                result = read_full_inter(file_path)
            elif read_fn.find("acbase") != -1:
                result = read_full_acbase(file_path)
            else:
                result = read_iterations(file_path)
            mml = dict()
            for k in result[1].keys():
                if len(result[1][k]) > 0:
                    min_r, max_r, last_r, num_steps = np.min(result[1][k]), np.max(result[1][k]), result[1][k][-1], result[0][-1]
                else:
                    min_r, max_r, last_r, num_steps = -1,-1,-1,0
                mml[k] = (min_r, max_r, last_r, num_steps)
            results.append((file_path, mml))
            print(file_path, mml)
            if file_path.find("trial_") != -1 and file_path.find(".txt") != -1:
                file_keys.append(file_path)
                key = file_path.split("/")[-1]
                key = key[:key.find("trial_")]
                if key in aggregated:
                    aggregated[key].append(mml)
                else:
                    aggregated[key] = [mml]
    new_aggregated = dict()
    for key in aggregated.keys():
        if key not in new_aggregated:
            new_aggregated[key] = dict()
        for kdict in aggregated[key]:
            for ltype in kdict.keys():
                if ltype in new_aggregated[key]:
                    new_aggregated[key][ltype].append(kdict[ltype])
                else:
                    new_aggregated[key][ltype] = [kdict[ltype]]
    namekeys = list(aggregated.keys())
    namekeys.sort()
    file_keys.sort()
    for fkey, key in zip(file_keys, namekeys):
        for ltype in new_aggregated[key].keys():
            if ltype == "flat error:1":
                print(key, ltype, np.mean(np.array(new_aggregated[key][ltype]), axis=0))
                print(key, ltype, np.std(np.array(new_aggregated[key][ltype]), axis=0))
                print(np.array(new_aggregated[key][ltype]))
            if ltype == "trace rate:1":
                print(key, ltype, np.mean(np.array(new_aggregated[key][ltype]), axis=0))
        print(fkey)
        if "flat error:1" not in list(new_aggregated[key].keys()):
            print("crashed", key)
            print(fkey)
    # print(new_aggregated)
    return results, aggregated

def read_iterations_train(folder, target):
    files = os.listdir(folder)
    dir_name = os.path.dirname(target)
    try:
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")    
    for filename in files:
        iters, vals = list(), list()
        with open(os.path.join(folder, filename), 'r') as rf:
            take_next= False
            for line in rf.readlines():
                if line.find("Reward_train:") != -1:
                    itr = line.split(",")[1]
                    print(line, int(itr[itr.find("Steps:") + 6:]))
                    iters.append(int(itr[itr.find("Steps:") + 6:]))
                    take_next = True
                elif take_next:
                    take_next = False
                    val = float(line.split(", ")[4])
                    vals.append(max(0,float(val) / 20))
        with open(os.path.join(target, filename), 'w') as wf:
            for itr, val in zip(iters, vals):
                wf.write(f"{itr},{val}\n")

def perturb(folder, target):
    files = os.listdir(folder)
    dir_name = os.path.dirname(target)
    try:
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")    
    for filename in files:
        iters, vals = list(), list()
        take_next= False
        with open(os.path.join(folder, filename), 'r') as rf:
            with open(os.path.join(target, filename), 'w') as wf:
                i = 0
                for line in rf.readlines():
                    if len(line) > 0:
                        itr, val = line.split(",")
                        for ncnt in range(np.random.randint(0,4)):
                            wf.write(f"{int(itr) + ncnt * 50000},{float(val) + np.random.rand() * 100 + i * 3}\n")
                        i += 1

def convert(folder, target):
    files = os.listdir(folder)
    dir_name = os.path.dirname(target)
    try:
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")    
    for filename in files:
        iters, vals = list(), list()
        take_next= False
        with open(os.path.join(folder, filename), 'r') as rf:
            with open(os.path.join(target, filename), 'w') as wf:
                i = 0
                for line in rf.readlines():
                    if len(line) > 0:
                        itr, val = line.split(",")
                        wf.write(f"{int(itr)},{max(0,(float(val) +150) / 150)}\n")
                        i += 1


def strip_wall(folder, target):
    files = os.listdir(folder)
    dir_name = os.path.dirname(target)
    try:
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")    
    for filename in files:
        iters, vals = list(), list()
        take_next= False
        if filename.find("desktop") == -1:
            with open(os.path.join(folder, filename), 'r') as rf:
                with open(os.path.join(target, filename), 'w') as wf:
                    i = 0
                    for line in rf.readlines():
                        if len(line) > 0 and i != 0:
                            print(line)
                            wall,itr, val = line.split(",")
                            wf.write(f"{int(itr)},{val}")
                        i += 1

def strip_vals(folder, target):
    files = os.listdir(folder)
    dir_name = os.path.dirname(target)
    try:
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")    
    for filename in files:
        iters, vals = list(), list()
        take_next= False
        if filename.find("desktop") == -1:
            with open(os.path.join(folder, filename), 'r') as rf:
                with open(os.path.join(target, filename), 'w') as wf:
                    i = 0
                    for line in rf.readlines():
                        if len(line) > 0 and i != 0:
                            print(line)
                            wall,itr, val = line.split(",")
                            wf.write(f"{int(itr)},{val}")
                        i += 1

def mul_vals(folder, target):
    files = os.listdir(folder)
    dir_name = os.path.dirname(target)
    try:
        os.makedirs(dir_name)
        print(f"Directory '{dir_name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_name}' already exists.")    
    for filename in files:
        iters, vals = list(), list()
        take_next= False
        if filename.find("desktop") == -1:
            with open(os.path.join(folder, filename), 'r') as rf:
                with open(os.path.join(target, filename), 'w') as wf:
                    i = 0
                    for line in rf.readlines():
                        if len(line) > 0 and i != 0:
                            wall,itr, val = line.split(",")
                            wf.write(f"{int(itr) * 500},{val}")
                        i += 1


if __name__ == '__main__':
    import sys
    folder, target = sys.argv[1], sys.argv[2]
    # read_iterations_train(folder, target)
    # convert(folder, target)
    strip_wall(folder, target)
    # mul_vals(folder, target)
