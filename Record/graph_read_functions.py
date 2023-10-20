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
AAT = "active at "
PAT = "passive at "
IAT = "interaction at "
FERR = "flat error:"
SFP = "soft FP:"
SFN = "soft FN:"

def read_full_inter(filename):
    # reads a file for the performance the the iterations
    file = open(filename, 'r')
    test_at = list()
    vals = {"fam": list(), "aat": list(), "pat": list(), "iat": list()}
    for line in file.readlines():
        if line.find(FAM) != -1:
            score = float(line.split(" ")[-1])
            vals["fam"].append(score)
        if line.find(AAT) != -1:
            itr_at = int(line.split(", ")[0].split(" ")[-1])
            test_at.append(itr_at)
            score = float(line.split(", ")[1].split(": ")[-1])
            vals["aat"].append(score)
        if line.find(PAT) != -1:
            score = float(line.split(", ")[1].split(": ")[-1])
            vals["pat"].append(score)
        if line.find(IAT) != -1:
            score = float(line.split(", ")[1].split(": ")[-1])
            vals["iat"].append(score)
        if line.find(FERR):
            rex = re.compile(r'\W+')
            nline = rex.sub(' ', line)
            ferrs = nline.split(' ')
            for i, v in enumerate(ferrs[1:]):
                if FERR + str(i) in vals: vals[FERR + str(i)].append(float(v.strip("[]")))
                else: vals[FERR + str(i)] = [float(v.strip("[]"))]
        if line.find(SFP):
            rex = re.compile(r'\W+')
            nline = rex.sub(' ', line)
            sfps = nline.split(' ')
            for i, v in enumerate(sfps[1:]):
                if SFP + str(i) in vals: vals[SFP + str(i)].append(float(v.strip("[]")))
                else: vals[SFP + str(i)] = [float(v.strip("[]"))]
        if line.find(SFN):
            rex = re.compile(r'\W+')
            nline = rex.sub(' ', line)
            sfns = nline.split(' ')
            for i, v in enumerate(sfns[1:]):
                if SFN + str(i) in vals: vals[SFN + str(i)].append(float(v.strip("[]")))
                else: vals[SFN + str(i)] = [float(v.strip("[]"))]


    return test_at, vals

def group_assess(read_fn, folder):
    results = list()
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
            else:
                result = read_iterations(file_path)
            mml = dict()
            for k in result[1].keys():
                min_r, max_r, last_r = np.min(result[1][k]), np.max(result[1][k]), result[1][k][-1]
                mml[k] = (min_r, max_r, last_r)
            results.append((file_path, mml))
            print(file_path, mml)
    return results

