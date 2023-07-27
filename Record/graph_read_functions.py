# graph read files
import numpy as np


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
    return test_at, vals