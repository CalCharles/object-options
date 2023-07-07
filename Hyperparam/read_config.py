import yaml
import copy
from State.object_dict import ObjDict
from Hyperparam.Argsets.full_args import full_args
from Hyperparam.Argsets.hype_args import hype_args
from Hyperparam.Argsets.cdl_args import cdl_args
from Hyperparam.Argsets.ride_args import ride_args

arg_dicts = {
    "full": full_args,
    "hints": full_args,
    "hype": hype_args,
}

def read_config(pth):
    # copied from https://pynative.com/python-yaml/
    with open(pth) as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print("error: ", exception)
    data = construct_namespace(data)
    return data

def construct_namespace(data):
    args = ObjDict()
    def add_data(add_dict, data_dict, exp_dict):
        if data_dict is None: # if only the name is returned, create a new dict to fill with values
            data_dict = ObjDict()
        for key in exp_dict.keys():
            if type(exp_dict[key]) == dict or type(exp_dict[key]) == ObjDict: # if we have a dict recursively call
                new_add = ObjDict()
                add_dict[key] = add_data(new_add, data_dict[key] if key in data_dict else dict(), exp_dict[key])
            else: # exp_dict contains the default values
                add_dict[key] = data_dict[key] if key in data_dict else exp_dict[key]
                if type(exp_dict[key]) == float: add_dict[key] = float(add_dict[key])
                # handling special characters
                if add_dict[key] == "None": add_dict[key] = None
                elif add_dict[key] == "[]" or (type(add_dict[key]) == list and len(add_dict[key]) == 0): add_dict[key] = list()
                elif type(exp_dict[key]) == list and key in data_dict:
                    if type(data_dict[key]) != str:
                        add_dict[key] = [data_dict[key]]
                    else:
                        try:
                            add_dict[key] = [float(v) for v in add_dict[key].split(" ")]
                        except ValueError as e:
                            add_dict[key] = add_dict[key].split(" ")
        return add_dict
    if "arg_dict" in data:
        if data["arg_dict"] == "hype":
            expected_args = hype_args
        elif data["arg_dict"] == "cdl":
            expected_args = cdl_args
        elif data["arg_dict"] == "ride":
            expected_args = ride_args
        elif data["arg_dict"] == "full":
            expected_args = full_args
        else:
            raise ValueError('invalid argument set: ' + str(data["arg_dict"]) + ". Valid choices: hype, full, hints" )
    else:
        expected_args = full_args
    args = add_data(args, data, expected_args)
    for key in data.keys():
        if key.find("_net") != -1: # _net 
            vargs = add_data(ObjDict(), data[key], args.network)
            args[key] = vargs
            args[key].gpu = args.torch.gpu
    print(args)
    return args