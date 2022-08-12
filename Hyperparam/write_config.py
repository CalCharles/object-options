import yaml
import copy
import os
import numpy as np
from State.object_dict import ObjDict
from Record.file_management import create_directory
from Hyperparam.read_config import read_config

def write_config(pth, args):
    prefix_val = "  "
    def get_lines(args, prefix):
        all_lines = list()
        for key,value in args.items():
            write_lines = list()
            if value is None:
                write_lines += [prefix + key + ": None\n"]
            elif type(value) == str and len(value) == 0:
                continue
            elif type(value) == list:
                if len(value) == 0:
                    write_lines += [prefix + key + ': []\n']
                else:
                    write_lines += [prefix + key + ': ' + ' '.join(str(v) for v in value) + '\n']
            elif type(value) == ObjDict:
                write_lines += [prefix + key + ":\n"]
                write_lines += get_lines(value, prefix = prefix + prefix_val)
            else:
                write_lines += [prefix + key + ": " + str(value) + '\n']
            all_lines += write_lines
        return all_lines
    lines = get_lines(args, "")
    lines = ['---\n'] + lines
    lines += ['...']
    with open(create_directory(pth, drop_last = True), 'w') as f:
        f.writelines(lines)

def write_multi_config(multi_pth):
    # read in base config

    # read in hyperparameter grid file
    with open(multi_pth, 'r') as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print("error: ", exception)
    data = ObjDict(data)
    multi_filename = data["metaparam"]["multi_filename"]
    log_endpoint = data["metaparam"]["log_endpoint"]
    graph_endpoint = data["metaparam"]["graph_endpoint"]
    yaml_endpoint = data["metaparam"]["yaml_endpoint"]
    save_endpoint = data["metaparam"]["save_endpoint"]
    bash_path = data["metaparam"]["bash_path"]
    runfile = data["metaparam"]["runfile"]
    gpu = data["metaparam"]["gpu"]
    match = data["metaparam"]["match"]
    base_config = read_config(data["metaparam"]["base_config"])
    del data["metaparam"]

    # generate list of list of names to reach each desired param
    name_paths = list()
    # also get the corresponding values to that list
    all_settings = list()
    def explore_name_paths(data_dict, current_path):
        for key in data_dict.keys():
            print(key, type(data_dict[key]))
            if type(data_dict[key]) == str:
                name_paths.append(current_path + [key])
                all_settings.append(data_dict[key])
            elif type(data_dict[key]) == dict:
                explore_name_paths(data_dict[key], current_path + [key])
            else:
                name_paths.append(current_path + [str(key)])
                all_settings.append(data_dict[key])                
    explore_name_paths(data, list())
    print(all_settings, name_paths)

    # convert a comma separated string value to list of the desired type
    def convert_single(base_config, name_path, str_value):
        final_val = base_config
        for n in name_path:
            final_val = final_val[n]
        final_type = type(final_val)
        full_values = list()
        if type(str_value) != str:
            return [str_value] # singleton values returned wrapped
        print(str_value.split(','))
        for v in str_value.split(','): # commas are not allowed in config files for this reason
            if final_type == list:
                try:
                    full_values.append([float(nv) for nv in v.split(" ")])
                except ValueError as e:
                    full_values.append(v.split(" "))
            else:
                full_values.append(final_type(v)) # casts other types without change
        print(full_values)
        return full_values

    # construct list (per name) of list of settings for that variable
    all_settings_grid = list()
    for setting, name_path in zip(all_settings, name_paths):
        print(setting, name_path)
        all_settings_grid.append(convert_single(base_config, name_path, setting))
    # all combinations of indexes
    if match:
        comb_array = np.array([[i for _ in range(len(all_settings_grid))] for i in range(len(all_settings_grid[0]))])
    else:
        comb_array = np.array(np.meshgrid(*[np.array(list(range(len(n)))) for n in all_settings_grid])).T.reshape(-1, len(all_settings_grid))
    print(all_settings_grid, [np.array(list(range(len(n)))) for n in all_settings_grid], np.array(np.meshgrid(*[np.array(list(range(len(n)))) for n in all_settings_grid])).T.reshape(-1, len(all_settings_grid)))
    print(comb_array)
    # create a config file corresponding to one combination of indexes
    def set_alt_network_values(base_config, name_path, setv):
        for b in base_config.keys():
            if b.find("_net") != -1:
                set_val = base_config[b]
                for n in name_path[1:-1]:
                    set_val = set_val[n]
                set_val[name_path[-1]] = setv


    def create_config(base_config, combination, idx):
        config = copy.deepcopy(base_config)
        for c, setting, name_path in zip(combination, all_settings_grid, name_paths):
            set_val = config
            for n in name_path[:-1]:
                if n == "network":
                    set_alt_network_values(config, name_path, setting[c])
                set_val = set_val[n]
            set_val[name_path[-1]] = setting[c]
            print(name_path[-1], setting[c])
        name = multi_filename + "_".join([str(c) for c in combination])
        config.record.record_graphs = os.path.join(graph_endpoint, name)
        config.record.log_filename = os.path.join(log_endpoint, name + '.log')
        config.record.save_dir = os.path.join(save_endpoint, name)
        config.torch.gpu = gpu
        config.collect.stream_print_file = os.path.join(log_endpoint, name + '_stream.txt')
        config.hyperparam = ObjDict()
        config.hyperparam.name_orders = ["+".join(name_path) for name_path in name_paths]
        return name, config
    
    # get corresponding config dicts and names for the files
    all_args = list()
    all_names = list()
    print(comb_array)
    for i, combination in enumerate(comb_array):
        name, config = create_config(base_config, combination, i)
        all_args.append(config)
        all_names.append(name)

    # write the config files to locations
    config_names = list()
    for config, name in zip(all_args, all_names):
        config_names.append(os.path.join(yaml_endpoint, name + '.yaml'))
        write_config(os.path.join(yaml_endpoint, name + '.yaml'), config)

    def write_bash(runfile, config_names, bash_path):
        file_lines = list()
        for cfn, n in zip(config_names, all_names):
            file_lines.append("python " + runfile + " --config " + cfn + " > " + os.path.join(create_directory(log_endpoint), n + '.txt\n'))
        with open(create_directory(bash_path, drop_last = True), 'w') as bash_file:
            bash_file.writelines(file_lines)
    write_bash(runfile, config_names, bash_path)
    return all_args, all_names
