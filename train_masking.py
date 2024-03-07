import numpy as np
import os, torch
from arguments import get_args
from Environment.Environments.initialize_environment import initialize_environment
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from train_interaction import init_names
from Buffer.train_test_buffers import generate_buffers
from Causal.Utils.get_error import get_error, error_types
from Causal.active_mask import ActiveMasking
from Causal.interaction_model import make_name
from Causal.dummy_interaction import ActionDummyInteraction
from Causal.Training.test_full import test_full
from Graph.graph import Graph, load_graph
from Option.primitive_option import PrimitiveOption
from State.feature_selector import construct_object_selector

def train_mask(args):
    print(args)
    torch.cuda.set_device(args.torch.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    environment, record = initialize_environment(args.environment, args.record)

    object_names = init_names(args.train.train_edge)

    # initializes the graph or loads it from args.record.load_dir
    action_dummy = ActionDummyInteraction(environment.action_shape, environment.discrete_actions, environment.num_actions)
    action_option = PrimitiveOption(args, None, environment)
    graph = Graph(environment.object_names, action_dummy, action_option) if "graph.gm" not in os.listdir(args.record.load_dir) or args.record.refresh else load_graph(args.record.load_dir, args.torch.gpu)

    full_model = torch.load(os.path.join(args.record.load_dir, make_name(object_names) + "_inter_model.pt"))
    full_model.regenerate(environment)
    args.target_select, args.full_parent_select, args.additional_select, args.additional_selectors, \
            args.padi_selector, args.parent_select, args.inter_select = full_model.extractor.get_selectors()

    print(args.inter.load_intermediate)
    # print(object_names.primary_parent, graph.nodes[object_names.primary_parent].option)
    print(graph.nodes[object_names.primary_parent].option, graph.nodes[object_names.primary_parent].interaction.active_mask, graph.nodes[object_names.primary_parent].interaction.mask.active_mask)

    if args.inter.load_intermediate: buffer = load_from_pickle("/hdd/datasets/object_data/temp/rollouts.pkl")
    else: buffer = generate_buffers(environment, args, object_names, full_model, train=False)
    if args.inter.save_intermediate: save_to_pickle("/hdd/datasets/object_data/temp/rollouts.pkl", buffer)

    buffer.inter[:len(buffer)] = get_error(full_model, buffer, error_type=error_types.INTERACTION_BINARIES).squeeze()
    print(get_error(full_model, buffer, error_type=error_types.INTERACTION_BINARIES)[:100], buffer.inter[:10], len(buffer), len(get_error(full_model, buffer, error_type=error_types.INTERACTION_RAW)))

    test_full(full_model, buffer, args, object_names, environment)
    print(object_names.primary_parent, graph.nodes[object_names.primary_parent].option)
    print (args.mask)
    full_model.mask = ActiveMasking(buffer, full_model, args.mask.min_sample_difference, args.mask.var_cutoff, 
        graph.nodes[object_names.primary_parent].interaction.active_mask, 
        parent_max_num= environment.num_actions if (environment.discrete_actions and object_names.primary_parent == "Action") else 10000,
        num_samples=args.mask.num_samples, sample_grid = args.mask.sample_grid, dynamics_difference=args.mask.dynamics_difference)
    full_model.active_mask = full_model.mask.active_mask
    full_model.active_select = construct_object_selector([object_names.target], environment, masks=[full_model.active_mask])

    graph.nodes[object_names.target].interaction = full_model
    graph.add_to_chain(object_names.target)
    if len(args.record.save_dir) > 0: 
        graph.save(args.record.save_dir)
        graph = load_graph(args.record.save_dir, -1)
    print(args.record.save_dir, graph.nodes.Action.option, graph.nodes.Action.option.sampler, graph.nodes.Action.option.sampler.mask)
    print(full_model.active_mask, full_model.mask.active_mask, graph.nodes.Action.option.sampler.mask.active_mask, len(graph.nodes[object_names.target].interaction.mask.filtered_active_set))
    print(graph.nodes.Action.option.name, graph.nodes.Action.option.interaction_model)

if __name__ == '__main__':
    args = get_args()
    train_mask(args)