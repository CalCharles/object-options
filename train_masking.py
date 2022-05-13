import numpy as np
import os, torch
from arguments import get_args
from Environment.Environments.initialize_environment import initialize_environment
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from train_interaction import generate_buffers
from Causal.Utils.get_error import get_error, error_types
from Causal.active_mask import ActiveMasking

if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    environment = initialize_environment(args, set_save=False)

    full_model = torch.load(os.path.join(args.record.load_dir, full_model.name + "_inter_model.pt"))

    if args.load_intermediate: buffer = load_from_pickle("/hdd/datasets/counterfactual_data/temp/full_rollouts.pkl")
    else: buffer = generate_buffers(environment, args, object_names, full_model, train=False)
    if args.inter.save_intermediate: save_to_pickle("/hdd/datasets/object_data/temp/rollouts.pkl", (train_buffer, test_buffer))


    buffer.inter = get_error(full_model, buffer, error_type=error_types.INTERACTION)

    model.masking = ActiveMasking(buffer, model, args.min_variance, args.num_samples)