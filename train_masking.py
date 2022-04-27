if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    environment = initialize_environment(args, set_save=False)

    model = load_interaction(args.load_path)

    if args.load_intermediate: buffer = load_from_pickle("/hdd/datasets/counterfactual_data/temp/rollouts.pkl")
    else: buffer = generate_buffers(args, train=False)

    buffer.inter = get_error(full_model, buffer, error_type=error_types.interaction)

    model.masking = ActiveMasking(rollouts, model, args.min_variance, args.num_samples)