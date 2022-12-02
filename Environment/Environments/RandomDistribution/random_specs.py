variants = {"default": (False, False, 3, True, 2, True, False, 0.0, "Gaussian", 0.0005, False),
			"small": (False, False, 1, True, 1, True, False, 0.0, "Gaussian", 0.0005, False),
			"passive_only": (False, False, 3, True, 0, True, False, 0.0, "Gaussian", 0.000, True),
			"passive_only_noise": (False, False, 3, True, 0, True, False, 0.0, "Gaussian", 0.0005, False),
			"conditional": (False, False, 3, True, 2, True, True, 0.0, "Gaussian", 0.0005, False),
			"conditional_rare": (False, False, 3, True, 2, True, True, 0.4, "Gaussian", 0.0005, False),
			"conditional_passive": (False, False, 3, True, 2, True, True, 0.0, "Gaussian", 0.0005, True),
			"conditional_small": (False, False, 1, True, 1, True, True, 0.0, "Gaussian", 0.0005, True),
			"zero_passive": (False, False, 4, True, 3, True, True, 0.0, "Gaussian", 0.0005, False)}


# discrete_actions, allow uncontrollable, num_objects, multi_instanced, num_related, relate_dynamics, conditional, distribution, noise_percentage, require_passive (dynamics)