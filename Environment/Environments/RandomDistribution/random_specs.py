variants = {"default": (False, False, 3, 1, 2, True, False, 0.0, "Gaussian", 0.0005, False),
			"small": (False, False, 1, 1, 1, True, False, 0.0, "Gaussian", 0.0005, False),
			"passive_only": (False, False, 3, 1, 0, True, False, 0.0, "Gaussian", 0.000, True),
			"passive_only_noise": (False, False, 3, 1, 0, True, False, 0.0, "Gaussian", 0.0005, False),
			"conditional": (False, False, 3, 1, 2, True, True, 0.0, "Gaussian", 0.0005, False),
			"conditional_rare": (False, False, 3, 1, 2, True, True, 0.6, "Gaussian", 0.0005, True),
			"conditional_common": (False, False, 3, 1, 2, True, True, -0.7, "Gaussian", 0.0005, True),
			"conditional_passive": (False, False, 3, 1, 2, True, True, 0.0, "Gaussian", 0.0005, True),
			"cp_many": (False, False, 4, 1, 5, True, True, 0.0, "Gaussian", 0.0005, True),
			"cp_multi": (False, False, 4, 3, 3, True, True, 0.0, "Gaussian", 0.0005, True),
			"cp_multi_small": (False, False, 3, 3, 2, True, True, 0.0, "Gaussian", 0.0005, True),
			"conditional_small": (False, False, 1, True, 1, True, True, 0.0, "Gaussian", 0.0005, True),
			"zero_passive": (False, False, 4, True, 3, True, True, 0.0, "Gaussian", 0.0005, False)}


# discrete_actions, allow uncontrollable, num_objects, multi_instanced, num_related, relate_dynamics, conditional, distribution, noise_percentage, require_passive (dynamics)
