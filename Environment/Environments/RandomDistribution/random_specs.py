variants = {"default": (False, False, 3, 1, 2, 2, True, False, 0.0, "Gaussian", 0.0005, False, -1, -1),
			"small": (False, False, 1, 1, 1, 2, True, False, 0.0, "Gaussian", 0.0005, False, -1, -1),
			"passive_only": (False, False, 3, 1, 0, 2, True, False, 0.0, "Gaussian", 0.000, True, -1, -1),
			"passive_only_noise": (False, False, 3, 1, 0, 2, True, False, 0.0, "Gaussian", 0.0005, False, -1, -1),
			"conditional": (False, False, 3, 1, 2, 2, True, True, 0.0, "Gaussian", 0.0005, False, -1, -1),
			"conditional_rare": (False, False, 3, 1, 2, 2, True, True, 0.6, "Gaussian", 0.0005, True, -1, -1),
			"conditional_common": (False, False, 3, 1, 2, 2, True, True, -0.7, "Gaussian", 0.0005, True, -1, -1),
			"conditional_passive": (False, False, 3, 1, 2, 2, True, True, 0.0, "Gaussian", 0.0005, True, -1, -1),
			"cp_many": (False, False, 4, 1, 5, 2, True, True, 0.0, "Gaussian", 0.0005, True, -1, -1),
			"cp_multi": (False, False, 4, 3, 3, 2, True, True, 0.0, "Gaussian", 0.0005, True, -1, -1),
			"cp_multi_small": (False, False, 3, 3, 2, 2, True, True, 0.0, "Gaussian", 0.0005, True, -1, -1),
			"conditional_small": (False, False, 1, True, 1, 2, True, True, 0.0, "Gaussian", 0.0005, True, -1, -1),
			"zero_passive": (False, False, 4, True, 3, 2, True, True, 0.0, "Gaussian", 0.0005, False, -1, -1),
			
			"multi_passive": (False, False, 6, 1, 6, 1, True, True, 0.0, "Gaussian", 0.0005, True, 2, 6),
			"multi_small": (False, True, 1, 1, 1, 1, True, True, 0.0, "Gaussian", 0.0005, False, 1, 2),
            "multi_passive_small": (False, True, 1, 1, 1, 1, True, True, 0.0, "Gaussian", 0.0005, True, 1, 2)}


# discrete_actions, allow uncontrollable, num_objects, multi_instanced, num_related, relate_dynamics, conditional, conditional_weight, distribution, noise_percentage, require_passive (dynamics), min live, max live
