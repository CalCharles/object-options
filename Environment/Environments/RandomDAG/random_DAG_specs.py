variants = {"2-chain": (False, True, "2-chain", -1, 1, 4,4, True, False, True, 0.0, "Gaussian", 0.000, True, -1, -1, "", 0.0, 50),
            "3-chain": (False, True, "3-chain", -1, 1, 4,4, True, False, True, 0.0, "Gaussian", 0.000, True, -1, -1, "$C", 0.5, 50),
            "3-c-small": (False, True, "3-chain", -1, 1, 4,4, True, False, True, 0.0, "Gaussian", 0.000, False, -1, -1, "", 0.0, 50),
            "4-chain": (False, True, "4-chain", -1, 1, 4,4, True, False, True, 0.0, "Gaussian", 0.007, False, -1, -1, "$D", 0.5, 50),
            "1-in": (False, False, "3-chain", -1, 1, 4,4, False, True, True, 0.0, "Gaussian", 0.000, True, -1, -1, "", 0.0, 50),
            "2-in": (False, False, "2-inchain", -1, 1, 4,4, False, True, True, 0.0, "Gaussian", 0.000, True, -1, -1, "", 0.0, 50),
            "3-in": (False, False, "3-inchain", -1, 1, 4,4, False, True, True, 0.0, "Gaussian", 0.000, True, -1, -1, "", 0.0, 50),
            "1-in-nt": (False, False, "3-chain", -1, 1, 4,4, False, True, True, 0.0, "Gaussian", 0.000, True, -1, -1, "", 0.0, 2),
            "1-hdim": (False, False, "3-chain", -1, 1, 10,10, False, True, True, 0.0, "Gaussian", 0.000, True, -1, -1, "", 0.0, 50),
            "1-rare": (False, False, "3-chain", -1, 1, 4,4, False, True, True, 0.23, "Gaussian", 0.000, True, -1, -1, "", 0.0, 50),
            "multi-in": (False, False, "2-multi", -1, 1, 4,4, True, True, True, 0.0, "Gaussian", 0.000, True, -1, -1, "", 0.0, 50)}

# discrete_actions, allow uncontrollable, graph_skeleton, num_nodes (only used for random graph skeleton), 
# multi_instanced, min dim, max dim, instant_update, relate_dynamics, conditional, conditional_value
# distribution, noise_percentage, require_passive (dynamics), min live, max live, intervention_state, intervention_rate, horizon

def parse_edges(graph_skeleton):
    if graph_skeleton == "rand":
        # TODO: generate a random skeleton
        pass
    elif graph_skeleton == "2-chain":
        return [("Action", "$C")], ["Action", "$C", "Reward", "Done"]
    elif graph_skeleton == "3-chain":
        return [("Action", "$B"), ("$B", "$C")], ["Action", "$B", "$C", "Reward", "Done"]
    elif graph_skeleton == "4-chain":
        return [("Action", "$B"), ("$B", "$C"), ("$C", "$D")], ["Action", "$B", "$C", "$D", "Reward", "Done"]
    elif graph_skeleton == "2-in":
        return [("Action", "$C"), ("$B", "$C")], ["Action", "$B", "$C", "Reward", "Done"]
    elif graph_skeleton == "2-inchain":
        return [("Action", "$B"),("$C", "$D"), ("$B", "$D")], ["Action", "$B", "$C", "$D", "Reward", "Done"]
    elif graph_skeleton == "3-inchain":
        return [("Action", "$B"),("$C", "$E"), ("$B", "$E"), ("$D", "$E")], ["Action", "$B", "$C", "$D", "$E", "Reward", "Done"]
    elif graph_skeleton == "2-multi":
        return [("Action", "$B"),("$B", "$C", "$D", "$E")], ["Action", "$B", "$C", "$D", "$E", "Reward", "Done"]
    elif graph_skeleton == "3-multi-chain":
        return [("Action", "$B"), ("$A", "$C"), ("$B", "$D"), ("$B", "$C"), ("$D", "$C")], ["Action", "$A", "$B", "$D", "$C", "Reward", "Done"]
