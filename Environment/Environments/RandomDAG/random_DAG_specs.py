variants = {"3-chain": (False, False, "3-chain", -1, 1, 4,4, True, False, True, 0.0, "Gaussian", 0.0005, True, -1, -1),
            "2-in": (False, False, "2-inchain", -1, 1, 4,4, False, True, True, 0.0, "Gaussian", 0.0005, True, -1, -1),
            "3-in": (False, False, "3-inchain", -1, 1, 4,4, False, True, True, 0.0, "Gaussian", 0.0005, True, -1, -1),
            "1-in": (False, False, "3-chain", -1, 1, 4,4, False, True, True, 0.0, "Gaussian", 0.0005, True, -1, -1),
            "1-hdim": (False, False, "3-chain", -1, 1, 10,10, False, True, True, 0.0, "Gaussian", 0.0005, True, -1, -1),
            "1-rare": (False, False, "3-chain", -1, 1, 4,4, False, True, True, 2.0, "Gaussian", 0.0005, True, -1, -1),
            "multi-in": (False, False, "3-multi-chain", -1, 1, 4,4, True, True, True, 0.0, "Gaussian", 0.0005, True, -1, -1)}

# discrete_actions, allow uncontrollable, graph_skeleton, num_nodes (only used for random graph skeleton), 
# multi_instanced, min dim, max dim, instant_update, relate_dynamics, conditional, conditiona_value
# distribution, noise_percentage, require_passive (dynamics), min live, max live

def parse_edges(graph_skeleton):
    if graph_skeleton == "rand":
        # TODO: generate a random skeleton
        pass
    elif graph_skeleton == "3-chain":
        return [("Action", "$B"), ("$B", "$C")], ["Action", "$B", "$C", "Reward", "Done"]
    elif graph_skeleton == "2-in":
        return [("Action", "$C"), ("$B", "$C")], ["Action", "$B", "$C", "Reward", "Done"]
    elif graph_skeleton == "2-inchain":
        return [("Action", "$B"),("$C", "$D"), ("$B", "$D")], ["Action", "$B", "$C", "$D", "Reward", "Done"]
    elif graph_skeleton == "3-inchain":
        return [("Action", "$B"),("$C", "$E"), ("$B", "$E"), ("$D", "$E")], ["Action", "$B", "$C", "$D", "$E", "Reward", "Done"]
    elif graph_skeleton == "2-multi":
        return [("Action", "$B"),("$B", "$C", "$D", "$E")], ["Action", "$B", "$C", "$D", "$E", "Reward", "Done"]
    elif graph_skeleton == "3-multi-chain":
        return [("Action", "$A", "$B"), ("$B", "$C")], ["Action", "$A", "$B", "$C", "Reward", "Done"]
