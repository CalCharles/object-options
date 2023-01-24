from State.object_dict import ObjDict
import numpy as np

state_forms = ["target", "next_target", "target_diff", "parent_state", "rel_state", "obs", "done"]

def get_states(extractor, data, norm, object_names):
    factored_state = data[0]
    last_done = [0.0] # dones are shifted one back because we want to zero out the invalid frame (with the next state as the first state of the next episode)
    states = {sf: list() for sf in state_forms}
    for i, next_factored_state in enumerate(data[1:]):
        # assign selections of the state
        states["target"].append(extractor.target_selector(factored_state))
        states["next_target"].append(extractor.target_selector(next_factored_state))
        states["target_diff"].append(extractor.target_selector(next_factored_state) - extractor.target_selector(factored_state))

        # parent and additional states are unnormalized
        states["parent_state"].append(extractor.parent_selector(factored_state))
        states["rel_state"].append(extractor.target_selector(factored_state) - extractor.parent_selector(factored_state))

        states["obs"].append(np.concatenate([norm(states["target"][-1]), norm(states["parent_state"][-1], form = "parent"), norm(states["rel_state"][-1], form = "rel")]))
        use_done = factored_state["Done"]
        states["done"].append(use_done)
        print(use_done, states["target_diff"][-1], states["target"][-1], states["next_target"][-1])
        factored_state = next_factored_state
    return ObjDict({sf: np.stack(states[sf], axis=0) for sf in state_forms})