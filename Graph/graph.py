import os
from State.object_dict import ObjDict
from Causal.interaction_model import load_interaction
from Option.option import load_option
from Record.file_management import save_to_pickle, load_from_pickle

class Node:
	def __init__(self, name): # add the remaining attributes as constructed
		self.name = name
		self.interaction = None # the active-passive-interaction model
		self.option = None # the option to control the node

	def toString(self):
		return self.name

class Graph:
	def __init__(self, objects, action_interaction, action_option):
		self.nodes = ObjDict({name: Node(name) for name in objects})
		self.nodes.Action.interaction = action_interaction
		self.nodes.Action.option = action_option
		self.chain = ["Action"] # TODO : only supports chain structures

	def save(self, save_dir):
		inter_option = dict()
		for node in self.nodes.values():
			inter_option[node.name] = (node.interaction, node.option, node.option.next_option if node.option is not None else None)
			if node.interaction is not None:
				node.interaction.save(save_dir)
			if node.option is not None:
				node.option.save(save_dir)
		for node in self.nodes.values():
			if node.option is not None: node.option.next_option = None # to prevent double-saving, we remove the next_option
			node.option = None
			node.interaction = None
		save_to_pickle(os.path.join(save_dir, "graph.gm"), self)
		for node in self.nodes.values():
			node.interaction, node.option, next_option = inter_option[node.name]
			if node.option is not None: node.option.next_option = next_option

	def add_to_chain(self, name):
		self.chain.append(name)

def load_graph(load_dir, device):
	load_name = os.path.join(load_dir, "graph.gm")
	graph = load_from_pickle(load_name)
	for node in graph.nodes.values():
		node.interaction = load_interaction(load_dir, node.name, device) # returns none if there is no interaction model in the directory
		node.option = load_option(load_dir, node.name, device) # returns none if there is no option in the directory
		# reassigns to the latest interaction model, if changed
		if node.option is not None:
			node.option.sampler.mask = node.interaction.mask
			node.option.terminate_reward.interaction_model = node.interaction
	# to prevent double-saving, we removed the next_option, and add it back here
	for node in graph.nodes.values():
		if node.option is not None and node.option.name != "Action":
			node.option.next_option = graph.nodes[node.option.next_option_name].option
	return graph