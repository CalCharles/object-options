import numpy as np

class Object:
    def __init__(self):
        self.state = None
        self.interaction_trace = list()

class Action(Object):
    def __init__(self):
    	super().__init__()
    	self.name = "Action"
    	self.action = 0
    	self.interaction_trace = list()

    def step(self, action):
    	self.action = action

    def get_state(self):
    	return np.array([self.action])

class Bound(Object):
	def __init__(self, limits, n_dim=2):
		super().__init__()
		self.name = "Bound"
		self.limits = limits

	def get_state(self):
		return np.array(self.limits)

class Pusher(Object):
	def __init__(self, pos, bound):
		super().__init__()
		self.name = "Pusher"
		self.pos = pos
		self.pos_change = np.zeros(pos.shape)
		self.bound = bound
		self.movement_type = "coordinate" # could have a different movement type

	def out_of_bounds(self, pos):
		return 0 > pos[0] or pos[0] >= self.bound.limits[0] or 0 > pos[1] or pos[1] >= self.bound.limits[1]  

	def step(self, action, occupancy_matrix):
		self.pos_change = np.zeros(self.pos.shape)
		action.interaction_trace = [self]
		self.interaction_trace = [action]
		if self.movement_type == "coordinate":
			if action.action == 0: self.pos_change[0] -= 1
			if action.action == 1: self.pos_change[0] += 1
			if action.action == 2: self.pos_change[1] -= 1
			if action.action == 3: self.pos_change[1] += 1
			new_pos = self.pos + self.pos_change
			if self.out_of_bounds(new_pos): return # there will be no changes if moving out of bounds in update
			obj_at = occupancy_matrix[int(new_pos[0])][int(new_pos[1])]
			if obj_at is not None:
				if type(obj_at) == Block:
					if not obj_at.moveable(self.pos_change, occupancy_matrix):
						self.interaction_trace += [obj_at]
						self.pos_change = np.zeros(self.pos.shape)
				if type(obj_at) == tuple:
					obj_at = obj_at[0]
					if not obj_at.moveable(self.pos_change, occupancy_matrix):
						self.interaction_trace += [obj_at]
						self.pos_change = np.zeros(self.pos.shape)
				elif type(obj_at) == Obstacle:
					self.interaction_trace += [obj_at]
					self.pos_change = np.zeros(self.pos.shape)

	def update(self): # TODO: bounds don't have trace
		old_pos = self.pos.copy()
		self.pos[0] = np.clip(self.pos[0] + self.pos_change[0], 0, self.bound.limits[0]-1)
		self.pos[1] = np.clip(self.pos[1] + self.pos_change[1], 0, self.bound.limits[1]-1)
		return old_pos

	def get_state(self):
		return np.array(self.pos.tolist())



class Obstacle(Object):
	def __init__(self, pos, idx, bound):
		super().__init__()
		self.pos = pos
		self.name = "Obstacle" + str(idx)
	
	def get_state(self):
		return np.array(self.pos.tolist())

class Block(Object):
	def __init__(self, pos, idx, bound):
		super().__init__()
		self.pos = pos
		self.pushed = False
		self.name = "Block" + str(idx)
		self.bound = bound

	def moveable(self, change, occupancy_matrix):
		self.pushed = True
		new_pos = self.pos + change
		# print("new pos", new_pos)
		if 0 > new_pos[0] or new_pos[0] >= self.bound.limits[0] or 0 > new_pos[1] or new_pos[1] >= self.bound.limits[1]:
			return False
		obj_at = occupancy_matrix[int(new_pos[0])][int(new_pos[1])]
		if obj_at is not None:
			if type(obj_at) != Target:
				return False
		return True 

	def step(self, pusher, occupancy_matrix):
		self.pos_change = self.pos
		self.interaction_trace = list()
		if self.pushed:
			self.interaction_trace.append(pusher)
			pusher.interaction_trace.append(self)
			change = pusher.pos_change
			if self.moveable(change, occupancy_matrix):
				self.pos_change = self.pos + change.copy()

	def update(self):
		self.pushed = False
		old = self.pos.copy()
		self.pos = self.pos_change
		moved = None
		# print(np.linalg.norm(old - self.pos), self.pos.copy())
		if np.linalg.norm(old - self.pos) > 0.01:
			moved = self
		return old, moved

	def get_state(self):
		return np.array(self.pos.tolist())

class Target(Object):
	def __init__(self, pos, idx, bound):
		super().__init__()
		self.pos = pos
		self.attribute = 0
		self.attribute_change = 0
		self.name = "Target" + str(idx)

	def step(self, occupancy_matrix):
		obj_at = occupancy_matrix[self.pos[0]][self.pos[1]]
		if type(obj_at) == tuple: 
			if type(obj_at[0]) == Block:
				self.interaction_trace.append(obj_at[0])
				self.attribute_change = 1
		else:
			self.attribute_change = 0

	def update(self): 
		self.attribute = self.attribute_change

	def get_state(self):
		return np.array(self.pos.tolist() + [self.attribute])
