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

def same_pos(p1, p2):
	return p1[0] == p2[0] and p1[1] == p2[1]

class Taxi(Object):
	def __init__(self, pos, bound, ped_penalty=-1, crash_penalty=-1, dropoff_reward=1):
		super().__init__()
		self.name = "Taxi"
		self.pos = pos
		self.pos_change = np.zeros(pos.shape)
		self.bound = bound
		self.movement_type = "coordinate" # could have a different movement type
		self.pickup_dropoff = False
		self.passenger = None
		self.crash_penalty = crash_penalty
		self.dropoff_reward = dropoff_reward

	def out_of_bounds(self, pos):
		return 0 > pos[0] or pos[0] >= self.bound[0] or 0 > pos[1] or pos[1] >= self.bound[1]  

	def pre_step(self, action):
		self.pos_change = np.zeros(self.pos.shape)
		self.pickup_dropoff = False
		self.interaction_trace = [action]
		if self.movement_type == "coordinate":
			if action.action == 0: self.pos_change[0] -= 1
			if action.action == 1: self.pos_change[0] += 1
			if action.action == 2: self.pos_change[1] -= 1
			if action.action == 3: self.pos_change[1] += 1
			if action.action == 4: self.pickup_dropoff = True


	def step(self, vehicles, passengers, reward):
		held_passenger = self.passenger
		if self.pickup_dropoff:
			if self.passenger is not None and not self.passenger.arrived:
				self.passenger.riding = False
				self.passenger.just_dropped = True
				self.passenger.interaction_trace.append(self)
				self.passenger = None

		for passenger in passengers:
			if self.pickup_dropoff:
				if same_pos(self.pos, passenger.pos) and (held_passenger is None or held_passenger.name != passenger.name) and (not passenger.at_target):
					if held_passenger is not None and held_passenger.name != passenger.name:
						held_passenger.interaction_trace += [self]
					self.passenger = passenger
					self.passenger.riding = True
					passenger.interaction_trace += [self]

		for vehicle in vehicles:
			if same_pos(self.pos + self.pos_change, vehicle.pos) or same_pos(self.pos + self.pos_change, vehicle.pos + vehicle.vel):
				self.pos_change = 0
				reward.attribute += crash_penalty
				self.interaction_trace += [vehicle]

	def update(self): # TODO: bounds don't have trace
		old_pos = self.pos.copy()
		self.pos[0] = np.clip(self.pos[0] + self.pos_change[0], 0, self.bound.limits[0]-1)
		self.pos[1] = np.clip(self.pos[1] + self.pos_change[1], 0, self.bound.limits[1]-1)
		return old_pos

	def get_state(self):
		return np.array(self.pos.tolist() + [int(self.passenger is not None)] + self.bound.tolist())



class Pedestrian(Object):
	def __init__(self, pos, idx, bound, vel):
		super().__init__()
		self.pos = pos
		self.name = "Pedestrian" + str(idx) if idx >= 0 else "Pedestrian"
		self.vel = vel
		self.pos_change = np.zeros(self.pos.shape)
		self.hit = 0
		self.bound = bound

	def step(self, vehicles, taxi, reward):
		self.pos_change = self.vel
		self.hit = 0
		for vehicle in vehicles:
			if same_pos(self.pos + self.pos_change, vehicle.pos) or same_pos(self.pos + self.pos_change, vehicle.pos + vehicle.vel):
				self.pos_change = np.zeros(self.pos.shape)
				self.interaction_trace.append(vehicle)

		if same_pos(taxi.pos + taxi.pos_change, self.pos) or same_pos(taxi.pos + taxi.pos_change, self.pos + self.vel):
			self.hit = 1
			reward.attribute += taxi.ped_penalty
			reward.interaction_trace.append(taxi)
			reward.interaction_trace.append(self)
			self.interaction_trace.append(taxi)

	def update(self): # TODO: bounds don't have trace
		old_pos = self.pos.copy()
		self.pos[0] = np.clip(self.pos[0] + self.pos_change[0], 0, self.bound.limits[0]-1)
		self.pos[1] = np.clip(self.pos[1] + self.pos_change[1], 0, self.bound.limits[1]-1)
		if self.pos[0] + self.pos_change[0] > self.bound.limits[0] or self.pos[0] + self.pos_change[0] < 0 or self.pos[1] + self.pos_change[1] > self.bound.limits[1] or self.pos[1] + self.pos_change[1] < 0:
			self.vel = - self.vel
		return old_pos		

	def get_state(self):
		return np.array(self.pos.tolist() + self.vel.tolist() + [self.hit] + self.bound.pos.tolist())

class Vehicle(Object)
	def __init__(self, pos, idx, bound, vel):
		super().__init__()
		self.pos = pos
		self.name = "Vehicle" + str(idx) if idx >= 0 else "Vehicle"
		self.vel = vel
		self.pos_change = np.zeros(self.pos.shape)
		self.hit = 0
		self.bound = bound

	def step(self, vehicles, taxi, reward):
		self.pos_change = self.vel

	def update(self): # same as pedestrian update rule
		old_pos = self.pos.copy()
		self.pos[0] = np.clip(self.pos[0] + self.pos_change[0], 0, self.bound.limits[0]-1)
		self.pos[1] = np.clip(self.pos[1] + self.pos_change[1], 0, self.bound.limits[1]-1)
		if self.pos[0] + self.pos_change[0] > self.bound.limits[0] or self.pos[0] + self.pos_change[0] < 0 or self.pos[1] + self.pos_change[1] > self.bound.limits[1] or self.pos[1] + self.pos_change[1] < 0:
			self.vel = - self.vel
		return old_pos

	def get_state(self):
		return np.array(self.pos.tolist() + self.vel.tolist() + self.bound.pos.tolist())

class Passenger(Object):
	def __init__(self, pos, idx, bound, target):
		super().__init__()
		self.pos = pos
		self.name = "Passenger" + str(idx) if idx >= 0 else "Passenger"
		self.pos_change = np.zeros(self.pos.shape)
		self.riding = False
		self.just_dropped = False
		self.target = target
		self.arrived = False
		self.at_target = False

	def step(self, taxi, reward, targets):
		self.pos_change = np.zeros(self.pos.shape)
		if self.riding:
			self.pos_change = taxi.pos_change
			if taxi not in self.interaction_trace:
				self.interaction_trace.append(taxi)

		if same_pos(self.pos, self.target) and self.just_dropped:
			self.arrived = True
			self.at_target = True
			self.target.attribute_change = 1
			self.target.interaction_trace.append(self)
			reward.attribute += taxi.dropoff_reward
	
	def update(self): # same as pedestrian update rule
		old_pos = self.pos.copy()
		self.pos[0] = np.clip(self.pos[0] + self.pos_change[0], 0, self.bound.limits[0]-1)
		self.pos[1] = np.clip(self.pos[1] + self.pos_change[1], 0, self.bound.limits[1]-1)
		if self.pos[0] + self.pos_change[0] > self.bound.limits[0] or self.pos[0] + self.pos_change[0] < 0 or self.pos[1] + self.pos_change[1] > self.bound.limits[1] or self.pos[1] + self.pos_change[1] < 0:
			self.vel = - self.vel
		return old_pos

	def get_state(self):
		return np.array(self.pos.tolist() + [float(self.arrived)] + [float(self.riding)])

class Target(Object):
	def __init__(self, pos, idx, bound):
		super().__init__()
		self.pos = pos
		self.attribute = 0
		self.attribute_change = 0
		self.name = "Target" + str(idx) if idx >= 0 else "Target"

	def step(self): # must step BEFORE passenger
		self.attribute_change = 0

	def update(self):
		self.attribute = self.attribute_change

	def get_state(self):
		return np.array(self.pos.tolist() + [self.attribute])
