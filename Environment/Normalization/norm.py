def hardcode_norm(env_name, obj_names):
	if env_name == "SelfBreakout":
		norm_dict = breakout_norms
	elif env_name == "RoboPushing":
		norm_dict = robo_norms
	elif env_name == "RoboStick":
		norm_dict = stick_norms
	# ONLY IMPLEMENTED FOR Breakout, Robosuite Pushing
	norm_mean = list()
	norm_var = list()
	norm_inv_var = list()
	for n in obj_names:
		norm_mean.append(norm_dict[n][0])
		norm_var.append(norm_dict[n][1])
		norm_inv_var.append(1.0/norm_dict[n][1])
	return np.concatenate(norm_mean, axis=0), np.concatenate(norm_var, axis=0), np.concatenate(norm_inv_var, axis=0)

def position_mask(env_name):
	if env_name == "SelfBreakout":
		return np.array([1,1,0,0,0]).astype(float)
	if env_name == "RoboPushing":
		return np.array([1,1,1]).astype(float)