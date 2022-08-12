import numpy as np

def sincos_to_angle(sinv, cosv):
	vangle = np.arctan(sinv / cosv)
	if cosv <= 0:
	    if vangle < 0:
	        vangle = vangle - np.pi
	    else:
	        vangle = vangle + np.pi
	if vangle < 0: angle = 2 * np.pi + vangle
	else: angle = vangle
	return angle

def sincos_to_angle2(sinv, cosv):
	vangle = np.arctan(sinv / cosv)
	if cosv <= 0:
	    if vangle < 0:
	        vangle = vangle + np.pi
	    else:
	        vangle = vangle - np.pi
	if vangle < 0: angle = 2 * np.pi + vangle
	else: angle = vangle
	return angle
