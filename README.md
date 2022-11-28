Implementation of HOOD

# Installation creating conda environment 
conda create -n obj python=3.8
conda activate obj
# Installing Robosuite with pushing domain (from source)
git clone https://github.com/kvablack/robosuite.git
conda activate obj
cd robosuite
copy mujoco download to: ~/.mujoco/mujoco200
copy mujoco key to ~/.mujoco/mjkey.txt
pip install -r requirements.txt
https://robosuite.ai/docs/installation.html
# install remaining components
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install tianshou / pip install git+https://github.com/thu-ml/tianshou.git@master --upgrade
conda install imageio
pip install opencv-python
conda install psutil
pip install pyyaml


Breakout: get perfect paddle policy
get perfect ball bouncing policy with inline training

Robopushing: retrain interaction
get gripper policy
get block policy

Sokoban: get block policy

Asteroids: get ship policy

CT asteroids: get ship policy

Run Random conditionals
python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random/ --num-frames 1000
python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random_conditional/ --num-frames 1000 --variant conditional
python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random_conditional_passive/ --num-frames 1000 --variant conditional_passive
python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random_passive_only/ --num-frames 1000 --variant passive_only
python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random_passive_only_noise/ --num-frames 1000 --variant passive_only_noise


Run Breakout Training:
python generate_random.py --env Breakout --record-rollouts /hdd/datasets/object_data/breakout/random/

Run Full Breakout Training:
python generate_random.py --env Breakout --record-rollouts /hdd/datasets/object_data/full/breakout/small/random/ --num-frames 50000

Run Robopushing training:
python generate_random.py --env RoboPushing --record-rollouts /hdd/datasets/object_data/robopushing/ --num-frames 5000
python generate_random.py --env RoboPushing --record-rollouts ../data/object_data/robopushing/random/ --num-frames 5000


Run Asteroids training:
python generate_random.py --env Asteroids --record-rollouts /hdd/datasets/object_data/asteroids/random/ --num-frames 10000

python generate_random.py --env Asteroids --record-rollouts /hdd/datasets/object_data/asteroids/coordinate_turn/random/ --variant coordinate_turn --num-frames 10000


Run Sokoban training:
python generate_random.py --env Sokoban --record-rollouts /hdd/datasets/object_data/Sokoban/random/ --num-frames 10000

python generate_random.py --env Sokoban --record-rollouts /hdd/datasets/object_data/Sokoban/random/ --variant few_obs --num-frames 10000

python generate_random.py --env Asteroids --demonstrate --num-frames 5000

Run Taxicar Training:
python generate_random.py --env TaxiCar --record-rollouts /hdd/datasets/object_data/TaxiCar/random/ --num-frames 10000



todos:
Asteroids: change action space to choose angle instead of sin-cos space
Existence hindsight
Make alignment angles 
buffer checks: check laser firing hindsight
Sampling checks: reachable angle
	reachable location
	laser starts at ship
interaction checks: on fire and nowhere else
Sokoban: Unit test obstacle avoidance 
Stuck resets
Displacement hindsight
Sampling checks: pusher sample reachable --DONE
	block sample reachable
interaction checks: on block push and nowhere else
Buffer checks: checks block moving hindsight and trajectories after random
Bfs range 
Both: Train iterations immediately after random --DONE
	One trajectory unit test --DONE
	Action remapping (round) --DONE
	Network resets --DONE
	Ground truth movement for Asteroids and Sokoban to train second level policies --Written, requires testing
	Human option controller --DONE
	hindsight random resampling for parameter --WRITTEN
	Hyperparameter tuning programmatic: gamma, lr, sampling, termination rewards --DONE
	Network parameter tuning: depth, width, activations --DONE
	Hyperparameters: lookahead, network_resets --DONE
	Binaries instead of interaction model -- DONE
	Hyperparameters: hindsight local reset
	dummy interaction models for laser and block
	make components optional at first
	Random network distillation
	Value stochastic policy
	Hyperparametersrandom network distillation, hindsight random resampling, rounded actions, network resets, sac/ddpg, 
	Policy smoothness criteria: penalize max action change relative to position
	truncation on dones, even time cutoff
	Penalize no-movement
	Action entropy reward
	Learned Sampling for temporal proximity


