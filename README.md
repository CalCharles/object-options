# Implementation of COInS

# Installation creating conda environment 
- conda create -n obj python=3.8
- conda activate obj
# Installing Robosuite with pushing domain (from source)
- git clone https://github.com/kvablack/robosuite.git
- conda activate obj
- cd robosuite
- copy mujoco download to: ~/.mujoco/mujoco200
- copy mujoco key to ~/.mujoco/mjkey.txt
- pip install -r requirements.txt
- https://robosuite.ai/docs/installation.html
# install remaining components
- conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
- conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
- pip install tianshou / pip install git+https://github.com/thu-ml/tianshou.git@master --upgrade
- conda install imageio
- pip install opencv-python
- conda install psutil
- pip install pyyaml

# installing for HYPE
- follow above install
- conda install -c anaconda scikit-learn
- pip -U install cma

# installing for cdl
- conda install pytorch=1.11=py3.8_cuda10.2_cudnn7.6.5_0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
- conda install -c conda-forge tensorboard
- conda install -c anaconda scikit-image
- conda install -c conda-forge matplotlib
- git clone https://github.com/kvablack/robosuite.git
- cd robosuite
- pip install -r requirements.txt
- conda install -c conda-forge gym
- pip install opencv-python
- conda install -c anaconda seaborn

# Run Random conditionals
- python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random/ --num-frames 1000
- python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random_conditional/ --num-frames 100000 --variant conditional
- python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random_conditional_passive/ --num-frames 100000 --variant conditional_passive
- python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random_conditional_rare/ --num-frames 200000 --variant conditional_rare
- python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random_conditional_common/ --num-frames 200000 --variant conditional_common
- python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random_cp_many/ --num-frames 100000 --variant cp_many
- python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random_cp_multi/ --num-frames 100000 --variant cp_multi
- python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random_cp_multi_small/ --num-frames 100000 --variant cp_multi_small
- python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random_passive_only/ --num-frames 1000 --variant passive_only
- python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/random_passive_only_noise/ --num-frames 1000 --variant passive_only_noise

# Run Random DAGs
-python generate_random.py --env RandomDAG --record-rollouts /hdd/datasets/object_data/RandomDAG/1_in/ --num-frames 1000 --variant 1-in
-python generate_random.py --env RandomDAG --record-rollouts /hdd/datasets/object_data/RandomDAG/3_chain/ --num-frames 1000 --variant 3-chain
-python generate_random.py --env RandomDAG --record-rollouts /hdd/datasets/object_data/RandomDAG/2_in/ --num-frames 1000 --variant 2-in
-python generate_random.py --env RandomDAG --record-rollouts /hdd/datasets/object_data/RandomDAG/multi_in/ --num-frames 1000 --variant multi-in
-python generate_random.py --env RandomDAG --record-rollouts /hdd/datasets/object_data/RandomDAG/3_in/ --num-frames 1000 --variant 3-in
-python generate_random.py --env RandomDAG --record-rollouts /hdd/datasets/object_data/RandomDAG/1_hdim/ --num-frames 1000 --variant 1-hdim
-python generate_random.py --env RandomDAG --record-rollouts /hdd/datasets/object_data/RandomDAG/1_rare/ --num-frames 1000 --variant 1-rare
-python generate_random.py --env RandomDAG --record-rollouts /hdd/datasets/object_data/RandomDAG/1_in_ndym/ --num-frames 1000 --variant 1-in-ndym
-python generate_random.py --env RandomDAG --record-rollouts /hdd/datasets/object_data/RandomDAG/1_in_nt/ --num-frames 1000 --variant 1-in-nt
-python generate_random.py --env RandomDAG --record-rollouts /hdd/datasets/object_data/RandomDAG/2_chain/ --num-frames 1000 --variant 2-chain

# create data for Breakout interaction
-python generate_random.py --env Breakout --record-rollouts /hdd/datasets/object_data/Breakout/rand_small --num-frames 1000000 --variant rand_small --policy RandAngle > logs/full/breakout/small_gen.txt
-python generate_random.py --env Breakout --record-rollouts /hdd/datasets/object_data/Breakout/rand_tiny --num-frames 1000000 --variant rand_tiny --policy RandAngle > logs/full/breakout/tiny_gen.txt
-python generate_random.py --env Pusher2D --num-frames 1000000 --variant small --record-rollouts /hdd/datasets/object_data/Pusher2D/rand_greedy_sticky --policy RandGreedySticky > logs/full/pusher2d/gen.txt
-python generate_random.py --env Pusher2D --num-frames 1000000 --variant tiny --record-rollouts /hdd/datasets/object_data/Pusher2D/rand_greedy_tiny --policy RandGreedy > logs/full/pusher2d/gen_tiny.txt


# multi random conditionals
python generate_random.py --env RandomDistribution --record-rollouts /hdd/datasets/object_data/RandomDistribution/multi_random/ --num-frames 1000 --variant multi_passive



# Run Breakout Training:
-python generate_random.py --env Breakout --record-rollouts /hdd/datasets/object_data/breakout/random/ --variant drop_stopping
-python main.py --main-train BreakoutStack
-python main.py --main-train BreakoutVariants

# Run Robopushing training:
-python generate_random.py --env RoboPushing --record-rollouts /hdd/datasets/object_data/robopushing/testrun/random --num-frames 5000
-python main.py --main-train RoboPushingStack
-python main.py --main-train RoboPushingObstacle

# Robopushing variants
-python generate_random.py --env RoboPushing --record-rollouts /hdd/datasets/object_data/robopushing/fixed/testrun/random/ --num-frames 5000 --fixed-limits
-python generate_random.py --env RoboPushing --record-rollouts /hdd/datasets/object_data/robopushing/discrete/random --variant discrete --num-frames 5000


# Run Full Breakout Training:
python generate_random.py --env Breakout --record-rollouts /hdd/datasets/object_data/full/breakout/small/random/ --num-frames 50000


# Run Asteroids training:
python generate_random.py --env Asteroids --record-rollouts /hdd/datasets/object_data/asteroids/random/ --num-frames 10000 --fixed-limits

python generate_random.py --env Asteroids --record-rollouts /hdd/datasets/object_data/asteroids/coordinate_turn/random/ --variant coordinate_turn --num-frames 10000 --fixed-limits


# Run Sokoban training:
python generate_random.py --env Sokoban --record-rollouts /hdd/datasets/object_data/sokoban/random/ --num-frames 10000

python generate_random.py --env Sokoban --record-rollouts /hdd/datasets/object_data/sokoban/fixed/few_obs/random/ --variant few_obs --num-frames 10000 --fixed-limits

python generate_random.py --env Asteroids --demonstrate --num-frames 5000

# Run Taxicar Training:
python generate_random.py --env TaxiCar --record-rollouts /hdd/datasets/object_data/TaxiCar/random/ --num-frames 10000

# Run airhockey training:
python generate_random.py --env AirHockey --record-rollouts /hdd/datasets/object_data/airhockey/ --demonstrate --num-frames 5000


<!---
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
-->


