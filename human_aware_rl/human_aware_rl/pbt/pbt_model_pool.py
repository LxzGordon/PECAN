import os, gym, time, sys, random, itertools, math
import torch as th

from turtle import shape
from re import S
import numpy as np
import tensorflow as tf
from collections import defaultdict
from memory_profiler import profile
from tensorflow.saved_model import simple_save

from sacred import Experiment
from sacred.observers import FileStorageObserver

PBT_DATA_DIR = os.environ['PBT_DATA_DIR']
ex = Experiment('PBT')
from human_aware_rl.context.model import Net

from overcooked_ai_py.utils import profile, load_pickle, save_pickle, save_dict_to_file, load_dict_from_file
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair_context,AgentPair_eval

from human_aware_rl.utils import create_dir_if_not_exists, delete_dir_if_exists, reset_tf, set_global_seed
from human_aware_rl.baselines_utils import create_model,create_model_context, get_vectorized_gym_env, update_model,update_model_eval,update_model_context, get_agent_from_model,get_agent_from_model_context, save_baselines_model, overwrite_model, load_baselines_model, LinearAnnealer
from scipy.stats import rankdata

class PBTAgent_context(object):
    """An agent that can be saved and loaded and all and the main data it contains is the self.model
    
    Goal is to be able to pass in save_locations or PBTAgents to workers that will load such agents
    and train them together.
    """

    def __init__(self, agent_name, start_params, start_logs=None, model=None, gym_env=None):
        self.params = start_params
        self.context=np.zeros(4)
        self.logs = start_logs if start_logs is not None else {
            "agent_name": agent_name,
            "avg_rew_per_step": [],
            "avg_rew_per_step_2": [],
            "params_hist": defaultdict(list),
            "num_ppo_runs": 0,
            "reward_shaping": []
        }
        with tf.device('/device:GPU:{}'.format(self.params["GPU_ID"])):
            self.model = model if model is not None else create_model_context(gym_env, agent_name, **start_params)

    @property
    def num_ppo_runs(self):
        return self.logs["num_ppo_runs"]
    
    @property
    def agent_name(self):
        return self.logs["agent_name"]

    def get_agent(self,eval=False):
        return get_agent_from_model_context(self.model,self.context, self.params["sim_threads"])

    def update(self, gym_env, metric_np=None,c=None,ens_population=None,w=None,c_identifier=None):
        with tf.device('/device:GPU:{}'.format(self.params["GPU_ID"])):
            if ens_population is None:
                train_info = update_model_context(gym_env, self.model, metric_np=metric_np,c=c,c_identifier=c_identifier, **self.params)
            else:
                train_info = update_model(gym_env, self.model,ens_population=ens_population,w=w,c=c,ent_version=3,c_identifier=c_identifier, **self.params)
            for k, v in train_info.items():
                if k not in self.logs.keys():
                    self.logs[k] = []
                self.logs[k].extend(v)
            self.logs["num_ppo_runs"] += 1

    def update_avg_rew_per_step_logs(self, avg_rew_per_step_stats):
        self.logs["avg_rew_per_step"] = avg_rew_per_step_stats

    def update_avg_rew_per_step_logs_2(self, avg_rew_per_step_stats):
        self.logs["avg_rew_per_step_2"] = avg_rew_per_step_stats

    def save(self, save_folder):
        """Save agent model, logs, and parameters"""
        create_dir_if_not_exists(save_folder)
        save_baselines_model(self.model, save_folder)
        save_dict_to_file(dict(self.logs), save_folder + "logs")
        save_dict_to_file(self.params, save_folder + "params")

    @staticmethod
    def from_dir(load_folder, agent_name):
        logs = load_dict_from_file(load_folder + "logs.txt")
        params = load_dict_from_file(load_folder + "params.txt")
        model = load_baselines_model(load_folder[0:-1], agent_name, params)
        return PBTAgent(agent_name, params, start_logs=logs, model=model)

    @staticmethod
    def update_from_files(file0, file1, gym_env, save_dir):
        pbt_agent0 = PBTAgent.from_dir(file0)
        pbt_agent1 = PBTAgent.from_dir(file1)
        gym_env.other_agent = pbt_agent1
        pbt_agent0.update(gym_env)
        return pbt_agent0

    def save_predictor(self, save_folder):
        """Saves easy-to-load simple_save tensorflow predictor for agent"""
        simple_save(
            tf.get_default_session(),
            save_folder,
            inputs={"obs": self.model.act_model.X,'c': self.model.act_model.context},
            outputs={
                "action": self.model.act_model.action, 
                "value": self.model.act_model.vf,
                "action_probs": self.model.act_model.action_probs
            }
        )

    def update_pbt_iter_logs(self):
        for k, v in self.params.items():
            self.logs["params_hist"][k].append(v)
        self.logs["params_hist"] = dict(self.logs["params_hist"])

    def explore_from(self, best_training_agent):
        overwrite_model(best_training_agent.model, self.model)
        self.logs["num_ppo_runs"] = best_training_agent.num_ppo_runs
        self.params = best_training_agent.params.copy()

class PBTAgent(object):
    """An agent that can be saved and loaded and all and the main data it contains is the self.model
    
    Goal is to be able to pass in save_locations or PBTAgents to workers that will load such agents
    and train them together.
    """
    
    def __init__(self, agent_name, start_params, start_logs=None, model=None, gym_env=None):
        self.params = start_params
        self.context=np.zeros(4)
        self.logs = start_logs if start_logs is not None else {
            "agent_name": agent_name,
            "avg_rew_per_step": [],
            "avg_rew_per_step_2": [],
            "params_hist": defaultdict(list),
            "num_ppo_runs": 0,
            "reward_shaping": []
        }
        with tf.device('/device:GPU:{}'.format(self.params["GPU_ID"])):
            self.model = model if model is not None else create_model(gym_env, agent_name, **start_params)

    @property
    def num_ppo_runs(self):
        return self.logs["num_ppo_runs"]
    
    @property
    def agent_name(self):
        return self.logs["agent_name"]

    def get_agent(self,eval=False):
        return get_agent_from_model(self.model,self.context, self.params["sim_threads"],eval=eval)

    def update(self, gym_env,ens_population,w, metric_np=None):
        with tf.device('/device:GPU:{}'.format(self.params["GPU_ID"])):
            train_info = update_model(gym_env, self.model,ens_population=ens_population,w=w,ent_version=3, **self.params)

            for k, v in train_info.items():
                if k not in self.logs.keys():
                    self.logs[k] = []
                self.logs[k].extend(v)
            self.logs["num_ppo_runs"] += 1
    def update_eval(self, gym_env, metric_np=None):
        with tf.device('/device:GPU:{}'.format(self.params["GPU_ID"])):
            train_info = update_model_eval(gym_env, self.model, metric_np=metric_np, **self.params)

            for k, v in train_info.items():
                if k not in self.logs.keys():
                    self.logs[k] = []
                self.logs[k].extend(v)
            self.logs["num_ppo_runs"] += 1

    def update_avg_rew_per_step_logs(self, avg_rew_per_step_stats):
        self.logs["avg_rew_per_step"] = avg_rew_per_step_stats

    def update_avg_rew_per_step_logs_2(self, avg_rew_per_step_stats):
        self.logs["avg_rew_per_step_2"] = avg_rew_per_step_stats

    def save(self, save_folder):
        """Save agent model, logs, and parameters"""
        create_dir_if_not_exists(save_folder)
        save_baselines_model(self.model, save_folder)
        save_dict_to_file(dict(self.logs), save_folder + "logs")
        save_dict_to_file(self.params, save_folder + "params")

    @staticmethod
    def from_dir(load_folder, agent_name):
        logs = load_dict_from_file(load_folder + "logs.txt")
        params = load_dict_from_file(load_folder + "params.txt")
        model = load_baselines_model(load_folder[0:-1], agent_name, params)
        return PBTAgent(agent_name, params, start_logs=logs, model=model)

    @staticmethod
    def update_from_files(file0, file1, gym_env, save_dir):
        pbt_agent0 = PBTAgent.from_dir(file0)
        pbt_agent1 = PBTAgent.from_dir(file1)
        gym_env.other_agent = pbt_agent1
        pbt_agent0.update(gym_env)
        return pbt_agent0

    def save_predictor(self, save_folder):
        """Saves easy-to-load simple_save tensorflow predictor for agent"""
        simple_save(
            tf.get_default_session(),
            save_folder,
            inputs={"obs": self.model.act_model.X},
            outputs={
                "action": self.model.act_model.action, 
                "value": self.model.act_model.vf,
                "action_probs": self.model.act_model.action_probs
            }
        )

    def update_pbt_iter_logs(self):
        for k, v in self.params.items():
            self.logs["params_hist"][k].append(v)
        self.logs["params_hist"] = dict(self.logs["params_hist"])

    def explore_from(self, best_training_agent):
        overwrite_model(best_training_agent.model, self.model)
        self.logs["num_ppo_runs"] = best_training_agent.num_ppo_runs
        self.params = best_training_agent.params.copy()

@ex.config
def my_config():

    ##################
    # GENERAL PARAMS #
    ##################

    TIMESTAMP_DIR = True
    EX_NAME = "undefined_name"

    if TIMESTAMP_DIR:
        SAVE_DIR = PBT_DATA_DIR + time.strftime('%Y_%m_%d-%H_%M_%S_') + EX_NAME + "/"
    else:
        SAVE_DIR = PBT_DATA_DIR + EX_NAME + "/"

    print("Saving data to ", SAVE_DIR)

    RUN_TYPE = "pbt"

    # Reduce parameters to be able to run locally to test for simple bugs
    LOCAL_TESTING = False

    # GPU id to use
    GPU_ID = 1

    # List of seeds to run
    SEEDS = [0]

    # Number of parallel environments used for simulating rollouts
    sim_threads = 50 if not LOCAL_TESTING else 2

    ##############
    # PBT PARAMS #
    ##############

    #TOTAL_STEPS_PER_AGENT = 1.5e7 if not LOCAL_TESTING else 1e4
    TOTAL_STEPS_PER_AGENT = 3e7 if not LOCAL_TESTING else 1e4
    POPULATION_SIZE = 4

    ITER_PER_SELECTION = POPULATION_SIZE # How many pairings and model training updates before the worst model is overwritten

    RESAMPLE_PROB = 0.33
    MUTATION_FACTORS = [0.75, 1.25]
    HYPERPARAMS_TO_MUTATE = ["LAM", "CLIPPING", "LR", "STEPS_PER_UPDATE", "ENTROPY", "VF_COEF"]

    NUM_SELECTION_GAMES = 10 if not LOCAL_TESTING else 2

    ##############
    # PPO PARAMS #
    ##############
    TRAINING_ITERATIONS=100
    # Decay from population partners to policy-ensemble partners
    ALPHA_DECAY_HORIZON=70
    ALPHA_FINAL=0.3

    # Total environment timesteps for the PPO run
    PPO_RUN_TOT_TIMESTEPS = 40000 if not LOCAL_TESTING else 1000
    NUM_PBT_ITER = int(TOTAL_STEPS_PER_AGENT * math.sqrt(POPULATION_SIZE) // (ITER_PER_SELECTION * PPO_RUN_TOT_TIMESTEPS))

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    TOTAL_BATCH_SIZE = 20000 if not LOCAL_TESTING else 1000

    # Number of minibatches we divide up each batch into before
    # performing gradient steps
    MINIBATCHES = 5 if not LOCAL_TESTING else 1

    BATCH_SIZE = TOTAL_BATCH_SIZE // sim_threads

    # Number of gradient steps to perform on each mini-batch
    STEPS_PER_UPDATE = 8 if not LOCAL_TESTING else 1

    # Learning rate
    LR = 5e-3

    # Entropy bonus coefficient
    ENTROPY = 0.5

    # Entropy bonus coefficient for the model pool
    ENTROPY_POOL = 0.0

    # Epsilon for calculating the sampling probability
    EPSILON = 1e-6

    # Use prioritized sampling or not
    PRIORITIZED_SAMPLING = False

    # Alpha for prioritized sampling
    ALPHA = 1.0

    # Metric for prioritized sampling
    METRIC = 1.0

    # Paths of the member agents in the model pool
    LOAD_FOLDER_LST = ''

    # Value function coefficient
    VF_COEF = 0.1

    # Gamma discounting factor
    GAMMA = 0.99

    # Lambda advantage discounting factor
    LAM = 0.98

    # Max gradient norm
    MAX_GRAD_NORM = 0.1

    # PPO clipping factor
    CLIPPING = 0.05

    # 0 is default value that does no annealing
    REW_SHAPING_HORIZON = 0

    ##################
    # NETWORK PARAMS #
    ##################

    # Network type used
    NETWORK_TYPE = "conv_and_mlp"

    # Network params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3


    ##################
    # MDP/ENV PARAMS #
    ##################

    # Mdp params
    layout_name = None
    start_order_list = None

    rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0.015,
        "POT_DISTANCE_REW": 0.03,
        "SOUP_DISTANCE_REW": 0.1,
    }
    
    # Env params
    horizon = 400


    #########
    # OTHER #
    #########

    # For non fixed MDPs
    mdp_generation_params = {
        "padded_mdp_shape": (11, 7),
        "mdp_shape_fn": ([5, 11], [5, 7]),
        "prop_empty_fn": [0.6, 1],
        "prop_feats_fn": [0, 0.6]
    }

    # Approximate info stats
    GRAD_UPDATES_PER_AGENT = STEPS_PER_UPDATE * MINIBATCHES * (PPO_RUN_TOT_TIMESTEPS // TOTAL_BATCH_SIZE) * ITER_PER_SELECTION * NUM_PBT_ITER // POPULATION_SIZE

    print("Total steps per agent", TOTAL_STEPS_PER_AGENT)
    print("Grad updates per agent", GRAD_UPDATES_PER_AGENT)

    params = {
        "LOCAL_TESTING": LOCAL_TESTING,
        "RUN_TYPE": RUN_TYPE,
        "EX_NAME": EX_NAME,
        "SAVE_DIR": SAVE_DIR,
        "GPU_ID": GPU_ID,
        "mdp_params": {
            "layout_name": layout_name,
            "start_order_list": start_order_list,
            "rew_shaping_params": rew_shaping_params
        },
        "env_params": {
            "horizon": horizon
        },
        'TRAINING_ITERATIONS':TRAINING_ITERATIONS,
        'ALPHA_DECAY_HORIZON':ALPHA_DECAY_HORIZON,
        'ALPHA_FINAL':ALPHA_FINAL,
        "PPO_RUN_TOT_TIMESTEPS": PPO_RUN_TOT_TIMESTEPS,
        "NUM_PBT_ITER": NUM_PBT_ITER,
        "ITER_PER_SELECTION": ITER_PER_SELECTION,
        "POPULATION_SIZE": POPULATION_SIZE,
        "RESAMPLE_PROB": RESAMPLE_PROB,
        "MUTATION_FACTORS": MUTATION_FACTORS,
        "mdp_generation_params": mdp_generation_params, # NOTE: currently not used
        "HYPERPARAMS_TO_MUTATE": HYPERPARAMS_TO_MUTATE,
        "REW_SHAPING_HORIZON": REW_SHAPING_HORIZON,
        "ENTROPY": ENTROPY,
        "ENTROPY_POOL": ENTROPY_POOL,
        "EPSILON": EPSILON,
        "PRIORITIZED_SAMPLING": PRIORITIZED_SAMPLING,
        "ALPHA": ALPHA,
        "METRIC": METRIC,
        "LOAD_FOLDER_LST": LOAD_FOLDER_LST.split(':'),
        "GAMMA": GAMMA,
        "sim_threads": sim_threads,
        "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "LR": LR,
        "VF_COEF": VF_COEF,
        "STEPS_PER_UPDATE": STEPS_PER_UPDATE,
        "MINIBATCHES": MINIBATCHES,
        "CLIPPING": CLIPPING,
        "LAM": LAM,
        "NETWORK_TYPE": NETWORK_TYPE,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS": SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS": NUM_FILTERS,
        "NUM_CONV_LAYERS": NUM_CONV_LAYERS,
        "SEEDS": SEEDS,
        "NUM_SELECTION_GAMES": NUM_SELECTION_GAMES,
        "total_steps_per_agent": TOTAL_STEPS_PER_AGENT,
        "grad_updates_per_agent": GRAD_UPDATES_PER_AGENT
    }

@ex.named_config
def fixed_mdp():
    LOCAL_TESTING = False
    # fixed_mdp = True
    layout_name = "simple"

    sim_threads = 30 if not LOCAL_TESTING else 2
    PPO_RUN_TOT_TIMESTEPS = 36000 if not LOCAL_TESTING else 1000
    TOTAL_BATCH_SIZE = 12000 if not LOCAL_TESTING else 1000

    STEPS_PER_UPDATE = 5
    MINIBATCHES = 6 if not LOCAL_TESTING else 2

    LR = 5e-4

@ex.named_config
def fixed_mdp_rnd_init():
    # NOTE: Deprecated
    LOCAL_TESTING = False
    fixed_mdp = True
    layout_name = "scenario2"

    sim_threads = 10 if LOCAL_TESTING else 50
    PPO_RUN_TOT_TIMESTEPS = 24000
    TOTAL_BATCH_SIZE = 8000

    STEPS_PER_UPDATE = 4
    MINIBATCHES = 4

    # RND_OBJS = True
    # RND_POS = True

    LR = 5e-4

@ex.named_config
def padded_all_scenario():
    # NOTE: Deprecated
    LOCAL_TESTING = False
    fixed_mdp = ["scenario2", "simple", "schelling_s", "unident_s"]
    PADDED_MDP_SHAPE = (10, 5)

    sim_threads = 10 if LOCAL_TESTING else 60
    PPO_RUN_TOT_TIMESTEPS = 40000 if not LOCAL_TESTING else 1000
    TOTAL_BATCH_SIZE = 20000 if not LOCAL_TESTING else 1000

    STEPS_PER_UPDATE = 8
    MINIBATCHES = 4

    # RND_OBJS = False
    # RND_POS = True

    LR = 5e-4
    REW_SHAPING_HORIZON = 1e7

def pbt_one_run(params, seed):
    # Iterating noptepochs over same batch data but shuffled differently
    # dividing each batch in `nminibatches` and doing a gradient step for each one
    create_dir_if_not_exists(params["SAVE_DIR"])
    save_dict_to_file(params, params["SAVE_DIR"] + "config")

    #######
    # pbt #
    #######

    mdp = OvercookedGridworld.from_layout_name(**params["mdp_params"])
    overcooked_env = OvercookedEnv(mdp, **params["env_params"])

    print("Sample training environments:")
    for _ in range(5):
        overcooked_env.reset()
        print(overcooked_env)

    gym_env = get_vectorized_gym_env(
        overcooked_env, 'Overcooked-v0', agent_idx=0, featurize_fn=lambda x: mdp.lossless_state_encoding(x), **params
    )
    gym_env.update_reward_shaping_param(1.0) # Start reward shaping from 1

    
    annealer = LinearAnnealer(horizon=params["REW_SHAPING_HORIZON"])

    # AGENT POPULATION INITIALIZATION
    population_size = params["POPULATION_SIZE"]
    pbt_population = []
    pbt_agent_names = ['agent' + str(i) for i in range(population_size)]
    print(f"population_size {population_size} len(params['LOAD_FOLDER_LST']) {len(params['LOAD_FOLDER_LST'])}")
    assert population_size == len(params['LOAD_FOLDER_LST'])
    for agent_name, load_folder in zip(pbt_agent_names, params['LOAD_FOLDER_LST']):
        if not (agent_name == 'agent0'):
            agent = PBTAgent.from_dir(load_folder, agent_name)
            agent.context=np.zeros((population_size))
            agent.context[int(agent_name[5:])]=1

            print(f'loaded model from {load_folder}')
        else: ## agent0
            agent = PBTAgent_context(agent_name, params, gym_env=gym_env)
            agent.context=np.zeros((4))
            agent.context[0]=1
            print(f'Initialized {agent_name}')
            
        print(agent.context)
        pbt_population.append(agent)
    c_identifier=Net(params['mdp_params']['layout_name'])
    c_identifier.load_state_dict(th.load('context/net_level_'+params['mdp_params']['layout_name']+'.pth'))
    c_identifier=c_identifier.cuda()
        
    print("Initialized agent models")

    all_pairs = []
    for i in range(population_size):
        for j in range(i + 1, population_size):
            all_pairs.append((i, j))

    best_sparse_rew_avg = [-np.Inf] * population_size
    best_sparse_rew_avg_2 = [-np.Inf] * population_size
    metric_np = np.zeros(population_size*2)
    metric_train_np = np.zeros(population_size*2)
    metric_dense_np = np.zeros(population_size*2)
    metric_hand1_np = np.concatenate([np.zeros(population_size),np.ones(population_size)*100])

    reward_shaping_param=1

    rand_group=False
    performance=[0+np.random.uniform() for _ in range(4*2)]
    beta=3

    # MAIN LOOP
    def population_train(epoch,save=False,best_reward0=-1,best_reward1=-1):
        for pbt_iter in range(1,2):
            print("\nEVALUATION PHASE\n")

            # Dictionary with average returns for each agent when matched with each other agent
            avg_ep_returns_dict = defaultdict(list)
            avg_ep_returns_sparse_dict = defaultdict(list)
            avg_ep_returns_dict_2 = defaultdict(list)
            avg_ep_returns_sparse_dict_2 = defaultdict(list)

            i = 0
            pbt_agent = pbt_population[i]

            # Saving each agent model at the end of the pbt iteration
            pbt_agent.update_pbt_iter_logs()

            agent_env_steps = pbt_agent.num_ppo_runs * params["PPO_RUN_TOT_TIMESTEPS"]
            reward_shaping_param,_ = annealer.param_value(agent_env_steps)
            for j in range(i, population_size):
                # Pairs each agent with all other agents including itself in assessing generalization performance
                print("Evaluating agent {} and {}".format(i, j))
                pbt_agent_other = pbt_population[j]

                agent_pair = AgentPair_context(pbt_agent.get_agent(eval=True), pbt_agent_other.get_agent(eval=True))
                #trajs = overcooked_env.get_rollouts_context(agent_pair, params["NUM_SELECTION_GAMES"], reward_shaping=reward_shaping_param)
                trajs = overcooked_env.get_rollouts_identify_c(pbt_agent,pbt_agent_other,5,c_identifier=c_identifier)
                dense_rews=[traj['ep_returns'] for traj in trajs]
                sparse_rews=[traj['ep_returns_sparse'] for traj in trajs]
                lens=[traj['ep_lengths'] for traj in trajs]
  
                rew_per_step = np.sum(dense_rews) / np.sum(lens)
                avg_ep_returns_dict[i].append(rew_per_step)
                avg_ep_returns_sparse_dict[i].append(sparse_rews)
                metric_np[i] = np.mean(sparse_rews)
                metric_dense_np[i] = np.mean(dense_rews)
                print('Average reward:',str(np.mean(sparse_rews)))
                if j != i:
                    avg_ep_returns_dict[j].append(rew_per_step)
                    avg_ep_returns_sparse_dict[j].append(sparse_rews)
                    metric_np[j] = np.mean(sparse_rews)
                    metric_dense_np[j] = np.mean(dense_rews)

                # switch the agent pair
                print("Evaluating agent {} and {}".format(j, i))    
                agent_pair = AgentPair_context(pbt_agent_other.get_agent(eval=True), pbt_agent.get_agent(eval=True))
                #trajs = overcooked_env.get_rollouts_context(agent_pair, params["NUM_SELECTION_GAMES"], reward_shaping=reward_shaping_param)
                trajs = overcooked_env.get_rollouts_identify_c(pbt_agent_other,pbt_agent,5,c_identifier=c_identifier)
                dense_rews=[traj['ep_returns'] for traj in trajs]
                sparse_rews=[traj['ep_returns_sparse'] for traj in trajs]
                lens=[traj['ep_lengths'] for traj in trajs]

                rew_per_step = np.sum(dense_rews) / np.sum(lens)
                avg_ep_returns_dict_2[i].append(rew_per_step)
                avg_ep_returns_sparse_dict_2[i].append(sparse_rews)
                metric_np[population_size+i] = np.mean(sparse_rews)
                metric_dense_np[population_size+i] = np.mean(dense_rews)
                print('Average reward:',str(np.mean(sparse_rews)))
                if j != i:
                    avg_ep_returns_dict_2[j].append(rew_per_step)
                    avg_ep_returns_sparse_dict_2[j].append(sparse_rews)
                    metric_np[population_size+j] = np.mean(sparse_rews)
                    metric_dense_np[population_size+j] = np.mean(dense_rews)

            print("AVG ep rewards dict", avg_ep_returns_dict)
            print("AVG ep rewards dict_2", avg_ep_returns_dict_2)
            print(f'Evaluation metric_np  {metric_np} {metric_np.shape}')

            i = 0
            pbt_agent = pbt_population[i]

            pbt_agent.update_avg_rew_per_step_logs(avg_ep_returns_dict[i])
            pbt_agent.update_avg_rew_per_step_logs_2(avg_ep_returns_dict_2[i])

            avg_sparse_rew = np.mean(avg_ep_returns_sparse_dict[i])
            avg_sparse_rew_2 = np.mean(avg_ep_returns_sparse_dict_2[i])
            if (avg_sparse_rew > best_reward0) and (avg_sparse_rew_2 > best_reward1):
                best_reward0=avg_sparse_rew
                best_reward1=avg_sparse_rew_2
                agent_name = pbt_agent.agent_name
                print("New best avg sparse rews {} and {} for agent {}, saving...".format(best_sparse_rew_avg, best_sparse_rew_avg_2, agent_name))
                best_save_folder = params["SAVE_DIR"] + agent_name + '/best/'
                delete_dir_if_exists(best_save_folder, verbose=True)
                pbt_agent.save_predictor(best_save_folder)
                pbt_agent.save(best_save_folder)

            #Train        
            print("\n\n\nPBT ITERATION NUM {}".format(pbt_iter))

            # TRAINING PHASE
            assert params["ITER_PER_SELECTION"] == population_size
            pairs_to_train = list(itertools.product(range(population_size), range(1)))

            for sel_iter in range(15):
                # Randomly select agents to be trained
                if params["PRIORITIZED_SAMPLING"]:
                    if params["METRIC"] == 1.0:
                        sampling_prob_np = metric_np.copy()
                    else:
                        print("METRIC version is unknown")
                        exit()
                    print(f'params["METRIC"] {params["METRIC"]}')
                    sampling_prob_np += params["EPSILON"]
                    sampling_prob_np = 1/sampling_prob_np
                    sampling_rank_np = rankdata(sampling_prob_np, method='dense')
                    print(f'sampling_rank_np {sampling_rank_np}')
                    sampling_prob_np = sampling_rank_np/sampling_rank_np.sum()
                    assert params["ALPHA"] >= 0
                    sampling_prob_np = sampling_prob_np**params["ALPHA"]
                    sampling_prob_np = sampling_prob_np/sampling_prob_np.sum()
                    print(f'sampling_prob_np {sampling_prob_np} alpha {params["ALPHA"]}')
                    pair_idx = np.random.choice(2*population_size, p=sampling_prob_np)
                    idx0 = pair_idx % population_size
                    agent_idx = pair_idx // population_size
                    idx1 = 0
                else:
                    pair_idx = np.random.choice(len(pairs_to_train))
                    idx0, idx1 = pairs_to_train.pop(pair_idx)
                    agent_idx = np.random.choice([0, 1])

                print(f'idx0 {idx0} idx1 {idx1} agent_idx {agent_idx}')
                pbt_agent0, pbt_agent1 = pbt_population[idx0], pbt_population[idx1]

                c=np.zeros((4))
                if idx0==0:
                    c[0]=1 # SP
                if idx0 in [1,2,3,4]:
                    c[1]=1 # Low
                if idx0 in [5,6,7,8,9]:
                    c[2]=1 # Mid
                if idx0 in [10,11,12,13,14]:
                    c[3]=1 # High

                c=np.repeat(np.reshape(c,(1,-1)),params['sim_threads'],0)
                # Training agent 1, leaving agent 0 fixed
                print("Training agent {} ({}) with agent {} ({}) fixed (pbt #{}/{}, sel #{}/{})".format(
                    idx1, pbt_agent1.num_ppo_runs, 
                    idx0, pbt_agent0.num_ppo_runs, 
                    pbt_iter, 1, sel_iter, 15)
                )

                agent_env_steps = pbt_agent1.num_ppo_runs * params["PPO_RUN_TOT_TIMESTEPS"]
                reward_shaping_param,_ = annealer.param_value(agent_env_steps)
                print("Current reward shaping:", reward_shaping_param, "\t Save_dir", params["SAVE_DIR"])
                pbt_agent1.logs["reward_shaping"].append(reward_shaping_param)
                gym_env.update_reward_shaping_param(reward_shaping_param)

                gym_env.other_agent = pbt_agent0.get_agent(eval=True)
                gym_env.venv.remote_set_agent_idx(agent_idx) # debug

                if params["METRIC"] == 1.0:
                    pbt_agent1.update(gym_env, metric_np=metric_np,c=c,c_identifier=c_identifier)
                    print(f'metric_np {metric_np} {metric_np.shape}')
                else:
                    print("METRIC version is unknown")
                    exit()
                ep_sparse_rew_mean = pbt_agent1.logs["ep_sparse_rew_mean"][-1]
                metric_train_np[pair_idx] = ep_sparse_rew_mean

                agent_pair = AgentPair_context(pbt_agent0.get_agent(eval=True), pbt_agent1.get_agent(eval=True))
                overcooked_env.get_rollouts_identify_c(pbt_agent0,pbt_agent1,1,c_identifier=c_identifier)

        i = 0
        pbt_agent = pbt_population[i]
        if save:
            save_folder = params["SAVE_DIR"] + pbt_agent.agent_name + '/'
            pbt_agent.save_predictor(save_folder + "pbt_iter{}/".format(epoch))
            pbt_agent.save(save_folder + "pbt_iter{}/".format(epoch))
            delete_dir_if_exists(save_folder + "pbt_iter{}/".format(epoch-1), verbose=True)
        return best_reward0,best_reward1

    def ensemble_train(epoch,save=False,sel=5,best_reward0=-1,best_reward1=-1):
        metric_np = np.zeros(population_size*2)

        i = 0
        pbt_agent = pbt_population[i]
        agent_env_steps = pbt_agent.num_ppo_runs * params["PPO_RUN_TOT_TIMESTEPS"]
        reward_shaping_param,_ = annealer.param_value(agent_env_steps)
        #sp, low,mid,high level
        group=[[pbt_population[0]],pbt_population[1:5],pbt_population[5:10],pbt_population[10:15]]
        SEL_ITER=5
        PBT_ITER=3

        for pbt_iter in range(1, PBT_ITER+1):
            if not rand_group:
                ep_r,ep_r_sparse=[],[]
                for i in range(len(pbt_population)):
                    print("Evaluating 0 and ",str(i))
                    trajectory=overcooked_env.get_rollouts_identify_c(pbt_population[0],pbt_population[i],5,c_identifier=c_identifier)
                    ep_r.append(np.mean([traj['ep_returns'] for traj in trajectory]))
                    ep_r_sparse.append(np.mean([traj['ep_returns_sparse'] for traj in trajectory]))
                    print('Average return:',str(np.mean([traj['ep_returns'] for traj in trajectory])))
                performance[0]=ep_r[0]
                performance[1]=np.mean(ep_r[1:5])
                performance[2]=np.mean(ep_r[5:10])
                performance[3]=np.mean(ep_r[10:15])
                avg_rew=np.mean(ep_r_sparse)
                
                #switch
                ep_r,ep_r_sparse=[],[]
                for i in range(len(pbt_population)):
                    print('Evaluating ',str(i),' and 0')
                    trajectory=overcooked_env.get_rollouts_identify_c(pbt_population[i],pbt_population[0],5,c_identifier=c_identifier)
                    ep_r.append(np.mean([traj['ep_returns'] for traj in trajectory]))
                    ep_r_sparse.append(np.mean([traj['ep_returns_sparse'] for traj in trajectory]))
                    print('Average return:',str(np.mean([traj['ep_returns'] for traj in trajectory])))
                performance[4]=ep_r[0]
                performance[5]=np.mean(ep_r[1:5])
                performance[6]=np.mean(ep_r[5:10])
                performance[7]=np.mean(ep_r[10:15])
                avg_rew2=np.mean(ep_r_sparse)

                if avg_rew>best_reward0 and avg_rew2>best_reward1:
                    best_reward0,best_reward1=avg_rew,avg_rew2
                    pbt_agent = pbt_population[0]
                    best_save_folder = params["SAVE_DIR"] + pbt_agent.agent_name + '/best/'
                    delete_dir_if_exists(best_save_folder, verbose=True)
                    pbt_agent.save_predictor(best_save_folder)
                    pbt_agent.save(best_save_folder)

            print(params["REW_SHAPING_HORIZON"],params["NUM_PBT_ITER"] + 1)
            print(performance)
            print("\n\n\nPBT ITERATION NUM {}".format(pbt_iter))

            # TRAINING PHASE
            assert params["ITER_PER_SELECTION"] == population_size

            for sel_iter in range(SEL_ITER):
                # Training agent 1, leaving agent 0 fixed
                if rand_group is True:
                    ind=np.random.randint(4*2)
                else:
                    p=np.zeros((4*2)) #1/rank
                    sorted_performance=np.sort(performance)
                    for i in range(4*2):
                        p[i]=(1/(sorted_performance.tolist().index(performance[i])+1))**beta
                    p/=p.sum()
                    ind=np.random.choice(np.arange(4*2),p=p)

                sample_population=group[ind%4]    
                if ind != 0 and ind!=4:
                    p=np.random.uniform(size=(len(sample_population)-1))
                    sorted_p=np.sort(p)/p.sum()
                    w=np.zeros(shape=(len(sample_population)))
                    w[0]=sorted_p[0]
                    for i in range(1,len(sample_population)-1):
                        w[i]=sorted_p[i]-sorted_p[i-1]
                    w[len(sample_population)-1]=1-sorted_p[len(sample_population)-2]
                else:
                    w=np.array([1])
                    
                np.random.shuffle(w)
                print(ind,w)
                population=[agent.get_agent() for agent in sample_population]

                c=np.zeros((4))
                if ind in [0,4]:
                    c[0]=1
                if ind in [1,5]:
                    c[1]=1
                if ind in [2,6]:
                    c[2]=1
                if ind in [3,7]:
                    c[3]=1

                c=np.repeat(np.reshape(c,(1,-1)),params['sim_threads'],0)

                pbt_agent1=pbt_population[0]
                print("Training agent {} ({}) with agent {} ({}) fixed (pbt #{}/{}, sel #{}/{})".format(
                    0, pbt_agent1.num_ppo_runs, 
                    0, pbt_agent1.num_ppo_runs, 
                    pbt_iter, PBT_ITER, sel_iter, SEL_ITER)
                )

                agent_env_steps = pbt_agent1.num_ppo_runs * params["PPO_RUN_TOT_TIMESTEPS"]
                reward_shaping_param,_ = annealer.param_value(agent_env_steps)
                print("Current reward shaping:", reward_shaping_param, "\t Save_dir", params["SAVE_DIR"])
                pbt_agent1.logs["reward_shaping"].append(reward_shaping_param)
                gym_env.update_reward_shaping_param(reward_shaping_param)

                if ind>=4:
                    agent_idx=1
                else:
                    agent_idx=0
                gym_env.venv.remote_set_agent_idx(agent_idx) # debug
                if params["METRIC"] == 1.0:
                    pbt_agent1.update(gym_env,ens_population=population,w=w, c=c,c_identifier=c_identifier,metric_np=metric_np)
                else:
                    print("METRIC version is unknown")
                    exit()
                ep_sparse_rew_mean = pbt_agent1.logs["ep_sparse_rew_mean"][-1]


                save_folder = params["SAVE_DIR"] + pbt_agent1.agent_name + '/'
                pbt_agent1.save(save_folder)

        if save:
            pbt_agent = pbt_population[0]
            save_folder = params["SAVE_DIR"] + pbt_agent.agent_name + '/'
            pbt_agent.save_predictor(save_folder + "pbt_iter{}/".format(epoch))
            pbt_agent.save(save_folder + "pbt_iter{}/".format(epoch))
            delete_dir_if_exists(save_folder + "pbt_iter{}/".format(epoch-1), verbose=True)
        return best_reward0,best_reward1

    cnt_ens,cnt_pop=0,0
    best_reward0=-1
    best_reward1=-1
    for i in range(1,params['TRAINING_ITERATIONS']):
        alpha=max(params['ALPHA_FINAL'],1-i*(1-params['ALPHA_FINAL'])/params['ALPHA_DECAY_HORIZON'])
        if not i%5 and i>80:
            save=True
        else:
            save=False
            
        if alpha<np.random.uniform():
            cnt_ens+=1
            print('Ensemble Partner ',cnt_ens)
            best_reward0,best_reward1=ensemble_train(i,save,best_reward0=best_reward0,best_reward1=best_reward1)
        else:
            cnt_pop+=1
            print('Population Partner ',cnt_pop)
            best_reward0,best_reward1=population_train(i,save,best_reward0=best_reward0,best_reward1=best_reward1)
    reset_tf()
    print(params["SAVE_DIR"])

@ex.automain
def run_pbt(params):
    create_dir_if_not_exists(params["SAVE_DIR"])
    save_dict_to_file(params, params["SAVE_DIR"] + "config")
    for seed in params["SEEDS"]:
        set_global_seed(seed)
        curr_seed_params = params.copy()
        curr_seed_params["SAVE_DIR"] += "seed_{}/".format(seed)
        pbt_one_run(curr_seed_params, seed)
