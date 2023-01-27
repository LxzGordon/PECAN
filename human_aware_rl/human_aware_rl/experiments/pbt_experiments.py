import time, gym, copy, seaborn
#from human_aware_rl.pbt.pbt import PBT_DATA_DIR
import numpy as np
import tensorflow as tf
import torch as th
import matplotlib.pyplot as plt
from collections import defaultdict
from human_aware_rl.context.test_level import Net
from overcooked_ai_py.utils import save_pickle, load_pickle, load_dict_from_txt
from overcooked_ai_py.agents.agent import AgentPair_context
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

from human_aware_rl.utils import set_global_seed, prepare_nested_default_dict_for_pickle, common_keys_equal
from human_aware_rl.baselines_utils import get_pbt_agent_from_config_eval,get_pbt_agent_from_config_context
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
#from human_aware_rl.pbt.pbt import PBT_DATA_DIR

PBT_DATA_DIR='pbt_data_dir_2/'
# Visualization

def plot_pbt_runs(pbt_model_paths, seeds, single=False, save=False, show=False):
    """Plots sparse rewards"""
    for layout in pbt_model_paths.keys():
        try:
            logs_and_cfgs = get_logs(pbt_model_paths[layout], seeds=seeds)
            log, cfg = logs_and_cfgs[0]

            ep_rew_means = []
            for l, cfg in logs_and_cfgs:
                rews = np.array(l['ep_sparse_rew_mean'])
                ep_rew_means.append(rews)
            ep_rew_means = np.array(ep_rew_means)

            ppo_updates_per_pairing = int(cfg['PPO_RUN_TOT_TIMESTEPS'] / cfg['TOTAL_BATCH_SIZE'])
            x_axis = list(range(0, log['num_ppo_runs'] * cfg['PPO_RUN_TOT_TIMESTEPS'], cfg['PPO_RUN_TOT_TIMESTEPS'] // ppo_updates_per_pairing))
            plt.figure(figsize=(7,4.5))
            if single:
                for i in range(len(logs_and_cfgs)):
                    plt.plot(x_axis, ep_rew_means[i], label=str(i))
                plt.legend()
            else:
                seaborn.tsplot(time=x_axis, data=ep_rew_means)
            plt.xlabel("Environment timesteps")
            plt.ylabel("Mean episode reward")
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.tight_layout()
            if save: plt.savefig("rew_pbt_" + layout, bbox_inches='tight')
            if show: plt.show()
        except:
            continue

def get_logs(save_dir, seeds, agent_array=None):
    """
    Get training logs across seeds for all PBT runs. By default take the logs for
    agent0, but can specify custom agents through the `agent_array` parameter.
    """
    save_dir = PBT_DATA_DIR + save_dir + "/"
    logs_across_seeds = []
    if agent_array is None:
        agent_array = [0] * len(seeds)
    for seed_idx, seed in enumerate(seeds):
        seed_log = load_dict_from_txt(save_dir + "seed_{}/agent{}/logs".format(seed, agent_array[seed_idx]))
        seed_cfg = load_dict_from_txt(save_dir + "config")
        logs_across_seeds.append((seed_log, seed_cfg))
    return logs_across_seeds

# Evaluation

def evaluate_all_pbt_models(pbt_model_paths, best_bc_model_paths, num_rounds, seeds, best=False):
    pbt_performance = defaultdict(lambda: defaultdict(list))
    for layout in pbt_model_paths.keys():
        print(layout)
        pbt_performance = evaluate_pbt_for_layout(layout, num_rounds, pbt_performance, pbt_model_paths, best_bc_model_paths['test'], seeds=seeds, best=best)
    return prepare_nested_default_dict_for_pickle(pbt_performance)

def evaluate_pbt_for_layout(layout_name, num_rounds, pbt_performance, pbt_model_paths, best_test_bc_models, seeds, best=False):
    bc_agent, bc_params = get_bc_agent_from_saved(model_name=best_test_bc_models[layout_name])
    ae = AgentEvaluator(mdp_params=bc_params["mdp_params"], env_params=bc_params["env_params"])

    pbt_save_dir = PBT_DATA_DIR + pbt_model_paths[layout_name] + "/"
    pbt_config = load_dict_from_txt(pbt_save_dir + "config")
    assert common_keys_equal(bc_params["mdp_params"], pbt_config["mdp_params"]), "Mdp params differed between PBT and BC models training"
    assert common_keys_equal(bc_params["env_params"], pbt_config["env_params"]), "Env params differed between PBT and BC models training"

    #for i in range(70,110,10):
    for i in [999]:
        print('ckp ',i)
        pbt_agents = [get_pbt_agent_from_config_context(pbt_save_dir, pbt_config["sim_threads"], seed=seed, agent_idx=0, best=best,iter=i) for seed in seeds]
        eval_pbt_over_seeds(pbt_agents, bc_agent, layout_name, num_rounds, pbt_performance, ae)
    #return pbt_performance
    #pbt_agents = [get_pbt_agent_from_config_context(pbt_save_dir, pbt_config["sim_threads"], seed=seed, agent_idx=0, best=best) for seed in seeds]
    #eval_pbt_over_seeds(pbt_agents, bc_agent, layout_name, num_rounds, pbt_performance, ae)
    return pbt_performance

def eval_pbt_over_seeds(pbt_agents, bc_agent, layout_name, num_rounds, pbt_performance, agent_evaluator):
    ae = agent_evaluator
    c_identifier=Net(layout_name).cuda()
    c_identifier.load_state_dict(th.load('context/net_level_'+layout_name+'.pth'))


    for i in range(len(pbt_agents)):
        '''pbt_agents[i].context=np.array([0,1,0,0])
        pbt_and_pbt = ae.evaluate_agent_pair_context(pbt_agents[i], pbt_agents[i], num_games=num_rounds)
        avg_pbt_and_pbt = np.mean([a['ep_returns'] for a in pbt_and_pbt])
        pbt_performance[layout_name]["PBT+PBT"].append(avg_pbt_and_pbt)'''

        #pbt_agents[i].context=np.array([1,0,0,0])
        #pbt_and_bc = ae.evaluate_agent_pair_context(pbt_agents[i], bc_agent, num_games=num_rounds,c_identifier=c_identifier)
        #avg_pbt_and_bc = np.mean([a['ep_returns'] for a in pbt_and_bc])
        #pbt_performance[layout_name]["PBT+BC_0"].append(avg_pbt_and_bc)
        print('==========================================')
        bc_and_pbt = ae.evaluate_agent_pair_context(bc_agent, pbt_agents[i], num_games=num_rounds,c_identifier=c_identifier)
        avg_bc_and_pbt = np.mean([a['ep_returns'] for a in bc_and_pbt])
        pbt_performance[layout_name]["PBT+BC_1"].append(avg_bc_and_pbt)
        print(avg_bc_and_pbt)

    return pbt_performance

def run_all_pbt_experiments(best_bc_model_paths):

    # best_bc_models = load_pickle("data/bc_runs/best_bc_models")
    #seeds = [8015, 3554,  581, 5608, 4221] #581：33，1000：47
    seeds=[8015]
    '''pbt_model_paths = {
        "simple": "pbt_simple",
        "unident_s": "pbt_unident_s",
        "random1": "pbt_random1",
        "random3": "pbt_random3",
        "random0": "pbt_random0"
    }'''
    pbt_model_paths = {
      "random3": "pbt_random3",}

    # Plotting
    plot_pbt_runs(pbt_model_paths, seeds, save=True)

    # Evaluating
    set_global_seed(512)
    num_rounds = 50
    pbt_performance = evaluate_all_pbt_models(pbt_model_paths, best_bc_model_paths, num_rounds, seeds, best=True)
    save_pickle(pbt_performance, PBT_DATA_DIR + "pbt_performance")

best_bc=load_pickle("data/bc_runs/best_bc_model_paths")
run_all_pbt_experiments(best_bc)
performance=load_pickle(PBT_DATA_DIR + 'pbt_performance')
print(performance)
#performance=load_pickle(PBT_DATA_DIR + "pbt_performance_99995")
#print(performance)
# r3
# 63.6 62.4 58.4 65.2
# [74.57142857142857, 81.42857142857143, 85.71428571428571, 90.28571428571429]
# 18 14 15.6 22.4
#[30.285714285714285, 17.142857142857142, 25.142857142857142, 43.714285714285715]
