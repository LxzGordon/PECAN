import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv


def traj_segment_generator(policy, env, horizon, reward_giver=None, gail=False):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param policy: (MLPPolicy) the policy
    :param env: (Gym Environment) the environment
    :param horizon: (int) the number of timesteps to run per batch
    :param reward_giver: (TransitionClassifier) the reward predicter from obsevation and action
    :param gail: (bool) Whether we are using this generator for standard trpo or with gail
    :return: (dict) generator that returns a dict with the following keys:

        - ob: (np.ndarray) observations
        - rew: (numpy float) rewards (if gail is used it is the predicted reward)
        - vpred: (numpy float) action logits
        - dones: (numpy bool) dones (is end of episode -> True if first timestep of an episode)
        - ac: (np.ndarray) actions
        - prevac: (np.ndarray) previous actions
        - nextvpred: (numpy float) next action logits
        - ep_rets: (float) cumulated current episode reward
        - ep_lens: (int) the length of the current episode
        - ep_true_rets: (float) the real environment reward
    """
    # Check when using GAIL
    assert not (gail and reward_giver is None), "You must pass a reward giver when using GAIL"

    overcooked = 'spec' in env.__dict__.keys() and env.spec.id == "Overcooked-v0"

    # Initialize state variables
    step = 0
    action = env.action_space.sample()  # not used, just so we have the datatype
    observation = env.reset()

    cur_ep_ret = 0  # return in current episode
    current_it_len = 0  # len of current iteration
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # Episode lengths

    # Initialize history arrays
    if overcooked:
        ob0, ob1 = observation
        # History will be comprised of observations for player 0
        # TODO: Figure out why we do this weird padding at the beginning!!!
        observations = np.array([ob0 for _ in range(horizon)])        
    else:
        observations = np.array([observation for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    dones = np.zeros(horizon, 'int32')
    actions = np.array([action for _ in range(horizon)])
    prev_actions = actions.copy()
    states = policy.initial_state
    done = True  # marks if we're on first timestep of an episode
    
    

    while True:

        prevac = action
        if overcooked:
            ob0, ob1 = observation
            # Have to make each action a joint action, so will use policy for thiS
            action, vpred, states, _ = policy.step(ob0.reshape(-1, *ob0.shape), states, done)
            other_action, _, _, _ = policy.step(ob1.reshape(-1, *ob1.shape), states, done)
        else:
            action, vpred, states, _ = policy.step(observation.reshape(-1, *observation.shape), states, done)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if step > 0 and step % horizon == 0:
            # Fix to avoid "mean of empty slice" warning when there is only one episode
            if len(ep_rets) == 0:
                current_it_timesteps = current_it_len
            else:
                current_it_timesteps = sum(ep_lens) + current_it_len

            yield {
                    "ob": observations,
                    "rew": rews,
                    "dones": dones,
                    "true_rew": true_rews,
                    "vpred": vpreds,
                    "ac": actions,
                    "prevac": prev_actions,
                    "nextvpred": vpred[0] * (1 - done),
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens,
                    "ep_true_rets": ep_true_rets,
                    "total_timestep": current_it_timesteps
            }
            if overcooked:
                # TODO: Figure out why this repeat?
                _, vpred, _, _ = policy.step(ob0.reshape(-1, *ob0.shape))
            else:
                _, vpred, _, _ = policy.step(observation.reshape(-1, *observation.shape))
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
            # make sure current_it_timesteps increments correctly
            current_it_len = 0
        i = step % horizon

        if overcooked:
            observations[i] = ob0
        else:
            observations[i] = observation
        
        vpreds[i] = vpred[0]
        actions[i] = action[0]
        prev_actions[i] = prevac

        clipped_action = action
        # Clip the actions to avoid out of bound error
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)

        if overcooked:
            # Clip the actions to avoid out of bound error
            clipped_other_action = other_action
            
            if isinstance(env.action_space, gym.spaces.Box):
                clipped_other_action = np.clip(other_action, env.action_space.low, env.action_space.high)
                
            # Ordering of actions is handled internally to the Gym Overcooked environment
            # Main player might have been assigned index 1 rather than index 0 to ensure both
            # roles, but handled in environment
            single_action = clipped_action
            clipped_action = np.array([[clipped_action[0], clipped_other_action[0]]])

        if gail:
            if overcooked:
                # TODO: Understand reward giver in GAIL more. This might be important, not sure.
                rew = reward_giver.get_reward(ob0, single_action[0])
            else:
                rew = reward_giver.get_reward(observation, clipped_action[0])
            observation, true_rew, done, _info = env.step(clipped_action[0])
        else:
            observation, rew, done, _info = env.step(clipped_action[0])
            true_rew = rew
        rews[i] = rew
        true_rews[i] = true_rew
        dones[i] = done

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        current_it_len += 1
        if done:
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(current_it_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            current_it_len = 0
            if not isinstance(env, VecEnv):
                observation = env.reset()
        step += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    new = np.append(seg["dones"], 0)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(rew_len, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - new[step + 1]
        delta = rew[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        gaelam[step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def flatten_lists(listoflists):
    """
    Flatten a python list of list

    :param listoflists: (list(list))
    :return: (list)
    """
    return [el for list_ in listoflists for el in list_]
