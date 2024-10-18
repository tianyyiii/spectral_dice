import os
from absl import app
from absl import flags
import numpy as np
import gym
import time
import tensorflow as tf

import sys
sys.path.append("/storage/home/hcoda1/6/tchen667/spectral-dice")
from dice_rl.environments.env_policies import get_target_policy
from dice_rl.data.dataset import Dataset, EnvStep, StepType
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset
import dice_rl.data.dataset as dataset_lib

def seconds_to_hours_minutes(seconds):
    hours = seconds // 3600  
    minutes = (seconds % 3600) / 60  
    return hours, minutes


class TabularSpederAgent():
  def __init__(self, dataset_path, env_name, policy_path, feature_dim=1024, info=False):
    self.dataset = Dataset.load(dataset_path)
    self.target_policy = get_target_policy(policy_path, env_name, tabular_obs=True)
    if env_name == "taxi":
      self.state_num = 2000
      self.action_num = 6
    elif env_name == "four_rooms":
      self.state_num = 121
      self.action_num = 4
    self.num = self.action_num * self.state_num
    self.feature_dim = feature_dim
    self.p_table = np.zeros((self.state_num, self.action_num, self.state_num, self.action_num))
    self.p_table_one = np.zeros((self.state_num, self.action_num))
    self.reward = np.zeros(self.num)
    self.mu_0 = np.zeros(self.num)
    self.info = info

  def get_index(self, state, action):
    return state * self.action_num + action
  
  def train(self):
    '''
    First step: construct the prob matrix
    '''
    #parallelize some of the operations
    start_time = time.time()
    all_steps = self.dataset.get_all_steps(num_steps=2)
    for i in range(all_steps.observation.shape[0]):
      first_state = all_steps.observation[i, 0]
      second_state = all_steps.observation[i, 1]
      first_action = all_steps.action[i, 0]
      self.p_table[first_state, first_action, second_state, :] += 1 
    #note: p_table is a (state_num, action_num, state_num, action_num) np array, each item is the joint probability distribution

    all_steps = self.dataset.get_all_steps()
    tfagents_timestep = dataset_lib.convert_to_tfagents_timestep(all_steps)
    action_probs = []
    for action in range(self.action_num):
        prob = np.exp(self.target_policy.distribution(tfagents_timestep).action.log_prob(action))
        action_probs.append(prob)
    action_probs = np.stack(action_probs, axis=-1)
    indicator = np.zeros(self.state_num)
    for i in range(all_steps.observation.shape[0]):
      second_state = all_steps.observation[i]
      if indicator[second_state] == 0:
        self.p_table[:, :, second_state, :] *= action_probs[i, :]
        indicator[second_state] = 1 
    self.p_table = self.p_table / np.sum(self.p_table)
    #note: action_probs is a (step_num, action_num) np array

    for i in range(all_steps.observation.shape[0]):
      first_state = all_steps.observation[i]
      first_action = all_steps.action[i]
      self.p_table_one[first_state, first_action] += 1 
    self.p_table_one = self.p_table_one / np.sum(self.p_table_one)
    #note: p_table_one is a (state_num, action_num) np array, each item is the joint probability of P(s, a)

    epsilon = 1e-11  
    print(np.min(self.p_table), "p_table_min")
    print(np.max(self.p_table), "p_table_before")
    print(np.mean(self.p_table), "p_table_mean")
    print(np.count_nonzero(self.p_table == 0), "p_table_0_count")
    print(np.min(self.p_table_one), "p_table_one")    
    print(np.mean(self.p_table_one), "p_table_one_mean")
    print(np.max(self.p_table_one), "p_table_one_max")
    p_table_before = self.p_table.copy()
    # epsilon = 1e-3
    # self.p_table /= ((self.p_table_one[:, :, np.newaxis, np.newaxis] + epsilon) * (self.p_table_one[np.newaxis, np.newaxis, :, :] + epsilon))
    self.p_table /= ((self.p_table_one[:, :, np.newaxis, np.newaxis]) * (self.p_table_one[np.newaxis, np.newaxis, :, :]) + epsilon)

    if self.info:
      for i in range(all_steps.observation.shape[0]):
        step_index = self.get_index(all_steps.observation[i], all_steps.action[i])
        self.reward[step_index] = all_steps.reward[i]

      first_step = self.dataset.get_all_episodes(truncate_episode_at=1)
      first_step = first_step[0]
      first_step = first_step._replace(observation=tf.squeeze(first_step.observation))
      tfagents_timestep = dataset_lib.convert_to_tfagents_timestep(first_step)
      for action in range(self.action_num):
        prob = np.exp(self.target_policy.distribution(tfagents_timestep).action.log_prob(action))
        for i in range(first_step.observation.shape[0]):
          init_state = first_step.observation[i]
          s_a_index = self.get_index(init_state, action)
          self.mu_0[s_a_index] += prob[i]
      self.mu_0 = self.mu_0 / np.sum(self.mu_0)

    '''
    Second step: use SVD to get phi and mu
    '''
    flat_indices = np.argsort(self.p_table, axis=None)[-10:][::-1] 
    top_10_indices = np.unravel_index(flat_indices, self.p_table.shape)
    for idx in range(len(flat_indices)):
      print(f"index: {tuple(top_10_indices[i][idx] for i in range(4))}, value: {self.p_table[top_10_indices[0][idx], top_10_indices[1][idx], top_10_indices[2][idx], top_10_indices[3][idx]]}")
      print(p_table_before[top_10_indices[0][idx], top_10_indices[1][idx], top_10_indices[2][idx], top_10_indices[3][idx]])
      print(self.p_table_one[top_10_indices[0][idx], top_10_indices[1][idx]])
      print(self.p_table_one[top_10_indices[2][idx], top_10_indices[3][idx]])
    self.p_table = np.reshape(self.p_table, (self.num, self.num))
    print(np.max(self.p_table), "p_table")
    print(np.mean(self.p_table), "p_table_mean_after")
    U, S, Vt = np.linalg.svd(self.p_table, full_matrices=True)
    S_diag = np.diag(S)
    sqrt_S = np.sqrt(S_diag)
    U = np.dot(U, sqrt_S)
    Vt = np.dot(sqrt_S, Vt)
    U = U[:, :self.feature_dim]
    Vt = Vt[:self.feature_dim, :]
    #note: U is a (num, feature_dim) np array, Vt is a (feature_dim, num) np array

    end_time = time.time()
    time_usage = end_time - start_time
    hours, minutes = seconds_to_hours_minutes(time_usage)
    print(f"Stage 1 finished with {hours} hours and {minutes} minutes.")
    return U, Vt, self.p_table, self.reward, self.mu_0, self.p_table_one

  