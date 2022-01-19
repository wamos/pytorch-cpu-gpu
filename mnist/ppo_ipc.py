from __future__ import print_function
import gym
import argparse
import multiprocessing as mp
import torch
from time import sleep
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3 import PPO
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

#import gym
#from stable_baselines3.ppo.ppo import PPO for testing directly in sb3 dir

def train(model: BaseAlgorithm, iter: int, num_steps: int):
  model.learn(total_timesteps=num_steps)
  print("train ", num_steps, " steps in iteration", iter)

def multiprcess_predict(queue: mp.Queue, iters: int):
  print("predict in ", iter)
  current_iter = 0
  env = gym.make("CartPole-v1")
  model = PPO("MlpPolicy", env, device="cpu", verbose=1)
  obs = env.reset()
  while(current_iter < iters):
    if(queue.empty() == False):
      params_dicts = queue.get()
      model.set_parameters(params_dicts)
      print("recv from training process!")
      current_iter = current_iter + 1
    else:
      # print("queue is empty!")
      continue
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(obs)
        if done:
          obs = env.reset()
    
def main():
  parser = argparse.ArgumentParser(description='sb3 PPO mp')
  parser.add_argument('--iters', type=int, default=10, metavar='N',
                      help='number of epochs to train (default: 14)')
  args = parser.parse_args()

  env = gym.make("CartPole-v1")
  model = PPO("MlpPolicy", env, verbose=1)
  # model.learn(total_timesteps=10)
  # params_dicts = model.get_parameters()
  # for k,v in params_dicts.items():
  #   print(k)
  #   print(type(v))
  #   # for k1, v1 in v.items():
  #   #   print(v1.size())

  # del model
  # model2 = PPO("MlpPolicy", env, verbose=1)
  # model2.set_parameters(params_dicts)

  #TODO multi-processing
  mp.set_start_method('spawn')
  ipc_queue = mp.Queue()    
  inf_device = torch.device("cpu")  
  inf_process = mp.Process(target=multiprcess_predict, args=(ipc_queue, args.iters))
  inf_started=False


  for iter in range(1, args.iters + 1):
    train(model, iter, 1000)
    if inf_started == False:
      inf_process.start()
      inf_started = True

    params_dicts = model.get_parameters()    
    ipc_queue.put(params_dicts)
    sleep(0.5)

  env.close()
  print("break from for loop")
  inf_process.join()
  print("inf_process joined")

if __name__ == '__main__':
    main()