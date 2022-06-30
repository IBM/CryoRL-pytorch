import argparse
import os
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import A2CPolicy, ImitationPolicy
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
#from tianshou.utils import TensorboardLogger
from tianshou.utils import BasicLogger
from tianshou.utils.net.common import ActorCritic # , Net

from cryoEM_dataset import get_dataset
from cryoEM_env import CryoEMEnv
from cryoEM_config import *
from actor_critic import NetV3, ActorV2, CriticV2
import copy
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='CartPole-v0')
    parser.add_argument('--dataset', type=str, default='CryoEM-5-5')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--il-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--il-step-per-epoch', type=int, default=1000)
    parser.add_argument('--episode-per-collect', type=int, default=16)
    parser.add_argument('--step-per-collect', type=int, default=16)
    parser.add_argument('--update-per-step', type=float, default=1 / 16)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 256, 128])
    parser.add_argument('--imitation-hidden-sizes', type=int, nargs='*', default=[128])
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    #parser.add_argument(
    #    '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    #)
    # a2c special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=1.)
    parser.add_argument('--rew-norm', action="store_true", default=False)

    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval', action="store_true", default=False)
    parser.add_argument('--print-trajectory', action="store_true", default=False)
    parser.add_argument('--use-one-hot', action="store_true", default=False)
    parser.add_argument('--train-prediction', action="store_true", default=False)
    parser.add_argument('--use-penalty', action="store_true", default=False)
    parser.add_argument('--prediction-type', type=str, default='classification')

    parser.add_argument('--duration', type=float, default=120.0)
    parser.add_argument('--ctf-thresh', type=float, default=6.0)

    parser.add_argument('--action-elimination', action="store_true", default=False)

    args = parser.parse_known_args()[0]
    return args

def update_config(args):
    if 'duration' in args:
        CryoEMConfig.Searching_Limit = args.duration
        print ('duration', CryoEMConfig.Searching_Limit)

    if 'ctf_thresh' in args:
        CryoEMConfig.LOW_CTF_THRESH = args.ctf_thresh
        print ('low CTF threshold', CryoEMConfig.LOW_CTF_THRESH)

    if 'feature_dim' in args:
        CryoEMConfig.FEATURE_DIM = args.feature_dim
        print ('feature dim', CryoEMConfig.FEATURE_DIM)

    if 'hist_bins' in args:
        CryoEMConfig.FEATURE_HISTOGRAM_BIN = args.hist_bins
        print ('feature histogram bin', CryoEMConfig.FEATURE_HISTOGRAM_BIN)

def test_a2c_with_il(args=get_args()):
    torch.set_num_threads(1)  # for poor CPU

    prediction_type = CryoEMConfig.CLASSIFICATION if args.prediction_type == 'classification' else CryoEMConfig.REGRESSION
    train_dataset, val_dataset, feature_dim, category_bins = get_dataset(args.dataset,
                                                                                       #category_bins=[0,CryoEMConfig.LOW_CTF_THRESH, 99999],
                                                                                       prediction_type=prediction_type,
                                                                                       use_one_hot=args.use_one_hot)
    # update configuration
    args.feature_dim = feature_dim
    args.hist_bins = category_bins
    update_config(args)


#    env = gym.make(args.task)
#    args.state_shape = env.observation_space.shape or env.observation_space.n
#    args.action_shape = env.action_space.shape or env.action_space.n
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv([lambda: CryoEMEnv(copy.deepcopy(train_dataset),
                                                   id=k,
                                                   #history_size=CryoEMConfig.HISTORY_SIZE,
                                                   ctf_thresh=CryoEMConfig.LOW_CTF_THRESH,
                                                   #hist_bins=category_bins,
                                                   use_prediction=args.train_prediction,
                                                   action_elimination=args.action_elimination,
                                                   use_penalty=args.use_penalty) \
                                 for k in range(args.training_num)])

    # test_num set to 1 for evaluation
    test_num = args.test_num if not args.eval else 1
    test_envs = DummyVectorEnv([lambda: CryoEMEnv(copy.deepcopy(val_dataset),
                                                  id=k,
                                                  #history_size=CryoEMConfig.HISTORY_SIZE,
                                                  ctf_thresh=CryoEMConfig.LOW_CTF_THRESH,
                                                  #hist_bins=category_bins,
                                                  use_prediction=True,
                                                  action_elimination=args.action_elimination,
                                                  use_penalty=args.use_penalty,
                                                  evaluation=True,
                                                  print_trajectory=args.print_trajectory) \
                                for k in range(test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    state_shape = CryoEMConfig.HISTORY_SIZE * CryoEMConfig.FEATURE_DIM
    action_shape = 1

    # model
    print (args.hidden_sizes, state_shape)
    net = NetV3(state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorV2(net, action_shape, device=args.device).to(args.device)
    critic = CriticV2(net, device=args.device).to(args.device)
    optim = torch.optim.Adam(ActorCritic(actor, critic).parameters(), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = A2CPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        reward_normalization=args.rew_norm,
        action_space=train_envs.get_env_attr('action_space')[0]
    )
    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
    )
    test_collector = Collector(policy, test_envs)


    # log directory
    model_dir = 'a2c{}'.format(args.hidden_sizes[0]) if args.hidden_sizes[0] != 128 else 'a2c' # 128 is default sizes
    #if args.prioritized_replay:
    #    model_dir += '-replay'
    model_dir += '-{}-train{}-test{}-step{}-e{}'.format(args.dataset, args.training_num, args.test_num, args.step_per_epoch, args.epoch)
    model_dir += '-pred' if args.train_prediction else '-gt'
    model_dir += '-pred' if args.test_prediction else '-gt'
    if prediction_type == CryoEMConfig.CLASSIFICATION:
        model_dir += '-hard' if args.use_one_hot else '-soft'
    else:
        model_dir += '-regress'
    model_dir += '-ctf{}'.format(int(args.ctf_thresh))
   # if args.dynamic_reward:
   #     model_dir += '-dR'
   # if args.use_penalty:
   #     model_dir += '-penalty'
    log_path = os.path.join(args.logdir, model_dir)
    Path(log_path).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer)

    def load_policy(ckpt_path, policy):
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        policy.load_state_dict(checkpoint)
        return policy

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold

    # evaluation
    if args.eval:
        policy = load_policy(os.path.join(log_path, 'policy.pth'), policy)
#        policy.set_eps(args.eps_test)

        policy.eval()
        test_collector = Collector(policy, test_envs)
        result = test_collector.collect(n_episode=50, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()} +/ {rews.std()}, length: {lens.mean()} +/ {lens.std()}")
        return

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        episode_per_collect=args.episode_per_collect,
        #stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger
    )
 #   assert stop_fn(result['best_reward'])

    '''
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

    policy.eval()
    # here we define an imitation collector with a trivial policy
    if args.task == 'CartPole-v0':
        env.spec.reward_threshold = 190  # lower the goal
    net = NetV3(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    net = Actor(net, args.action_shape, device=args.device).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.il_lr)
    il_policy = ImitationPolicy(net, optim, action_space=env.action_space)
    il_test_collector = Collector(
        il_policy,
        DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    )
    train_collector.reset()
    result = offpolicy_trainer(
        il_policy,
        train_collector,
        il_test_collector,
        args.epoch,
        args.il_step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger
    )
    assert stop_fn(result['best_reward'])
    
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        il_policy.eval()
        collector = Collector(il_policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

    '''

if __name__ == '__main__':
    test_a2c_with_il()
