import os
import gym
import torch
import pickle
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import DQNPolicy
from tianshou.policy.random import RandomPolicy
from tianshou.utils import BasicLogger
from tianshou.env import DummyVectorEnv
from tianshou.data import Batch
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer

from cryoEM_dataset import get_dataset
from cryoEM_env import CryoEMEnv
from cryoEM_config import *
from dqn import NetV1, NetV2, DummyNet
#from dqn import Net
from pathlib import Path
import copy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='CryoEM')
    parser.add_argument('--dataset', type=str, default='CryoEM-5-5')
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--step-per-epoch', type=int, default=10000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-sizes', type=int,
                        nargs='*', default=[128, 256, 128])
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--prioritized-replay',
                        action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument(
        '--save-buffer-name', type=str,
        default="./expert_DQN_CartPole-v0.pkl")
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--eval', action="store_true", default=False)
    parser.add_argument('--eval-model', type=str, default=None)
    parser.add_argument('--random-policy', action="store_true", default=False)
    parser.add_argument('--planning', action="store_true", default=False)
    parser.add_argument('--print-trajectory', action="store_true", default=False)
    parser.add_argument('--use-one-hot', action="store_true", default=False)
    parser.add_argument('--train-prediction', action="store_true", default=False)
#    parser.add_argument('--test-prediction', action="store_true", default=False)
    parser.add_argument('--use-penalty', action="store_true", default=False)
    parser.add_argument('--prediction-type', type=str, default='classification')
    parser.add_argument('--gaussian-filter-size', type=float, default=0.0)

    parser.add_argument('--duration', type=float, default=120.0)
    parser.add_argument('--ctf-thresh', type=float, default=6.0)
    parser.add_argument('--dynamic-reward', action="store_true", default=False)
    parser.add_argument('--dueling', action="store_true", default=False)
    
    parser.add_argument('--action-elimination', action="store_true", default=False)

    #args = parser.parse_known_args()[0]
    args = parser.parse_args()
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

    if 'gaussian_filter_size' in args:
        CryoEMConfig.GAUSS_FILTER_SIZE = args.gaussian_filter_size
        print ('gaussian filter size', CryoEMConfig.GAUSS_FILTER_SIZE)

def test_dqn(args=get_args()):
#    env = gym.make(args.task)
#    args.state_shape = env.observation_space.shape or env.observation_space.n
#    args.action_shape = env.action_space.shape or env.action_space.n

#    print ('Print trajectory: {}'.format(args.print_trajectory))
    print (args)
   
    prediction_type = CryoEMConfig.CLASSIFICATION if args.prediction_type == 'classification' else CryoEMConfig.REGRESSION
    train_dataset, val_dataset, feature_dim, feature_hist_bins = get_dataset(args.dataset, prediction_type=prediction_type, use_one_hot=args.use_one_hot)

    #for k in range(val_dataset.num_holes()):
    #    print (k, val_dataset.get_hole(k).parent_idx)

    # update configuration
    args.feature_dim = feature_dim
    args.hist_bins = feature_hist_bins
    update_config(args)
    #print(CryoEMConfig)

    #train_envs = CryoEMEnv(train_dataset, history_size=CryoEMConfig.HISTORY_SIZE, ctf_thresh=CryoEMConfig.LOW_CTF_THRESH)
    #test_envs = CryoEMEnv(val_dataset, history_size=CryoEMConfig.HISTORY_SIZE, ctf_thresh=CryoEMConfig.LOW_CTF_THRESH)
    # !!!! each environment needs its own copy of data as the data status changes as holes are visisted
    train_envs =  DummyVectorEnv([lambda:CryoEMEnv(copy.deepcopy(train_dataset),
                                                   id=k,
#                                                   history_size=CryoEMConfig.HISTORY_SIZE,
                                                   ctf_thresh=CryoEMConfig.LOW_CTF_THRESH,
#                                                   hist_bins=feature_hist_bins,
                                                   use_prediction=args.train_prediction,
                                                   action_elimination=args.action_elimination,
                                                   dynamic_reward=args.dynamic_reward,
                                                   use_penalty=args.use_penalty) \
                  for k in range(args.training_num)])

    # test_num set to 1 for evaluation
    test_num = args.test_num if not args.eval else 1
    test_envs =  DummyVectorEnv([lambda:CryoEMEnv(copy.deepcopy(val_dataset),
                                                  id=k,
#                                                  history_size=CryoEMConfig.HISTORY_SIZE,
                                                  ctf_thresh=CryoEMConfig.LOW_CTF_THRESH,
#                                                  hist_bins=feature_hist_bins,
                                                  use_prediction=True,
                                                  action_elimination=args.action_elimination,
                                                  dynamic_reward=args.dynamic_reward,
                                                  use_penalty=args.use_penalty,
                                                  evaluation=True,
                                                  planning=args.planning,
                                                  print_trajectory=args.print_trajectory) \
                 for k in range(test_num)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    state_shape = CryoEMConfig.HISTORY_SIZE * CryoEMConfig.FEATURE_DIM
    action_shape = 1

    Q_param = V_param = {"hidden_sizes": [128]}
    if not args.planning:
        if args.dueling:
            net = NetV2(state_shape, action_shape,
                  hidden_sizes=args.hidden_sizes, device=args.device,
                   dueling_param=(Q_param, V_param),
                  ).to(args.device)
        else:
            net = NetV2(state_shape, action_shape,
                  hidden_sizes=args.hidden_sizes, device=args.device,
                  ).to(args.device)

    else:
        assert args.eval, "planning mode can only be used for evaluation."
        net = DummyNet(state_shape, action_shape,
              hidden_sizes=args.hidden_sizes, device=args.device,
              # dueling=(Q_param, V_param),
              ).to(args.device)

#net = Net(state_shape, action_shape, hidden_dim=256).to(args.device)

    #net = Net(state_shape, action_shape, hidden_dim=256)
    #device_ids = [args.gpu]
    #print (device_ids)
    #os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
    ##device_ids = list(range(len(device_ids)))
    #net = net.cuda(device_ids[0])

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = DQNPolicy(
                net, optim, args.gamma, args.n_step,
                target_update_freq=args.target_update_freq)

    buf = None
    if not args.eval:
        # buffer
        if args.prioritized_replay:
            buf = PrioritizedVectorReplayBuffer(
                args.buffer_size, buffer_num=len(train_envs),
                alpha=args.alpha, beta=args.beta)
        else:
            buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))

    # collector
    if not args.eval:
        train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
        test_collector = Collector(policy, test_envs, exploration_noise=False)
        # policy.set_eps(1)
        train_collector.collect(n_step=args.batch_size * args.training_num)
        #test_collector.collect(n_episode=3)
        #rain_collector.collect(n_episode=1, render=1 / 35)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def load_policy(ckpt_path, policy):
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        policy.load_state_dict(checkpoint)
        return policy

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save({
            'model': policy.state_dict(),
            'optim': optim.state_dict(),
        }, os.path.join(log_path, 'checkpoint.pth'))
        #pickle.dump(train_collector.buffer,
        #        open(os.path.join(log_path, 'train_buffer.pkl'), "wb"))

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold

    '''
    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                  40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)
    '''

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        step_per_epoch = args.step_per_epoch
        if env_step <= step_per_epoch:
            policy.set_eps(args.eps_train)
        elif env_step <= 5 * step_per_epoch:
            eps = args.eps_train - (env_step - step_per_epoch + 1e-4) / \
                  (4 * step_per_epoch) * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # log directory
    model_dir = 'dqn{}'.format(args.hidden_sizes[0]) if args.hidden_sizes[0] != 128 else 'dqn' # 128 is default sizes
    if args.prioritized_replay:
        model_dir += '-replay'
    model_dir += '-{}-train{}-test{}-step{}-e{}'.format(args.dataset, args.training_num, args.test_num, args.step_per_epoch, args.epoch)
    model_dir += '-pred' if args.train_prediction else '-gt'
#    model_dir += '-pred' if args.test_prediction else '-gt'
    if prediction_type == CryoEMConfig.CLASSIFICATION:
        model_dir += '-hard' if args.use_one_hot else '-soft'
    else:
        model_dir += '-regress'
    model_dir += '-ctf{}'.format(int(args.ctf_thresh))
    if args.dueling:
        model_dir += '-duel'
    if args.gaussian_filter_size > 0:
        model_dir += '_smooth{:4.1f}'.format(args.gaussian_filter_size)
    if args.dynamic_reward:
        model_dir += '-dR'
    if args.use_penalty:
        model_dir += '-penalty'
    log_path = os.path.join(args.logdir, model_dir)

    # skip directory creation in the evaluation mode
    if not args.eval:
        Path(log_path).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer)

    
    # evaluation
    if args.eval:
        if args.random_policy:
            policy = RandomPolicy()
        else:
            eval_model_path = os.path.join(log_path, 'policy.pth') if args.eval_model is None else args.eval_model
            policy = load_policy(eval_model_path, policy)
            policy.set_eps(args.eps_test)
        policy.eval()
        test_collector = Collector(policy, test_envs, exploration_noise=False)
        result = test_collector.collect(n_episode=50, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()} +/ {rews.std()}, length: {lens.mean()} +/ {lens.std()}")
        return
   
    
    # trainer
    result = offpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.step_per_collect, test_num,
        args.batch_size, update_per_step=args.update_per_step, train_fn=train_fn,
        test_fn=test_fn, save_fn=save_fn, logger=logger, verbose=True)
   # assert stop_fn(result['best_reward'])
    
    '''
    # save buffer in pickle format, for imitation learning unittest
    buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(test_envs))
    policy.set_eps(0.2)
    collector = Collector(policy, test_envs, buf, exploration_noise=False)
    result = collector.collect(n_step=args.buffer_size)
    #pickle.dump(buf, open(args.save_buffer_name, "wb"))
    print(result["rews"].mean())
    '''

def test_pdqn(args=get_args()):
    args.prioritized_replay = True
    args.gamma = .95
    args.seed = 1
    #args.seed = 100
    test_dqn(args)


if __name__ == '__main__':
    import time
    start = time.time()
    test_dqn(get_args())
    end = time.time()
    print('time used:', end - start)
