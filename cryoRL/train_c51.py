import os
import gym
import torch
import pickle
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import C51Policy
from tianshou.policy.random import RandomPolicy
from tianshou.utils import BasicLogger
from tianshou.env import DummyVectorEnv
from tianshou.data import Batch
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer

from cryoEM_dataset import get_dataset
from cryoEM_env import CryoEMEnv
from cryoEM_config import *
from dqn import NetV2
from pathlib import Path
import copy

'''
def get_dataset(dataset, use_one_hot=False):
    train_visual_feature = None
    val_visual_feature = None
    if dataset == 'CryoEM-5-5':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv', './CryoEM_data/target_CTF_A.csv')
        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv', './CryoEM_data/target_CTF_B.csv')
    #elif dataset == 'CryoEM-5-5-pred-err10':
    #    train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
    #                               './CryoEM_data/target_CTF_A_ErrorRate10_ErrorMean0_ErrorStd2.csv', ground_truth=True)
    #    val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
    #                             './CryoEM_data/target_CTF_B_ErrorRate10_ErrorMean0_ErrorStd2.csv', ground_truth=True)
    elif dataset == 'CryoEM-5-5-pred-err20':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                   './CryoEM_data/target_CTF_A_ErrorRate20_ErrorMean0_ErrorStd2.csv', ground_truth=True)
        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 './CryoEM_data/target_CTF_B_ErrorRate20_ErrorMean0_ErrorStd2.csv', ground_truth=True)
    elif dataset == 'CryoEM-5-5-pred-err30':
            train_dataset = CryoEMData('./CryoEM_data/timestamps.csv', './CryoEM_data/target_CTF_A_ErrorRate30_ErrorMean0_ErrorStd2.csv', ground_truth=True)
            val_dataset = CryoEMData('./CryoEM_data/timestamps.csv', './CryoEM_data/target_CTF_B_ErrorRate30_ErrorMean0_ErrorStd2.csv', ground_truth=True)
    elif dataset == 'CryoEM-7-3':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv', './CryoEM_data/target_CTF_A_0.7.csv')
        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv', './CryoEM_data/target_CTF_B_0.3.csv')
    elif dataset == 'CryoEM-5-5-pred-err10':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv', ctf_file='./CryoEM_data/target_CTF_A_with_conf.csv',
                                   prediction_type = CryoEMConfig.REGRESSION,
                                   prediction_file='./CryoEM_data/target_CTF_A_ErrorRate10_ErrorMean0_ErrorStd2_new.csv')
        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv', ctf_file='./CryoEM_data/target_CTF_B_with_conf.csv',
                                 prediction_type = CryoEMConfig.REGRESSION,
                                 prediction_file='./CryoEM_data/target_CTF_B_ErrorRate10_ErrorMean0_ErrorStd2_new.csv')
    elif dataset == 'CryoEM-7-3-Category-by-hl':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_categorization_res50_train_by_hl.txt',
                                    use_one_hot=use_one_hot)
        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_categorization_res50_val_by_hl.txt',
                                 use_one_hot=use_one_hot)
    elif dataset == 'CryoEM-7-3-Weighted-Category-by-hl':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_categorization_weighted_res50_train_by_hl.txt',
                                    use_one_hot=use_one_hot)
        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_categorization_weighted_res50_val_by_hl.txt',
                                 use_one_hot=use_one_hot)
    elif dataset == 'CryoEM-resnet18':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_categorization_res18_train_by_hl.txt',
                                    use_one_hot=use_one_hot)
        train_file_list = [train_dataset.get_hole(k).name for k in range(train_dataset.num_holes())]
        train_visual_feature = get_cnn_features('./CryoEM_data/resnet18-train-cnn-feature.pickle', train_file_list)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_categorization_res18_val_by_hl.txt',
                                 use_one_hot=use_one_hot)
        val_file_list = [val_dataset.get_hole(k).name for k in range(val_dataset.num_holes())]
        val_visual_feature = get_cnn_features('./CryoEM_data/resnet18-val-cnn-feature.pickle', val_file_list)

    elif dataset == 'CryoEM-resnet50':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_categorization_res50_train_by_hl.txt',
                                    use_one_hot=use_one_hot)
        train_file_list = [train_dataset.get_hole(k).name for k in range(train_dataset.num_holes())]
        train_visual_feature = get_cnn_features('./CryoEM_data/resnet50-train-cnn-feature.pickle', train_file_list)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_categorization_res50_val_by_hl.txt',
                                 use_one_hot=use_one_hot)
        val_file_list = [val_dataset.get_hole(k).name for k in range(val_dataset.num_holes())]
        val_visual_feature = get_cnn_features('./CryoEM_data/resnet50-val-cnn-feature.pickle', val_file_list)
    elif dataset == 'CryoEM-weighted-resnet50':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_categorization_weighted_res50_train_by_hl.txt',
                                    use_one_hot=use_one_hot)
        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_categorization_weighted_res50_val_by_hl.txt',
                                 use_one_hot=use_one_hot)
    elif dataset == 'CryoEM-7-3-Category':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train.csv',
                                    prediction_file='./CryoEM_data/2_categorization_train.txt',
                                    use_one_hot=use_one_hot)
        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val.csv',
                                 prediction_file='./CryoEM_data/2_categorization_val.txt',
                                 use_one_hot=use_one_hot)
    else:
        raise ValueError('Dataset {} not available.'.format(dataset))

    return train_dataset, val_dataset, train_visual_feature, val_visual_feature

def align_cnn_features(file1, file2, image_list):
    with open(file1 , 'rb') as f:
        features1 = pickle.load(f)
    with open(file2 , 'rb') as f:
        features2 = pickle.load(f)

    output = {}
    for item in image_list:
        if item in features1:
            output[item] = features1[item]
        elif item in features2:
            output[item] = features2[item]
        else:
            print (item)

    return output

def get_cnn_features(filename, image_list):
    with open(filename , 'rb') as f:
        features = pickle.load(f)

    output = np.stack([features[item] for item in image_list])
    print ('----', output.shape)
    return output
'''


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
    parser.add_argument('--num-atoms', type=int, default=51)
    parser.add_argument('--v-min', type=float, default=-10.)
    parser.add_argument('--v-max', type=float, default=10.)
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
    parser.add_argument('--random-policy', action="store_true", default=False)
    parser.add_argument('--planning', action="store_true", default=False)
    parser.add_argument('--print-trajectory', action="store_true", default=False)
    parser.add_argument('--use-one-hot', action="store_true", default=False)
    parser.add_argument('--train-prediction', action="store_true", default=False)
    parser.add_argument('--test-prediction', action="store_true", default=False)
    parser.add_argument('--use-penalty', action="store_true", default=False)
    parser.add_argument('--visual-similarity', type=float, default=0.0)
    parser.add_argument('--prediction-type', type=str, default='classification')

    parser.add_argument('--duration', type=float, default=120.0)
    parser.add_argument('--ctf-thresh', type=float, default=6.0)
    parser.add_argument('--dynamic-reward', action="store_true", default=False)

    # args = parser.parse_known_args()[0]
    args = parser.parse_args()
    return args


def update_config(args):
    if 'duration' in args:
        CryoEMConfig.Searching_Limit = args.duration
        print('duration', CryoEMConfig.Searching_Limit)

    if 'ctf_thresh' in args:
        CryoEMConfig.LOW_CTF_THRESH = args.ctf_thresh
        print('low CTF threshold', CryoEMConfig.LOW_CTF_THRESH)

    if 'feature_dim' in args:
        CryoEMConfig.FEATURE_DIM = args.feature_dim
        print('feature dim', CryoEMConfig.FEATURE_DIM)

    if 'hist_bins' in args:
        CryoEMConfig.FEATURE_HISTOGRAM_BIN = args.hist_bins
        print('feature histogram bin', CryoEMConfig.FEATURE_HISTOGRAM_BIN)


def test_c51(args=get_args()):
    #    env = gym.make(args.task)
    #    args.state_shape = env.observation_space.shape or env.observation_space.n
    #    args.action_shape = env.action_space.shape or env.action_space.n

    #    print ('Print trajectory: {}'.format(args.print_trajectory))
    print(args)

    prediction_type = CryoEMConfig.CLASSIFICATION if args.prediction_type == 'classification' else CryoEMConfig.REGRESSION
    train_dataset, val_dataset, feature_dim, category_bins, train_visual_feature, val_visual_feature = get_dataset(
        args.dataset,
        # category_bins=[0,CryoEMConfig.LOW_CTF_THRESH, 99999],
        prediction_type=prediction_type,
        use_one_hot=args.use_one_hot,
        visual_similarity=args.visual_similarity > 0.0)
    # update configuration
    args.feature_dim = feature_dim
    args.hist_bins = category_bins
    update_config(args)
    # print(CryoEMConfig)

    # only doable in cpu due to memory issue
    if train_visual_feature is not None:
        train_visual_feature = torch.from_numpy(train_visual_feature[np.newaxis, :]).cpu()
    if val_visual_feature is not None:
        val_visual_feature = torch.from_numpy(val_visual_feature[np.newaxis, :]).cpu()

    # train_envs = CryoEMEnv(train_dataset, history_size=CryoEMConfig.HISTORY_SIZE, ctf_thresh=CryoEMConfig.LOW_CTF_THRESH)
    # test_envs = CryoEMEnv(val_dataset, history_size=CryoEMConfig.HISTORY_SIZE, ctf_thresh=CryoEMConfig.LOW_CTF_THRESH)
    # !!!! each environment needs its own copy of data as the data status changes as holes are visisted
    train_envs = DummyVectorEnv([lambda: CryoEMEnv(copy.deepcopy(train_dataset),
                                                   id=k,
                                                   cryoEM_visual_feature=train_visual_feature,
                                                   visual_similarity_coeff=args.visual_similarity,
                                                   history_size=CryoEMConfig.HISTORY_SIZE,
                                                   ctf_thresh=CryoEMConfig.LOW_CTF_THRESH,
                                                   hist_bins=category_bins,
                                                   use_prediction=args.train_prediction,
                                                   dynamic_reward=args.dynamic_reward,
                                                   use_penalty=args.use_penalty) \
                                 for k in range(args.training_num)])

    # test_num set to 1 for evaluation
    test_num = args.test_num if not args.eval else 1
    test_envs = DummyVectorEnv([lambda: CryoEMEnv(copy.deepcopy(val_dataset),
                                                  id=k,
                                                  cryoEM_visual_feature=val_visual_feature,
                                                  visual_similarity_coeff=args.visual_similarity,
                                                  history_size=CryoEMConfig.HISTORY_SIZE,
                                                  ctf_thresh=CryoEMConfig.LOW_CTF_THRESH,
                                                  hist_bins=category_bins,
                                                  use_prediction=args.test_prediction,
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

    net = NetV2(
        state_shape,
        action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        softmax=True,
        num_atoms=args.num_atoms
    )

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy = C51Policy(
        net,
        optim,
        args.gamma,
        args.num_atoms,
        args.v_min,
        args.v_max,
        args.n_step,
        target_update_freq=args.target_update_freq
    ).to(args.device)

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
        # test_collector.collect(n_episode=3)
        # rain_collector.collect(n_episode=1, render=1 / 35)

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
        # pickle.dump(train_collector.buffer,
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
    model_dir = 'c51{}'.format(args.hidden_sizes[0]) if args.hidden_sizes[0] != 128 else 'c51'  # 128 is default sizes
    if args.prioritized_replay:
        model_dir += '-replay'
    model_dir += '-{}-train{}-test{}-step{}-e{}'.format(args.dataset, args.training_num, args.test_num,
                                                        args.step_per_epoch, args.epoch)
    model_dir += '-pred' if args.train_prediction else '-gt'
    model_dir += '-pred' if args.test_prediction else '-gt'
    if prediction_type == CryoEMConfig.CLASSIFICATION:
        model_dir += '-hard' if args.use_one_hot else '-soft'
    else:
        model_dir += '-regress'
    model_dir += '-ctf{}'.format(int(args.ctf_thresh))
    if args.dynamic_reward:
        model_dir += '-dR'
    if args.use_penalty:
        model_dir += '-penalty'
    if args.visual_similarity > 0:
        model_dir += '-vs{:.1f}'.format(args.visual_similarity)
    log_path = os.path.join(args.logdir, model_dir)
    Path(log_path).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer)

    # evaluation
    if args.eval:
        if args.random_policy:
            policy = RandomPolicy()
        else:
            policy = load_policy(os.path.join(log_path, 'policy.pth'), policy)
            policy.set_eps(args.eps_test)
        policy.eval()
        test_collector = Collector(policy, test_envs, exploration_noise=False)
        result = test_collector.collect(n_episode=50, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()} +/ {rews.std()}, length: {lens.mean()} +/ {lens.std()}")
        return

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        save_fn=save_fn,
        logger=logger,
        #resume_from_log=args.resume,
        save_checkpoint_fn=save_checkpoint_fn
    )
    # assert stop_fn(result['best_reward'])

if __name__ == '__main__':
    test_c51(get_args())
