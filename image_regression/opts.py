import argparse
from models.model_builder import MODEL_REGISTRY

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action recognition Training')

    # model definition
    parser.add_argument('--backbone_net', default='s3d', type=str, help='backbone network',
                        choices=list(MODEL_REGISTRY._obj_map.keys()))
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='dropout ratio before the final layer')
    # training setting
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--loss_function', type=str, default='smoothl1', help="l1,l2,smoothl1")
    parser.add_argument('--save_prediction', action='store_true', help="Save prediction into the target files?")
    parser.add_argument('--log_norm', action='store_true', help="Use math.log to normalize the target data?")
    parser.add_argument('--evaluate_train', action='store_true', help="evaluate train_set or not?")
    parser.add_argument('--filter_confidence', action='store_true', help="Filter low confidence CTF that is lower than the confidence threshold?")
    parser.add_argument('--gpu_id', help='comma separated list of GPU(s) to use.', default=None)
    parser.add_argument('--disable_cudnn_benchmark', dest='cudnn_benchmark', action='store_false',
                        help='Disable cudnn to search the best mode (avoid OOM)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='base learning rate')
    parser.add_argument('--warmup_lr', default=None, type=float,
                        metavar='LR', help='warmup learning rate')
    parser.add_argument('--lr_scheduler', default='cosine', type=str,
                        help='learning rate scheduler',
                        choices=['step', 'multisteps', 'cosine', 'plateau', 'warmupcosine'])
    parser.add_argument('--lr_steps', default=[15, 30, 45], type=float, nargs="+",
                        metavar='LRSteps', help='[step]: use a single value: the periodto decay '
                                                'learning rate by 10. '
                                                '[multisteps] epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--nesterov', action='store_true',
                        help='enable nesterov momentum optimizer')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup_epochs', default=5, type=int, metavar='N',
                        help='number of total epochs for warmup')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--clip_gradient', '--cg', default=None, type=float,
                        help='clip the total norm of gradient before update parameter')
    parser.add_argument('--label_smoothing', default=0.0, type=float, metavar='SMOOTHING',
                        help='label_smoothing against the cross entropy')
    parser.add_argument('--no_imagenet_pretrained', dest='imagenet_pretrained',
                        action='store_false',
                        help='disable to load imagenet pretrained model')
    parser.add_argument('--transfer', action='store_true',
                        help='perform transfer learning, remove weights in the fc '
                             'layer or the original model.')
    parser.add_argument('--auto_resume', action='store_true', help='if the log folder includes a checkpoint, automatically resume')
    parser.add_argument('--lazy_eval', action='store_true', help="evaluate every 10 epochs and last 10 percentage of epochs")
    parser.add_argument('--precise_bn', action='store_true', help="use precise bn to get better stats of bn layer")
    
    # handling imbalanced data
    parser.add_argument('--weighted_data', action='store_true', help='handling imbanlanc data')

    # data-related
    parser.add_argument('-c', '--config', type=str,
                        help='YAML setting for the dataset')
    parser.add_argument('--datadir', default=None, type=str, help='Path to the dataset')
    parser.add_argument('--dataset', default=None, type=str, help='which dataset')
    parser.add_argument('-j', '--workers', default=18, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='input image size')
    parser.add_argument('--mean', type=float, nargs="+",
                        metavar='MEAN', help='mean, dimension should be 3 for RGB, 1 for flow')
    parser.add_argument('--std', type=float, nargs="+",
                        metavar='STD', help='std, dimension should be 3 for RGB, 1 for flow')
    parser.add_argument('--skip_normalization', action='store_true',
                        help='skip mean and std normalization, default use imagenet`s mean and std.')

    # logging
    parser.add_argument('--logdir', default='', type=str, help='log path')
    parser.add_argument('--prefix', default='', type=str, help='prefix in the logdir')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='frequency to print the log during the training')
    parser.add_argument('--show_model', action='store_true', help='show model summary')

    # for testing and validation
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--cpu', action='store_true', help='using cpu only')

    # for distributed learning
    parser.add_argument('--sync-bn', action='store_true',
                        help='sync BN across GPUs')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--hostfile', default='', type=str,
                        help='hostfile distributed learning')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', '--ddp', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    return parser


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()

    print(args)
