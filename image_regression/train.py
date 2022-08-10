import os
import builtins
import shutil
import time
import platform
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from fvcore.nn import flop_count, parameter_count
from torchvision.datasets import ImageFolder

from models import build_model
from utils.utils import (train, validate, evaluate, build_dataflow, get_augmentor,
                         save_checkpoint, get_class_distribution,
                         get_mean_and_std, setup_settings_in_yaml)
from utils.dataset import CryoEMDataset

from utils.lr_scheduler import CosineWarmupScheduler
from opts import arg_parser

def get_dataset(dataset, datadir):
    if dataset == 'Y1Data':
        train_img_dir = os.path.join(datadir, 'train')
        val_img_dir = os.path.join(datadir, 'val')
        train_ctf_file = os.path.join(datadir, 'target_CTF_Y_train.csv')
        val_ctf_file = os.path.join(datadir, 'target_CTF_Y_val.csv')
        # train_img_dir = os.path.join(datadir, 'val')
        # val_img_dir = os.path.join(datadir, 'train')
        # train_ctf_file = os.path.join(datadir, 'target_CTF_Y_val.csv')
        # val_ctf_file = os.path.join(datadir, 'target_CTF_Y_train.csv')
    elif dataset == 'Y2Data':
        train_img_dir = os.path.join(datadir, 'train')
        val_img_dir = os.path.join(datadir, 'val')
        train_ctf_file = os.path.join(datadir, 'target_CTF_Y_train.csv')
        val_ctf_file = os.path.join(datadir, 'target_CTF_Y_val.csv')
        # train_img_dir = os.path.join(datadir, 'val')
        # val_img_dir = os.path.join(datadir, 'train')
        # train_ctf_file = os.path.join(datadir, 'target_CTF_Y_val.csv')
        # val_ctf_file = os.path.join(datadir, 'target_CTF_Y_train.csv')
    elif dataset == 'Y3Data':
        train_img_dir = os.path.join(datadir, 'train')
        val_img_dir = os.path.join(datadir, 'val')
        train_ctf_file = os.path.join(datadir, 'target_CTF_Y_train.csv')
        val_ctf_file = os.path.join(datadir, 'target_CTF_Y_val.csv')
        # train_img_dir = os.path.join(datadir, 'val')
        # val_img_dir = os.path.join(datadir, 'train')
        # train_ctf_file = os.path.join(datadir, 'target_CTF_Y_val.csv')
        # val_ctf_file = os.path.join(datadir, 'target_CTF_Y_train.csv')
    elif dataset == 'Y4Data':
        train_img_dir = os.path.join(datadir, 'train')
        val_img_dir = os.path.join(datadir, 'val')
        train_ctf_file = os.path.join(datadir, 'target_CTF_Y_train.csv')
        val_ctf_file = os.path.join(datadir, 'target_CTF_Y_val.csv')
        # train_img_dir = os.path.join(datadir, 'val')
        # val_img_dir = os.path.join(datadir, 'train')
        # train_ctf_file = os.path.join(datadir, 'target_CTF_Y_val.csv')
        # val_ctf_file = os.path.join(datadir, 'target_CTF_Y_train.csv')
    elif dataset == 'MData':
        train_img_dir = os.path.join(datadir, 'train')
        val_img_dir = os.path.join(datadir, 'val')
        train_ctf_file = os.path.join(datadir, 'target_CTF_M_train.csv')
        val_ctf_file = os.path.join(datadir, 'target_CTF_M_val.csv')
    elif dataset == 'M2Data':
        train_img_dir = os.path.join(datadir, 'train')
        val_img_dir = os.path.join(datadir, 'val')
        train_ctf_file = os.path.join(datadir, 'target_CTF_M2_train.csv')
        val_ctf_file = os.path.join(datadir, 'target_CTF_M2_val.csv')
    else:
        raise ValueError('Dataset {} not available.'.format(dataset))

    train_dataset = CryoEMDataset(train_img_dir, train_ctf_file, use_augmentation=True)
    val_dataset = CryoEMDataset(val_img_dir, val_ctf_file, use_augmentation=False)
    return train_dataset, val_dataset

warnings.filterwarnings("ignore", category=UserWarning)

if os.environ.get('WSC', None):
    os.system("taskset -p 0xfffffffffffffffffffffffffffffffffffffffffff %d > /dev/null 2>&1" % os.getpid())

def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    args = setup_settings_in_yaml(args)

    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if args.hostfile != '':
        curr_node_name = platform.node().split(".")[0]
        with open(args.hostfile) as f:
            nodes = [x.strip() for x in f.readlines() if x.strip() != '']
            master_node = nodes[0].split(" ")[0]
        for idx, node in enumerate(nodes):
            if curr_node_name in node:
                args.rank = idx
                break
        args.world_size = len(nodes)
        args.dist_url = "tcp://{}:10598".format(master_node)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = args.cudnn_benchmark
    args.gpu = gpu

    # num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(args.dataset, args.loader_type)
    # args.num_classes = num_classes

    if args.gpu is not None:
        print(f"{platform.node().split('.')[0]}, Use GPU: {args.gpu} for training")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.rank != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    args.input_channels = 3

    model, arch_name = build_model(args)
    mean, std = get_mean_and_std(model, args)

    model = model.cuda(args.gpu)
    model.eval()

    dummy_data_shape = (1, args.input_channels, args.input_size, args.input_size)
    dummy_data = torch.rand(dummy_data_shape).cuda(args.gpu)

    if args.rank == 0:
        try:
            flops = flop_count(model, (dummy_data, ))[0]['conv']
        except:
            flops = 0
        params = parameter_count(model)['']
        torch.cuda.empty_cache()

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # the batch size should be divided by number of nodes as well
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int(args.workers / ngpus_per_node)

            if args.sync_bn:
                process_group = torch.distributed.new_group(list(range(args.world_size)))
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        # assign rank to 0
        model = torch.nn.DataParallel(model).cuda()
        args.rank = 0

    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(args.pretrained))
        if args.gpu is None:
            checkpoint = torch.load(args.pretrained, map_location='cpu')
        else:
            checkpoint = torch.load(args.pretrained, map_location='cuda:{}'.format(args.gpu))
        if args.transfer:
            new_dict = {}
            for k, v in checkpoint['state_dict'].items():
                # TODO: a better approach:
                if k.replace("module.", "").startswith("fc"):
                    continue
                new_dict[k] = v
        else:
            new_dict = checkpoint['state_dict']
        msg = model.load_state_dict(new_dict, strict=False)
        print(msg)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        print("=> creating model '{}'".format(arch_name))

    # define loss function (criterion) and optimizer
    if args.loss_function == 'l1':
        train_criterion = nn.L1Loss().cuda(args.gpu)
        val_criterion = nn.L1Loss().cuda(args.gpu)
    elif args.loss_function == 'l2':
        train_criterion = nn.MSELoss().cuda(args.gpu)
        val_criterion = nn.MSELoss().cuda(args.gpu)
    else:
        train_criterion = nn.SmoothL1Loss().cuda(args.gpu)
        val_criterion = nn.SmoothL1Loss().cuda(args.gpu)

    # Data loading code

    log_folder = os.path.join(args.logdir, arch_name)
    if args.rank == 0:
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

    #train_dataset = CryoEMDataset('/gpfs/wscgpfs02/qfan/cryo_data/hole_8bit_regression/train/regression', 'CTF_train_by_hl.csv', use_augmentation=True)
    #val_dataset = CryoEMDataset('/gpfs/wscgpfs02/qfan/cryo_data/hole_8bit_regression/val/regression', 'CTF_val_by_hl.csv', use_augmentation=False)
    train_dataset, val_dataset = get_dataset(args.dataset, args.datadir)

    weighted_sampler = None
    if args.weighted_data:
        target_list = torch.tensor(train_dataset.targets)
        class_count = [i for i in get_class_distribution(train_dataset).values()]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float)
        class_weights_all = class_weights[target_list]
        weighted_sampler = torch.utils.data.WeightedRandomSampler(weights=class_weights_all,
                                                 num_samples=len(class_weights_all),
                                                 replacement=True)

    train_loader = build_dataflow(train_dataset, is_train=True, batch_size=args.batch_size,
                                  workers=args.workers, is_distributed=args.distributed, sampler=weighted_sampler)
    val_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                  workers=args.workers, is_distributed=args.distributed)

    if args.evaluate:
        #val_top1, val_top5, val_losses, val_speed = validate(val_loader, model, val_criterion, gpu_id=args.gpu, rank=args.rank)
        val_top1, val_top5, val_losses, val_speed, score_dict1 = evaluate(val_loader, model, val_criterion, gpu_id=args.gpu, rank=args.rank)
        #if args.evaluate_train:
        #    _, _, _, _, score_dict2 = evaluate(train_loader, model, train_criterion, gpu_id=args.gpu, rank=args.rank)
        #if args.save_prediction:
        #    print(score_dict1)
        #    saveCTFdict("target_CTF_A_0.7.csv", "target_CTF_A_0.7_pred.csv",score_dict2)
        #    saveCTFdict("target_CTF_B_0.3.csv", "target_CTF_B_0.3_pred.csv",score_dict1)

        if args.rank == 0:
            logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
            print(args.pretrained, file=logfile, flush=True)
            print(f'Val@{args.input_size}: \tLoss: {val_losses:4.4f}\tTop@1: {val_top1:.4f}\t'
                  f'Top@5: {val_top5:.4f}\tSpeed: {val_speed * 1000.0:.2f} ms/batch\t'
                  f'Flops: {flops:.4f}\tParams: {params}', flush=True)
            print(f'Val@{args.input_size}: \tLoss: {val_losses:4.4f}\tTop@1: {val_top1:.4f}\t'
                  f'Top@5: {val_top5:.4f}\tSpeed: {val_speed * 1000.0:.2f} ms/batch\t'
                  f'Flops: {flops:.4f}\tParams: {params}', flush=True, file=logfile)
        return

    optimizer = torch.optim.SGD(model.parameters(), args.lr if args.warmup_lr is None else args.warmup_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, args.lr_steps[0], gamma=0.1)
    elif args.lr_scheduler == 'multisteps':
        scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_steps, gamma=0.1)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    elif args.lr_scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
    elif args.lr_scheduler == 'warmupcosine':
        scheduler = lr_scheduler.LambdaLR(optimizer, CosineWarmupScheduler(args.lr, args.warmup_lr, args.epochs, args.warmup_epochs))

    #best_top1 = 0.0
    best_loss = 99999999999999.0
    if args.auto_resume:
        checkpoint_path = os.path.join(log_folder, 'checkpoint.pth.tar')
        if os.path.exists(checkpoint_path):
            args.resume = checkpoint_path
    # optionally resume from a checkpoint
    if args.resume:
        if args.rank == 0:
            logfile = open(os.path.join(log_folder, 'log.log'), 'a')
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpu))
            args.start_epoch = checkpoint['epoch']
            # TODO: handle distributed version
            #best_top1 = checkpoint['best_top1']
            best_loss = checkpoint['best_loss']
            if not isinstance(best_top1, float):
                if args.gpu is not None:
                    #best_top1 = best_top1.to(args.gpu)
                    best_loss = best_loss.to(args.gpu)
                else:
                    #best_top1 = best_top1.cuda()
                    best_loss = best_loss.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except Exception as e:
                pass
            if args.rank == 0:
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})", flush=True)
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})", file=logfile, flush=True)
            del checkpoint  # dereference seems crucial
            torch.cuda.empty_cache()
        else:
            raise ValueError(f"Checkpoint is not found: {args.resume}")
    else:
        if os.path.exists(os.path.join(log_folder, 'log.log')) and args.rank == 0:
            shutil.copyfile(os.path.join(log_folder, 'log.log'), os.path.join(
                log_folder, 'log.log.{}'.format(int(time.time()))))
        if args.rank == 0:
            logfile = open(os.path.join(log_folder, 'log.log'), 'w')

    if args.rank == 0:
        command = " ".join(sys.argv)
        summary_writer = SummaryWriter(log_folder)
        print(command, flush=True)
        print(args, flush=True)
        print(model, flush=True)
        print(command, file=logfile, flush=True)
        print(f"FLops: {flops}, Parameters: {params}", flush=True)
        print(args, file=logfile, flush=True)

    if args.resume == '' and args.rank == 0:
        print(model, file=logfile, flush=True)
        print(f"FLops: {flops}G, Parameters: {params}", flush=True, file=logfile)

    try:
        #lrs = list(set(scheduler.get_lr()))
        #if len(lrs) != 1:
        #    print(f"Get multiple learning rates: {lrs}")
        #lr = lrs[0]
        lr = scheduler.optimizer.param_groups[0]['lr']
    except Exception as e:
        lr = None
    for epoch in range(args.start_epoch, args.epochs):
        print(f'Epoch {epoch}: learning rate = {lr:.6f}', flush=True)
        # train for one epoch
        train_top1, train_top5, train_losses, train_speed, speed_data_loader, train_steps = \
            train(train_loader, model, train_criterion, optimizer, epoch + 1,
                  display=args.print_freq, label_smoothing=args.label_smoothing,
                  clip_gradient=args.clip_gradient, gpu_id=args.gpu, rank=args.rank,
                  precise_bn=args.precise_bn)
        if args.distributed:
            dist.barrier()

        eval_this_epoch = True
        if args.lazy_eval:
            if (epoch + 1) % 10 == 0 or (epoch + 1) >= args.epochs * 0.9:
                eval_this_epoch = True
            else:
                eval_this_epoch = False

        if eval_this_epoch:
            # evaluate on validation set
            val_top1, val_top5, val_losses, val_speed = validate(val_loader, model, val_criterion,
                                                                 gpu_id=args.gpu, rank=args.rank)
        else:
            val_top1, val_top5, val_losses, val_speed = 0.0, 0.0, 0.0, 0.0

        # update current learning rate
        if args.lr_scheduler == 'plateau':
            scheduler.step(val_losses)
        else:
            scheduler.step(epoch+1)

        if args.distributed:
            dist.barrier()

        # only logging at rank 0
        if args.rank == 0:
            print(f'Train: [{epoch + 1:03d}/{args.epochs:03d}]\tLoss: {train_losses:4.4f}\t'
                  f'Top@1: {train_top1:.4f}\tTop@5: {train_top5:.4f}\tSpeed: {train_speed * 1000.0:.2f} ms/batch\t'
                  f'Data loading: {speed_data_loader * 1000.0:.2f} ms/batch', flush=True)
            print(f'Train: [{epoch + 1:03d}/{args.epochs:03d}]\tLoss: {train_losses:4.4f}\t'
                  f'Top@1: {train_top1:.4f}\tTop@5: {train_top5:.4f}\tSpeed: {train_speed * 1000.0:.2f} ms/batch\t'
                  f'Data loading: {speed_data_loader * 1000.0:.2f} ms/batch', file=logfile, flush=True)

            print(f'Val  : [{epoch + 1:03d}/{args.epochs:03d}]\tLoss: {val_losses:4.4f}\t'
                  f'Top@1: {val_top1:.4f}\tTop@5: {val_top5:.4f}\tSpeed: {val_speed * 1000.0:.2f} ms/batch', flush=True)
            print(f'Val  : [{epoch + 1:03d}/{args.epochs:03d}]\tLoss: {val_losses:4.4f}\t'
                  f'Top@1: {val_top1:.4f}\tTop@5: {val_top5:.4f}\tSpeed: {val_speed * 1000.0:.2f} ms/batch', file=logfile, flush=True)

            # remember best prec@1 and save checkpoint
            #is_best = val_top1 > best_top1
            is_best = val_losses < best_loss
            #best_loss = max(val_losses, best_loss)
            best_loss = min(val_losses, best_loss)

            save_dict = {'epoch': epoch + 1,
                         'arch': arch_name,
                         'state_dict': model.state_dict(),
                         'best_loss': best_loss,
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict()
                         }

            save_checkpoint(save_dict, is_best, filepath=log_folder)
            try:
                # get_lr get all lrs for every layer of current epoch, assume the lr for all layers are identical
                lr = scheduler.optimizer.param_groups[0]['lr']
            except Exception as e:
                lr = None

            if lr is not None:
                summary_writer.add_scalar('learning-rate', lr, epoch + 1)
            summary_writer.add_scalar('val-top1', val_top1, epoch + 1)
            summary_writer.add_scalar('val-loss', val_losses, epoch + 1)
            summary_writer.add_scalar('train-top1', train_top1, epoch + 1)
            summary_writer.add_scalar('train-loss', train_losses, epoch + 1)
            summary_writer.add_scalar('best-loss', best_loss, epoch + 1)

        if args.distributed:
            dist.barrier()

    if args.rank == 0:
        logfile.close()
        summary_writer.close()


if __name__ == '__main__':
    main()
