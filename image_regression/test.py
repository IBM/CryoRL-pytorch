import os
import time

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm
from fvcore.nn import flop_count, parameter_count

from models import build_model
from utils.utils import (build_dataflow, AverageMeter,
                         accuracy, get_num_input_channels, get_mean_and_std, setup_settings_in_yaml)
from utils.video_transforms import *
from utils.video_dataset import get_dataloader
from opts import arg_parser

if os.environ.get('WSC', None):
    os.system("taskset -p 0xfffffffffffffffffffffffffffffffffffffffffff %d > /dev/null 2>&1" % os.getpid())


def load_categories(file_path):
    id_to_label = {}
    label_to_id = {}
    with open(file_path) as f:
        cls_id = 0
        for label in f.readlines():
            label = label.strip()
            if label == "":
                continue
            id_to_label[cls_id] = label
            label_to_id[label] = cls_id
            cls_id += 1
    return id_to_label, label_to_id


def eval_a_batch(data, model, num_clips=1, num_crops=1, threed_data=False):
    with torch.no_grad():
        batch_size = data.shape[0]
        if threed_data:
            tmp = torch.chunk(data, num_clips * num_crops, dim=2)
            data = torch.cat(tmp, dim=0)
        else:
            data = data.view((batch_size * num_crops * num_clips, -1) + data.size()[2:])
        result = model(data)

        if threed_data:
            tmp = torch.chunk(result, num_clips * num_crops, dim=0)
            result = None
            for i in range(len(tmp)):
                result = result + tmp[i] if result is not None else tmp[i]
            result /= (num_clips * num_crops)
        else:
            result = result.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)

    return result


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    args = setup_settings_in_yaml(args)
    cudnn.benchmark = args.cudnn_benchmark
    data_list = args.val_list if args.evaluate else args.test_list

    if args.dataset == 'st2stv1':
        id_to_label, label_to_id = load_categories(os.path.join(args.datadir, args.label_file))

    if args.gpu_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    args.input_channels = get_num_input_channels(args.modality)
    model, arch_name = build_model(args, test_mode=True)
    mean, std = get_mean_and_std(model, args)
    model = model.cuda()
    model.eval()

    if args.threed_data:
        dummy_data_shape = (1, args.input_channels, args.groups, args.input_size, args.input_size)
    else:
        dummy_data_shape = (1, args.input_channels * args.groups, args.input_size, args.input_size)
    dummy_data = torch.rand(dummy_data_shape).cuda(args.gpu)
    flops = flop_count(model, (dummy_data, ))[0]['conv']
    params = parameter_count(model)['']

    flops = flops * (args.num_clips * args.num_crops)
    model = torch.nn.DataParallel(model).cuda()
    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> creating model '{}'".format(arch_name))

    # augmentor
    if args.disable_scaleup:
        scale_size = args.input_size
    else:
        scale_size = int(args.input_size / 0.875 + 0.5)

    augments = []
    if args.num_crops == 1:
        augments += [
            GroupScale(scale_size),
            GroupCenterCrop(args.input_size)
        ]
    else:
        flip = True if args.num_crops == 10 else False
        augments += [
            GroupOverSample(args.input_size, scale_size, num_crops=args.num_crops, flip=flip),
        ]
    augments += [
        Stack(threed_data=args.threed_data),
        ToTorchFormatTensor(num_clips_crops=args.num_clips * args.num_crops),
        GroupNormalize(mean=mean, std=std, threed_data=args.threed_data)
    ]

    augmentor = transforms.Compose(augments)

    # Data loading code
    sample_offsets = list(range(-args.num_clips // 2 + 1, args.num_clips // 2 + 1))
    print("Image is scaled to {} and crop {}".format(scale_size, args.input_size))
    print("Number of crops: {}".format(args.num_crops))
    print("Number of clips: {}, offset from center with {}".format(args.num_clips, sample_offsets))

    val_dataset = get_dataloader(args.loader_type, args.datadir, data_list, args.groups, args.frames_per_group,
                                 num_clips=args.num_clips, modality=args.modality,
                                 image_tmpl=args.image_tmpl, dense_sampling=args.dense_sampling,
                                 fixed_offset=not args.random_sampling,
                                 transform=augmentor, is_train=False, test_mode=not args.evaluate,
                                 seperator=args.separator, filter_video=args.filter_video)

    data_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                 workers=args.workers)

    log_folder = os.path.join(args.logdir, arch_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    batch_time = AverageMeter()
    if args.evaluate:
        logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
        top1 = AverageMeter()
        top5 = AverageMeter()
    else:
        logfile = open(os.path.join(log_folder,
                                    'test_{}crops_{}clips_{}.csv'.format(args.num_crops,
                                                                         args.num_clips,
                                                                         args.input_size))
                       , 'w')

    total_outputs = 0
    outputs = np.zeros((len(data_loader) * args.batch_size, args.num_classes))
    # switch to evaluate mode
    model.eval()
    total_batches = len(data_loader)
    with torch.no_grad(), tqdm(total=total_batches) as t_bar:
        end = time.time()
        for i, (video, label) in enumerate(data_loader):
            output = eval_a_batch(video, model, num_clips=args.num_clips,
                                  num_crops=args.num_crops, threed_data=args.threed_data)
            if args.evaluate:
                label = label.cuda(non_blocking=True)
                # measure accuracy
                prec1, prec5 = accuracy(output, label, topk=(1, 5))
                top1.update(prec1[0], video.size(0))
                top5.update(prec5[0], video.size(0))
                output = output.data.cpu().numpy().copy()
                batch_size = output.shape[0]
                outputs[total_outputs:total_outputs + batch_size, :] = output
            else:
                # testing, store output to prepare csv file
                # measure elapsed time
                output = output.data.cpu().numpy().copy()
                batch_size = output.shape[0]
                outputs[total_outputs:total_outputs + batch_size, :] = output
                predictions = np.argsort(output, axis=1)
                for ii in range(len(predictions)):
                    # preds = [id_to_label[str(pred)] for pred in predictions[ii][::-1][:5]]
                    temp = predictions[ii][::-1][:5]
                    preds = [str(pred) for pred in temp]
                    if args.dataset == 'st2stv1':
                        print("{};{}".format(label[ii], id_to_label[int(preds[0])]), file=logfile)
                    else:
                        print("{};{}".format(label[ii], ";".join(preds)), file=logfile)
            total_outputs += video.shape[0]
            batch_time.update(time.time() - end)
            end = time.time()
            if args.evaluate:
                t_bar.set_description(f"Top1: {top1.avg:.2f}")
            t_bar.update(1)

        # if not args.evaluate:
        outputs = outputs[:total_outputs]
        print(f"Predict {total_outputs} videos.", flush=True)
        np.save(os.path.join(log_folder, '{}_{}crops_{}clips_{}_details.npy'.format("val" if args.evaluate else "test", args.num_crops, args.num_clips, args.input_size)), outputs)

    if args.evaluate:
        print(args.pretrained, file=logfile, flush=True)
        print(f'Val@{args.input_size}({scale_size}) (# crops = {args.num_crops}, # clips = {args.num_clips}): \t'
              f'Top@1: {top1.avg:.4f}\tTop@5: {top5.avg:.4f}\tFLOPs: {flops:.4f}\tParams:{params}', flush=True)
        print(f'Val@{args.input_size}({scale_size}) (# crops = {args.num_crops}, # clips = {args.num_clips}): \t'
              f'Top@1: {top1.avg:.4f}\tTop@5: {top5.avg:.4f}\tFLOPs: {flops:.4f}\tParams:{params}', flush=True, file=logfile)

    logfile.close()


if __name__ == '__main__':
    main()
