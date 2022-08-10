from fvcore.common.registry import Registry
import torch.nn as nn
MODEL_REGISTRY = Registry('MODEL')
MODEL_REGISTRY.__doc__ = """
Registry for video model.
The registered object will be called with `obj(**vars)`.
The call should return a `torch.nn.Module` object.
"""

def build_model(args, test_mode: bool=False) -> (nn.Module, str):
    """
    Args:
        args: all options defined in opts.py and num_classes
        test_mode:
    Returns:
        network model
        architecture name
    """
    #model = MODEL_REGISTRY.get(args.backbone_net)(**vars(args))
    model = MODEL_REGISTRY.get(args.backbone_net)(pretrained=args.imagenet_pretrained)
    # print(model)
    model.fc = nn.Linear(model.fc.in_features, 1) # regression
    #model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256), nn.Dropout(p=0.5), nn.ReLU(), nn.Linear(256, 256), nn.Dropout(p=0.5), nn.ReLU(), nn.Linear(256, 1))

    # TODO:
    network_name = model.network_name if hasattr(model, 'network_name') else args.backbone_net
    arch_name = "{dataset}-{arch_name}".format(
        dataset=args.dataset, arch_name=network_name)

    if args.prefix != '':
        arch_name += f'-{args.prefix}'
    # add setting info only in training
    if not test_mode:
        arch_name += "-{}{}-bs{}-{}-e{}-l{}".format(args.lr_scheduler, "-syncbn" if args.sync_bn else "",
                                             args.batch_size, args.loss_function, args.epochs, args.lr)
    return model, arch_name


if __name__ == "__main__":
    @MODEL_REGISTRY.register()
    def abc():
        return 0

    print(MODEL_REGISTRY._obj_map.keys())
