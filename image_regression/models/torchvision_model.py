import torchvision.models as tvmodels

from . import MODEL_REGISTRY


def register_torchvision_models():
    # TODO: a hacky way to get all torchvision models
    object_methods = [method_name for method_name in dir(tvmodels)
                      if callable(getattr(tvmodels, method_name)) and method_name[0].islower()]

    for m in object_methods:
        model = getattr(tvmodels, m)
        MODEL_REGISTRY.register(model)


register_torchvision_models()
