
#from timm.models import vision_transformer

from . import MODEL_REGISTRY

'''
def register_vision_tranformer():
    # TODO: a hacky way to get all torchvision models
    object_methods = [method_name for method_name in dir(vision_transformer)
                      if callable(getattr(vision_transformer, method_name)) and method_name.startswith('vit')]

    for m in object_methods:
        model = getattr(vision_transformer, m)
        MODEL_REGISTRY.register(model)


register_vision_tranformer()
'''
