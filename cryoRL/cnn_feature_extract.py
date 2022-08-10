import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import os
import argparse
import numpy as np
import glob

def get_vector(image_name, model, layer, feature_size=512):
    # Image transforms
    scaler = transforms.Scale((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(feature_size)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding.numpy()


def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch evaluation')
    parser.add_argument('--image-dir', type=str, default='', help="where is the data")
    parser.add_argument('--model', type=str, default='resnet18', help="where is the data")
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser

def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()

    # Load the pretrained model
    model = models.resnet18(pretrained=True)
    model = model.to(args.device)

    layer = model._modules.get('avgpool')
    # Set model to evaluation mode
    model.eval()

    all_imgs = glob.glob(args.image_dir+"/*.png")
    for img_file in all_imgs:
        print (img_file)
        v = get_vector(img_file, model=model, layer=layer, feature_size=512)
#        img_file = os.path.basename(img).split('.')[0]
        print (v)

if __name__ == '__main__':
    main()
