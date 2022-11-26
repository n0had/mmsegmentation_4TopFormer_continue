from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

import numpy as np

import os
# from tqdm import tqdm
#from tqdm.notebook import tqdm
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

#from data.base_dataset import Normalize_image
#from utils.saving_utils import load_checkpoint_mgpu


def main():
    
    device = 'cuda'
    
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    #was 'cityscapes'
    parser.add_argument(
        '--palette',
        default='ade',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    
    # test a single image
    '''
    #result = inference_segmentor(model, args.img)
    result = inference_segmentor(model, False, args.img)
    
    #print(type(result))
    #print(result.size())
    
    print(type(result))
    arr_2 = np.array(result)
    print(arr_2.shape)'''
    
    # show the results
    '''
    show_result_pyplot(
        model,
        args.img,
        result,
        get_palette(args.palette),
        opacity=args.opacity,
        out_file=args.out_file)'''
    
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    
    '''
    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint_mgpu(net, checkpoint_path)
    net = net.to(device)'''
    #net = model
    model = model.eval()

    palette = get_palette_cloths(4)
    
    #img = Image.open(os.path.join(image_dir, image_name)).convert('RGB')
    img = Image.open(args.img).convert('RGB')
    img_size = img.size
    #img = img.resize((768, 768), Image.BICUBIC)######################################################################
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    
    output_tensor = model(image_tensor.to(device))
    '''
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    '''
    output_arr = output_tensor.cpu().numpy()
    
    print(type(output_tensor))
    print(type(output_arr))
    #print(result.size())
    print(output_arr.shape)
    
def get_palette_cloths(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

if __name__ == '__main__':
    main()
