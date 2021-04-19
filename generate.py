import argparse
import math

import torch
from torchvision import utils

from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

from networks import ResnetGenerator 


@torch.no_grad()
def sample(generator, step, mean_style, n_sample, device):
    image = generator(
        torch.randn(n_sample, 512).to(device),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
    
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type =str, default='/content/generate')
    parser.add_argument('--in_dir', type = str, default='/content/horse2zebra')
    parser.add_argument('--path', type=str, default='/content/test.pt', help='path to checkpoint file')
    parser.add_argument('--num_img', type=int, default=120)
    
    args = parser.parse_args()
    
    device = 'cuda'

    genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=256, light=True).to(device)
    genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=256, light=True).to(device)

    params = torch.load(args.path)
    genA2B.load_state_dict(params['genA2B'])
    genB2A.load_state_dict(params['genB2A'])

    genA2B.eval()
    genB2A.eval()


    test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    testA = ImageFolder(os.path.join(args.in_dir, 'testA'), test_transform)
    testB = ImageFolder(os.path.join(args.in_dir, 'testB'), test_transform)

    testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
    testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

    for i in range(args.num_img):
      img = sample(generator, step, mean_style, 1, device)
      name = args.dir + '/' + 'sample' + str(i) + '.png'
      utils.save_image(img, name, normalize=True, range=(-1, 1))
    
