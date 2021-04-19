import argparse
import math

import torch
from torchvision import utils
from utils import *
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

from networks import ResnetGenerator 

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
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    testA = ImageFolder(os.path.join(args.in_dir, 'testA'), test_transform)
    testB = ImageFolder(os.path.join(args.in_dir, 'testB'), test_transform)

    testA_loader = DataLoader(testA, batch_size=1, shuffle=False)
    testB_loader = DataLoader(testB, batch_size=1, shuffle=False)

    for i in range(args.num_img):
      try:
          real_A, _ = testA_iter.next()
          real_B, _ = testB_iter.next()
      except:
          testA_iter = iter(testA_loader)
          real_A, _ = testA_iter.next()
          testB_iter = iter(testB_loader)
          real_B, _ = testB_iter.next()

      real_A, real_B = real_A.to(device), real_B.to(device)

      fake_A2B, _, __ = genA2B(real_A)
      fake_B2A, _, __ = genB2A(real_B)

      A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
      B2A = RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))

      nameA2B = args.out_dir + '/A2B/' + 'sample' + str(i) + '.png'
      cv2.imwrite(nameA2B, A2B * 255.0)
      
      nameB2A = args.out_dir + '/B2A/' + 'sample' + str(i) + '.png'
      cv2.imwrite(nameB2A, B2A * 255.0)
    
