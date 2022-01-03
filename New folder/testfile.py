import os

import numpy as np
import torch
import torch.backends.cudnn as cuda
from PIL.Image import fromarray
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from configs.Config import TestConfig
# Check if CUDA is available
from src.Generator import Generator
from src.ProjectDataset import create_dataset
from utils.utils import sample, process

# cuda availability
is_cuda = torch.cuda.is_available()
if is_cuda:
    print('Cuda is available')
    cuda.enable = True
    cuda.benchmark = True

# Get Test Configurations
opts = TestConfig().parse

# Create Directory to store Result
os.makedirs('{:s}'.format(opts.result), exist_ok=True)

# Generator Initialize
generator = Generator(use_spectral_norm=True)
if opts.model != '':

    # load pretrained data
    dis_weights_path = os.path.join(opts.model, 'EdgeModel_dis.pth')
    gen_weights_path = os.path.join(opts.model, 'EdgeModel_gen.pth')

    if os.path.exists(gen_weights_path):

        if is_cuda:
            data = torch.load(gen_weights_path)
        else:
            data = torch.load(gen_weights_path, map_location=lambda storage, loc: storage)

        generator.load_state_dict(data['generator'])
    else:
        print('Please provide path with trained weights')
else:
    print('Please provide pre trained weights')

if is_cuda:
    generator = generator.cuda()

# Dataset Initialize
dataset = create_dataset(opts)

data_loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers, drop_last=False)
data_loader = sample(data_loader)


def tocuda(*args):
    return (item.cuda() for item in args)


def saveimage(names, png, file):
    try:
        fromarray(file.cpu().detach().numpy().astype(np.uint8).squeeze()).save(
            os.path.join(opts.result, names + '_.' + png))
    except:
        print('error' + names + png)
    pass


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    index = 0
    for items in data_loader:
        name = dataset.load_name(index)
        print(index)
        print(name)

        images, images_gray, edges, masks = tocuda(*items)
        index += 1

        gen_optimizer = optim.Adam(params=generator.parameters(), lr=float(0.0001), betas=(0.0, 0.9))
        gen_optimizer.zero_grad()
        gen_loss = 0

        edges_masked = (edges * (1 - masks))
        images_masked = (images_gray * (1 - masks)) + masks
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        print(inputs.shape)

        outputs = generator(inputs)

        outputs_merged = (outputs * masks) + (edges * (1 - masks))

        outputs_merged_temp = outputs_merged * 255.0
        outputs_merged_temp = outputs_merged_temp.permute(0, 2, 3, 1)

        fname, fext = name.split('.')
        fromarray(outputs_merged_temp[0].cpu().detach().numpy().astype(np.uint8).squeeze()).save(
            os.path.join(opts.result, fname + '_.' + fext))

        saveimage(fname + '_edges_masked', fext, edges_masked)
        saveimage(fname + '_images_masked', fext, images_masked)
        saveimage(fname + '_inputs', fext, inputs)
        saveimage(fname + '_outputs', fext, outputs)
        saveimage(fname + '_outputs_merged', fext, outputs_merged)
        saveimage(fname + '_outputs_merged_temp', fext, outputs_merged_temp)

    # if self.debug:
    #     edges = postprocess(1 - edges)[0]
    #     masked = postprocess(images * (1 - masks) + masks)[0]
    #     fname, fext = name.split('.')
    #
    #     imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
    #     imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

# print('\nEnd test....')

# print('Starting Test')
# with torch.no_grad():
#     generator.eval()
#
#     for _ in tqdm(range(opts.num_eval)):
#
#         ground_truth, gray_image, edge, mask = next(data_loader)
#
#         if is_cuda:
#             ground_truth, mask, edge, gray_image = ground_truth.cuda(), mask.cuda(), edge.cuda(), gray_image.cuda()
#
#         input_image, input_edge, input_gray_image = ground_truth * mask, edge * mask, gray_image * mask
#
#         output, _, _ = generator(input_image, torch.cat((input_image, input_gray_image), dim=1), mask)
#
#         output_comp = ground_truth * mask + output * (1 - mask)
#
#         output_comp = process(output_comp)
#
#         save_image(output_comp, opts.result, '/{:05d}.png'.format(_))
