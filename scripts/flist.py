import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset',
                    default='F:/Amrita/edge-connect/msks/')
parser.add_argument('--output', type=str, help='path to the file list', default='F:/Amrita/edge-connect/datasets/masks_train.flist')
args = parser.parse_args()

#if not os.path.exists(args.output):
#    os.makedirs(args.output)

ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}

images = []
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1].upper() in ext:
            images.append(os.path.join(root, file))

images = sorted(images)
np.savetxt(args.output, images, fmt='%s')
