import argparse
import os
import random
from shutil import copyfile

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.celeba_dataset import CelebADataset
from src.config import Config
from src.dataset import Dataset
from src.metrics import PSNR, EdgeAccuracy
from src.models import EdgeModel, InpaintingModel, RefineModel
from src.utils import getVGGModel, Progbar, stitch_images, create_dir


def load_config(mode=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], default=4)

    # test mode
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
        parser.add_argument('--output', type=str, help='path to the output directory')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')
    print(config_path)

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    configmain = Config(config_path)

    # train mode
    if mode == 1:
        configmain.MODE = 1
        if args.model:
            configmain.MODEL = args.model

    # test mode
    elif mode == 2:
        configmain.MODE = 2
        configmain.MODEL = args.model if args.model is not None else 3
        configmain.INPUT_SIZE = 0

        if args.input is not None:
            configmain.TEST_FLIST = args.input

        if args.mask is not None:
            configmain.TEST_MASK_FLIST = args.mask

        if args.edge is not None:
            configmain.TEST_EDGE_FLIST = args.edge

        if args.output is not None:
            configmain.RESULTS = args.output

    # eval mode
    elif mode == 3:
        configmain.MODE = 3
        configmain.MODEL = args.model if args.model is not None else 3

    return configmain


# Load Pre values
config = load_config(1)

# Check for GPU/CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
if torch.cuda.is_available():
    config.DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
else:
    config.DEVICE = torch.device("cpu")

# print(device_lib.list_local_devices())
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("Num XLA_GPUs Available: ", len(tf.config.list_physical_devices('XLA_GPU')))
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# print(tf.test.is_gpu_available())
# print(tf.test.gpu_device_name())


# set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
cv2.setNumThreads(0)

# initialize random seed
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
np.random.seed(config.SEED)
random.seed(config.SEED)

# get VGG for perceptual and style loss
vgg_model, selected_layers = getVGGModel()

# tf.debugging.set_log_device_placement(True)

# build the models and initialize
edge_model = EdgeModel(config).to(config.DEVICE)
inpaint_model = InpaintingModel(config).to(config.DEVICE)
refine_model = RefineModel(config).to(config.DEVICE)

# metrics
psnr = PSNR(255.0).to(config.DEVICE)
edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

# dataset train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST,
# augment=True, training=True)

train_dataset = CelebADataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST,
                              augment=True,
                              training=True)

val_dataset = CelebADataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, augment=False,
                            training=True)

pin_memory = True if config.DEVICE == 'cuda' else False


def create_iterator(dataset, batch_size):
    while True:
        sample_loader = DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, pin_memory=pin_memory)
        for item in sample_loader:
            yield item


sample_iterator = create_iterator(val_dataset, config.SAMPLE_SIZE)

# create path
samples_path = os.path.join(config.PATH, 'samples')
results_path = os.path.join(config.PATH, 'results')
if config.RESULTS is not None:
    results_path = os.path.join(config.RESULTS)

# create log file
log_file = os.path.join(config.PATH, 'log_' + 'model_name' + '.dat')

# load models
edge_model.load()
inpaint_model.load()
refine_model.load()


# assigning each item to GPU
def cuda(*args):
    return (item.to(config.DEVICE) for item in args)


# post process images
def postprocess(img):
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()


def sample(it=None):
    # do not sample when validation set is empty
    if len(val_dataset) == 0:
        return

    edge_model.eval()
    inpaint_model.eval()
    refine_model.eval()

    items_sample = next(sample_iterator)
    images_sample, images_gray_sample, edges_sample, masks_sample = cuda(*items_sample)

    iteration_sample = inpaint_model.iteration

    inputs_sample = (images_sample * (1 - masks_sample)) + masks_sample
    outputs_sample = edge_model(images_gray_sample, edges_sample, masks_sample).detach()

    edges_sample = (outputs_sample * masks_sample + edges_sample * (1 - masks_sample)).detach()
    outputs_sample = inpaint_model(images_sample, edges_sample, masks_sample)

    outputs_merged_sample = (outputs_sample * masks_sample) + (images_sample * (1 - masks_sample))
    outputs_sample = refine_model(inputs_sample, masks_sample, outputs_merged_sample)

    if it is not None:
        iteration_sample = it

    image_per_row = 2
    if config.SAMPLE_SIZE <= 6:
        image_per_row = 1

    images = stitch_images(
        postprocess(images_sample),
        postprocess(inputs_sample),
        postprocess(edges_sample),
        postprocess(outputs_sample),
        postprocess(outputs_merged_sample),
        img_per_row=image_per_row
    )

    path_sample = os.path.join(samples_path, 'model_name')
    name_sample = os.path.join(path_sample, str(iteration_sample).zfill(5) + ".png")
    create_dir(path_sample)
    print('\nsaving sample ' + name_sample)
    images.save(name_sample)


def log(logss):
    with open(log_file, 'a') as f:
        f.write('%s\n' % ' '.join([str(item[1]) for item in logss]))


def eval():
    print('\nstart eval...\n')
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE, drop_last=True, shuffle=True)

    total_eval = len(val_dataset)

    edge_model.eval()
    inpaint_model.eval()
    refine_model.eval()

    progbar_eval = Progbar(total_eval, width=20, stateful_metrics=['it'])
    iteration_eval = 0

    for items_eval in val_loader:
        iteration_eval += 1
        images_eval, images_gray_eval, edges_eval, masks_eval = cuda(*items_eval)

        # joint model
        e_outputs_eval, e_gen_loss_eval, e_dis_loss_eval, e_logs_eval = edge_model.process(images_gray_eval, edges_eval,
                                                                                           masks_eval)
        e_outputs_eval = e_outputs_eval * masks_eval + edges_eval * (1 - masks_eval)
        i_outputs_eval, i_gen_loss_eval, i_dis_loss_eval, i_logs_eval = inpaint_model.process(images_eval,
                                                                                              e_outputs_eval,
                                                                                              masks_eval)
        outputs_merged_eval = (i_outputs_eval * masks_eval) + (images_eval * (1 - masks_eval))
        r_outputs_eval, r_gen_loss_eval, r_dis_loss_eval, r_logs_eval = refine_model.process(images_eval, masks_eval,
                                                                                             outputs_merged_eval,
                                                                                             vgg_model, selected_layers)
        outputs_merged_eval = (r_outputs_eval * masks_eval) + (images_eval * (1 - masks_eval))

        # metrics
        psnr_eval = psnr(postprocess(images_eval), postprocess(outputs_merged_eval))
        mae_eval = (torch.sum(torch.abs(images_eval - outputs_merged_eval)) / torch.sum(images_eval)).float()
        precision_eval, recall_eval = edgeacc(edges_eval * masks_eval, e_outputs_eval * masks_eval)
        e_logs_eval.append(('pre_eval', precision_eval.item()))
        e_logs_eval.append(('rec_eval', recall_eval.item()))
        i_logs_eval.append(('psnr_eval', psnr_eval.item()))
        i_logs_eval.append(('mae_eval', mae_eval.item()))
        logs_eval = e_logs_eval + i_logs_eval

        logs_eval = [("it", iteration_eval), ] + logs_eval
        progbar_eval.add(len(images_eval), values=logs_eval)


def save():
    edge_model.save()
    inpaint_model.save()
    refine_model.save()


def main():
    print('\nstart training...\n')

    num_workers = 0 if config.DEVICE == 'cuda' else 2  # 4

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, num_workers=num_workers,
                              drop_last=True,
                              shuffle=True, pin_memory=pin_memory)

    epoch = 0
    keep_training = True
    max_iteration = 20  # int(float(config.MAX_ITERS))
    total = len(train_dataset)

    print(max_iteration)
    print(config.BATCH_SIZE)
    print(config.DEVICE)
    print(total)

    if total == 0:
        print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
        return
    else:
        while keep_training:

            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:

                print("inside Training Loader ")

                edge_model.train()
                print("after   edge_model.train")

                inpaint_model.train()
                print("after   inpaint_model.train")

                refine_model.train()
                print("after   refine_model.train")

                images, images_gray, edges, masks = cuda(*items)
                print("after   cuda(*items)")

                # train
                e_outputs, e_gen_loss, e_dis_loss, e_logs = edge_model.process(images_gray, edges, masks)
                e_outputs = e_outputs * masks + edges * (1 - masks)
                print('after e_outputs')

                i_outputs, i_gen_loss, i_dis_loss, i_logs = inpaint_model.process(images, e_outputs, masks)
                outputs_merged = (i_outputs * masks) + (images * (1 - masks))
                print('after i_outputs')

                r_outputs, r_gen_loss, r_dis_loss, r_logs = refine_model.process(images, masks, outputs_merged,
                                                                                 vgg_model, selected_layers)
                r_output_merged = r_outputs  # ( * masks) + (images * (1 - masks))
                print('after r_outputs')

                # metrics
                psnr1 = psnr(postprocess(images), postprocess(r_output_merged))
                mae = (torch.sum(torch.abs(images - r_output_merged)) / torch.sum(images)).float()
                precision, recall = edgeacc(edges * masks, e_outputs * masks)
                e_logs.append(('pre', precision.item()))
                e_logs.append(('rec', recall.item()))
                i_logs.append(('psnr', psnr1.item()))
                i_logs.append(('mae', mae.item()))
                logs = e_logs + i_logs + r_logs
                print(logs)

                # backward
                refine_model.backward(r_gen_loss, r_dis_loss)
                print('after backward refine')
                inpaint_model.backward(i_gen_loss, i_dis_loss)
                print('after backward inpaint')
                edge_model.backward(e_gen_loss, e_dis_loss)
                print('after backward edge')

                iteration = inpaint_model.iteration

                print(iteration)

                if iteration >= max_iteration:
                    keep_training = False
                    print('iteration done')
                    break

                logs = [
                           ("epoch", epoch),
                           ("iter", iteration),
                       ] + logs

                progbar.add(len(images),
                            values=logs if config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if config.LOG_INTERVAL and iteration % config.LOG_INTERVAL == 0:
                    print('log(logs)')
                    log(logs)

                # sample model at checkpoints
                if config.SAMPLE_INTERVAL and iteration % config.SAMPLE_INTERVAL == 0:
                    print('sample()')
                    sample()

                # evaluate model at checkpoints
                if config.EVAL_INTERVAL and iteration % config.EVAL_INTERVAL == 0:
                    print('eval()')
                    eval()

                # save model at checkpoints
                if config.SAVE_INTERVAL and iteration % config.SAVE_INTERVAL == 0:
                    print('save()')
                    save()

        print('\nEnd training....')


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()
