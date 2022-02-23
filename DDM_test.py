import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
import numpy as np
import time
from util.visualizer import Visualizer
import torch.nn.functional as F
from PIL import Image

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/DDM_test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training) or test(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    visualizer = Visualizer(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        test_set = Data.create_dataset_cardiac(dataset_opt, phase)
        test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    logger.info('Initial Test Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    ##### Model Inference ####
    registDice = np.zeros((test_set.data_len, 5))
    originDice = np.zeros((test_set.data_len, 5))
    registTime = []
    logger.info('Begin Model Evaluation.')
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    for istep,  val_data in enumerate(test_loader):
        idx += 1
        dataName = val_data['P'][0].split('/')[-1][:-4]
        print('Test Data: %s' % dataName)

        time1 = time.time()
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        time2 = time.time()

        visuals = diffusion.get_current_data()
        defm_frames = visuals['deform'].squeeze().numpy().transpose(0, 2, 3, 1)[1:]
        code_frames = visuals['code'].squeeze().numpy().transpose(0, 2, 3, 1)[1:]
        flow_frames = visuals['field'].squeeze().numpy().transpose(0, 3, 4, 2, 1)[1:]
        flow_frames_ES = flow_frames[-1]
        sflow = torch.from_numpy(flow_frames_ES.transpose(3, 2, 0, 1).copy()).unsqueeze(0)
        sflow = Metrics.transform_grid(sflow[:, 0], sflow[:, 1], sflow[:, 2])
        nb, nc, nd, nh, nw = sflow.shape
        segflow = torch.FloatTensor(sflow.shape).zero_()
        segflow[:, 2] = (sflow[:, 0] / (nd - 1) - 0.5) * 2.0  # D[0 -> 2]
        segflow[:, 1] = (sflow[:, 1] / (nh - 1) - 0.5) * 2.0  # H[1 -> 1]
        segflow[:, 0] = (sflow[:, 2] / (nw - 1) - 0.5) * 2.0  # W[2 -> 0]
        origin_seg = val_data['MS'].squeeze()
        origin_seg = origin_seg.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        regist_seg = F.grid_sample(origin_seg.cuda().float(), (segflow.cuda().float().permute(0, 2, 3, 4, 1)), mode='nearest')
        regist_seg = regist_seg.squeeze().cpu().numpy().transpose(1, 2, 0)
        label_seg = val_data['FS'][0].cpu().numpy()
        origin_seg = val_data['MS'][0].cpu().numpy()

        vals_regist = Metrics.dice_ACDC(regist_seg, label_seg)[::3]
        vals_origin = Metrics.dice_ACDC(origin_seg, label_seg)[::3]
        registDice[istep] = vals_regist
        originDice[istep] = vals_origin
        print('---- Original Dice: %03f | Deformed Dice: %03f' %(np.mean(vals_origin), np.mean(vals_regist)))

        data_origin = val_data['M'].squeeze().cpu().numpy().transpose(1, 2, 0)
        data_fixed = val_data['F'].squeeze().cpu().numpy().transpose(1, 2, 0)
        label_origin = val_data['MS'].squeeze().cpu().numpy()
        label_fixed = val_data['FS'].squeeze().cpu().numpy()

        dpt = 12
        savePath = os.path.join(result_path, '%s_mov.png' % (dataName))
        save_image((data_origin[:,:,dpt]+1)/2.*255, savePath)
        savePath = os.path.join(result_path, '%s_fix.png' % (dataName))
        save_image((data_fixed[:,:,dpt]+1)/2.*255, savePath)
        code_frames -= code_frames[-1][:,:,dpt].min()
        code_frames /= code_frames[-1][:,:,dpt].max()
        for iframe in range(code_frames.shape[0]):
            savePath = os.path.join(result_path, '%s_frame%d.png' % (dataName, iframe+1))
            save_image(defm_frames[iframe][:,:,dpt]*255, savePath)
            savePath = os.path.join(result_path, '%s_code%d.png' % (dataName, iframe+1))
            save_image(code_frames[iframe][:, :, dpt]*255, savePath)
        registTime.append(time2 - time1)

    omdice, osdice = np.mean(originDice), np.std(originDice)
    mdice, sdice = np.mean(registDice), np.std(registDice)
    mtime, stime = np.mean(registTime), np.std(registTime)

    print()
    print('---------------------------------------------')
    print('Total Dice and Time Metrics------------------')
    print('---------------------------------------------')
    print('origin Dice | mean = %.3f, std= %.3f' % (omdice, osdice))
    print('Deform Dice | mean = %.3f, std= %.3f' % (mdice, sdice))
    print('Deform Time | mean = %.3f, std= %.3f' % (mtime, stime))