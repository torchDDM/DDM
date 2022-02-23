import ipdb
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
from math import *
import time
from util.visualizer import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/DDM_train.json',
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

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    batchSize = opt['datasets']['train']['batch_size']
    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        train_set = Data.create_dataset_cardiac(dataset_opt, phase)
        train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        training_iters = int(ceil(train_set.data_len / float(batchSize)))
    logger.info('Initial Training Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    #### MOdel training ####
    while current_epoch < n_epoch:
        current_epoch += 1
        for istep, train_data in enumerate(train_loader):
            iter_start_time = time.time()
            current_step += 1

            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            # log
            if (istep+1) % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                t = (time.time() - iter_start_time) / batchSize
                visualizer.print_current_errors(current_epoch, istep+1, training_iters, logs, t, 'Train')
                visualizer.plot_current_errors(current_epoch, (istep+1) / float(training_iters), logs)

            # validation
            if (istep+1) % opt['train']['val_freq'] == 0:
                result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                os.makedirs(result_path, exist_ok=True)

                diffusion.test(continous=False)
                visuals = diffusion.get_current_visuals()
                visualizer.display_current_results(visuals, current_epoch, True)

        if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            logger.info('Saving models and training states.')
            diffusion.save_network(current_epoch, current_step)

    # save model
    logger.info('End of training.')