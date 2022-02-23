import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')
from . import metrics as Metrics


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')

        # set loss and load resume state
        self.set_loss()
        if self.opt['phase'] == 'train':
            self.loss_lambda = opt['train']['loss_lambda']
            self.centered = opt['datasets']['train']['centered']

        self.load_network()
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"], betas=(0.5, 0.999))
            self.log_dict = OrderedDict()
        self.print_network(self.netG)

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        score, loss = self.netG(self.data, self.loss_lambda)
        self.score, self.pdata = score

        l_pix, l_sim, l_smt, l_tot = loss
        l_tot.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()
        self.log_dict['l_sim'] = l_sim.item()
        self.log_dict['l_smt'] = l_smt.item()
        self.log_dict['l_tot'] = l_tot.item()

    def test(self, continous=False):
        self.netG.eval()
        input = torch.cat([self.data['M'], self.data['F']], dim=1)
        nsample = self.data['nS']
        if isinstance(self.netG, nn.DataParallel):
            self.code, self.deform, self.field = self.netG.module.ddm_inference(input, nsample, continous)
        else:
            self.code, self.deform, self.field = self.netG.ddm_inference(input, nsample, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        if self.centered:
            min_max = (-1, 1)
        else:
            min_max = (0, 1)
        out_dict['code'] = Metrics.tensor2im(self.code.detach().float().cpu(), min_max=min_max)
        out_dict['S'] = Metrics.tensor2im(self.data['M'].detach().float().cpu(), min_max=min_max)
        out_dict['T'] = Metrics.tensor2im(self.data['F'].detach().float().cpu(), min_max=min_max)
        out_dict['deform'] = Metrics.tensor2im(self.deform.detach().float().cpu(), min_max=min_max) #(0, 1)
        out_dict['field'] = Metrics.tensor2im(self.field.detach().float().cpu(), min_max=min_max)
        return out_dict

    def get_current_data(self):
        out_dict = OrderedDict()
        out_dict['code'] = self.code.detach().float().cpu()
        out_dict['deform'] =self.deform.detach().float().cpu()
        out_dict['field'] = self.field.detach().float().cpu()
        return out_dict

    def print_network(self, net):
        s, n = self.get_network_description(net)
        if isinstance(net, nn.DataParallel):
            net_struc_str = '{} - {}'.format(net.__class__.__name__, net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(net.__class__.__name__)

        logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        genG_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen_G.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, genG_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)
        logger.info(
            'Saved model in [{:s}] ...'.format(genG_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            genG_path = '{}_gen_G.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                genG_path), strict=(not self.opt['model']['finetune_norm']))

            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
