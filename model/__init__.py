import logging
logger = logging.getLogger('base')


def create_model(opt):
    if opt['path']['resume_state_D'] is None:
        from .model import DDPM as M
        m = M(opt)
        logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    else:
        from .model_2step import DDPM as M
        m = M(opt)
        logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
