import argparse
import collections
from re import split

from torch.utils.data import sampler
from data.wider_face import WiderFaceDataset

from torch.optim import lr_scheduler
from models.model import MobileNetSeg
import torch
import numpy as np

# from data.data_loader import *
import data.data_loader as module_data

from trainer.trainer import Trainer
from utils import prepare_device
from config import *
import models.model_loss as module_loss
import models.metric as module_metric

# from torch.utils.data import DataLoader
from data.data_loader import *
from parse_config import ConfigParser

from torch.utils.data import random_split

# for debugging
from IPython import embed

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    all_train_data = WiderFaceDataset(split='train')

    split_ratio = VALIDATION_SPLIT
    valid_len = int(len(all_train_data) * split_ratio)
    train_len = len(all_train_data) - valid_len

    train_data, valid_data = random_split(all_train_data, [train_len, valid_len])

    mini_train_data, train_skip = random_split(train_data, [100, len(train_data) - 100])
    mini_valid_data, val_skip = random_split(valid_data, [50, len(valid_data) - 50])

    # train_data_loader = CustomDataLoader(dataset=train_data)
    # valid_data_loader = CustomDataLoader(dataset=valid_data)

    train_data_loader = CustomDataLoader(dataset=mini_train_data)
    valid_data_loader = CustomDataLoader(dataset=mini_valid_data)
   
    # print('train data:', len(train_data))

    # embed(header='Debugging')

    # default_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # print('default data loader:', type(default_data_loader))

    # train_data_loader = DataLoader(split='train', batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, shuffle=True)

    # train_sampler, val_sampler = train_data_loader.sampler, train_data_loader.valid_sampler

    # valid_data_loader = train_data_loader.split_validation()

    # train_data_loader = config.init_obj('train_data_loader', module_data)

    heads = {
            'hm': 1,
            'wh': 2,
            "hm_offset": 2,
            "landmarks": 10
        }
    
    model = MobileNetSeg(base_name='mobilenetv2_10', heads=heads)
    logger.info(model)

    model = model.to(DEVICE)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, 'center_face_loss')
    embed()
    # metrics = [getattr(module_metric, met) for met in ["accuracy", "top_k_acc"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    # optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    optimizer = torch.optim.Adam(trainable_params, lr=0.01, weight_decay=0, amsgrad=True)

    # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # trainer = Trainer(model, criterion, metrics, optimizer,
    #                   config=config,
    #                   device=DEVICE,
    #                   data_loader=train_data_loader,
    #                   valid_data_loader=valid_data_loader,
    #                   lr_scheduler=lr_scheduler)

    trainer = Trainer(model, criterion, optimizer,
                      config=config,
                      device=DEVICE,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    

if __name__ == '__main__':
  args = argparse.ArgumentParser(description='PyTorch Template')
  args.add_argument('-c', '--config', default="config.json", type=str,
                    help='config file path (default: None)')
  args.add_argument('-r', '--resume', default=None, type=str,
                    help='path to latest checkpoint (default: None)')
  args.add_argument('-d', '--device', default=DEVICE, type=str,
                    help='indices of GPUs to enable (default: all)')

  # custom cli options to modify configuration from default values given in json file.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer; args; lr'),
      CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader; args; batch_size')
  ]
  config = ConfigParser.from_args(args, options)
  main(config)




    




    


