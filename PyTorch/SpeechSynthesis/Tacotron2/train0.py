import os
import time
import argparse
import numpy as np
from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from apex.parallel import DistributedDataParallel as DDP

import models
import loss_functions
import data_functions

from dllogger.logger import LOGGER
import dllogger.logger as dllg
from dllogger import tags
from dllogger.autologging import log_hardware, log_args
from scipy.io.wavfile import write as write_wav

from apex import amp
amp.lists.functional_overrides.FP32_FUNCS.remove('softmax')
amp.lists.functional_overrides.FP16_FUNCS.append('softmax')

def main():


    #DataLoader 객체 생성
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=args.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)