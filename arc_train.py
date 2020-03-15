import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from meta import Meta
from TaskDataSet import getTaskDataLoader
from learner import Learner

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
device = torch.device('cuda')

# ---------------------------------------------------------------------
num_states = 10

'''
conv2d: [c_out, c_in, k, k, stride, padding]
'''

model_config = [
    ('conv2d', [128, num_states, 3, 3, 1, 1]),
    ('relu', [True]),
    ('conv2d', [128, 128, 3, 3, 1, 1]),
    ('relu', [True]),
    ('conv2d', [128, 128, 3, 3, 1, 1]),
    ('relu', [True]),
    ('conv2d', [num_states, 128, 3, 3, 1, 1]),
]

base_path = Path('/root/dataset/abstraction-and-reasoning-challenge')
model_save_path = base_path / 'workspace' / 'model' / 'init_model.pth'
exp_config = {
    'train_path': base_path / "training",
    'valid_path': base_path / "evaluation",
    'test_path': base_path / "test",
    'epoch': 100,
    'meta_lr': 1e-3,
    'update_lr': 1e-2,
    'update_step': 5,
    'update_step_test': 10
}
# ---------------------------------------------------------------------


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def print_trainable_params(model):
    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(model)
    print('Total trainable tensors:', num)


def to_device(data):
    if isinstance(data, (list, tuple)):
        data = [to_device(x) for x in data]
    else:
        data = data.to(device)
    return data


def train_epoch(model, dataloader):
    # train epoch
    for step, data in enumerate(dataloader):
        data = to_device(data)
        accs, loss = model(data)

        # if step % 10 == 0:
        print('step:', step, '\ttraining acc:', accs, '\tloss:', loss)


def valid_epoch(model, dataloader):
    accs_all_test = []

    for data in dataloader:
        assert len(data) == 1

        data = to_device(data)
        accs = model.finetunning(data[0])
        accs_all_test.append(accs)

    # [b, update_step+1]
    accs = np.array(accs_all_test).sum()
    print('Test acc: %d/%d' % (accs, len(accs_all_test)))


def save_model(model):
    save_state_dict = {
        'model': model.state_dict()
    }
    torch.save(save_state_dict, model_save_path)


def load_model(model):
    if model_save_path.exists():
        print("load pretrain model")
        save_state_dict = torch.load(model_save_path)
        model.load_state_dict(save_state_dict['model'])
    return model


def main():
    model = Learner(model_config)
    model = load_model(model)
    maml = Meta(exp_config, model).to(device)
    print_trainable_params(maml)

    train_task_dataloader = getTaskDataLoader(exp_config['train_path'], batch_size=10)
    valid_task_dataloader = getTaskDataLoader(exp_config['valid_path'], batch_size=1)

    for epoch in range(exp_config['epoch']):
        print("---------------- epoch %d ----------------" % epoch)
        train_epoch(maml, train_task_dataloader)
        valid_epoch(maml, valid_task_dataloader)

        save_model(model)


if __name__ == '__main__':
    main()
