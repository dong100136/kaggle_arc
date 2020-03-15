import json
import time
from argparse import ArgumentParser
from itertools import chain
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import (Compose, HorizontalFlip, Normalize,
                            RandomBrightness, RandomContrast, Resize,
                            ShiftScaleRotate)
from matplotlib import animation, colors, rc
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------
basepath = Path("/root/dataset/abstraction-and-reasoning-challenge")
train_path = basepath / 'training'
evaluation_path = basepath / 'evaluation'
test_path = basepath / 'test'
prediction_path = basepath / 'prediction'
# ----------------------------------------------------------------------------------------------------

device = torch.device("cuda:0")

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)


def read_task(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_pictures(pictures, labels):
    fig, axs = plt.subplots(1, len(pictures), figsize=(2 * len(pictures), 32))
    for i, (pict, label) in enumerate(zip(pictures, labels)):
        axs[i].imshow(np.array(pict), cmap=cmap, norm=norm)
        axs[i].set_title(label)
    plt.show()


def plot_sample(sample, predict=None):
    if predict is None:
        plot_pictures([sample['input'], sample['output']], ['Input', 'Output'])
    else:
        plot_pictures([sample['input'], sample['output'], predict], ['Input', 'Output', 'Predict'])


def plot_task(task, save_path=None):
    if 'prediction' in task['train'][0]:
        prediction = True
    ncols = 3 if prediction else 2
    titles = ['Input', 'Output', 'Predict'] if prediction else ['Input', 'Output']

    n_samples = len(task['train']) + len(task['test'])
    fig, axs = plt.subplots(n_samples, ncols, figsize=(2 * ncols, 2 * n_samples))

    idx = 0
    for i, sample in enumerate(chain(task['train'], task['test'])):
        axs[i][0].imshow(np.array(sample['input']), cmap=cmap, norm=norm)
        axs[i][0].set_title('input')
        if i >= len(task['train']):
            axs[i][0].set_title("test_input")
        axs[i][0].axis('off')

        axs[i][1].imshow(np.array(sample['output']), cmap=cmap, norm=norm)
        axs[i][1].set_title('output')
        if i >= len(task['train']):
            axs[i][1].set_title("test_ouput")
        axs[i][1].axis('off')

        if not prediction:
            continue

        axs[i][2].imshow(np.array(sample['prediction']), cmap=cmap, norm=norm)
        axs[i][2].set_title('prediction')
        if i >= len(task['train']):
            axs[i][2].set_title("test_prediction")
        axs[i][2].axis('off')

    if save_path:
        plt.savefig(save_path)


def inp2img(inp):
    inp = np.array(inp)
    img = np.full((10, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(10):
        img[i] = (inp == i)
    return img


def input_output_shape_is_same(task):
    return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task['train']])


def calk_score(task_test, predict):
    return [int(np.equal(sample['output'], pred).all()) for sample, pred in zip(task_test, predict)]


def show_task(task):
    for sample in task['train']:
        plot_sample(sample)


# ----------------------------------------------------------------------------------------------------
train_tasks = {x.stem: read_task(x) for x in train_path.glob("*.json")}
eval_tasks = {x.stem: read_task(x) for x in evaluation_path.glob("*.json")}
test_tasks = {x.stem: read_task(x) for x in test_path.glob("*.json")}

train_tasks.update(eval_tasks)

# print("found %d train tasks, %d valid tasks, %d test tasks" % (
#     len(train_tasks), len(eval_tasks), len(test_tasks)
# ))
# ----------------------------------------------------------------------------------------------------


class CAModel(nn.Module):
    def __init__(self, num_states):
        super(CAModel, self).__init__()

        self.feats = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_states, 64, kernel_size=3, dilation=i, padding=i),
                nn.ReLU()
            )
            for i in range(1, 4)
        ])

        self.conv1x1 = nn.Conv2d(192, num_states, kernel_size=1)

    def forward(self, x, steps=1):
        for _ in range(steps):
            x = torch.softmax(x, dim=1)
            x = [layer(x) for layer in self.feats]
            x = torch.cat(x, dim=1)
            x = self.conv1x1(x)
        return x


# class CAModel(nn.Module):
#     def __init__(self, num_states):
#         super(CAModel, self).__init__()

#         self.conv3x3 = nn.Sequential(
#             nn.Conv2d(num_states, 64, kernel_size=3, padding=1),
#             nn.ReLU()
#         )

#         self.feats = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(64, 64, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU()
#             )
#             for i in range(1, 2)
#         ])

#         self.conv1x1 = nn.Conv2d(64, num_states, kernel_size=1)

#     def forward(self, x, steps=1):
#         x = self.conv3x3(x)
#         for layer in self.feats:
#             x = layer(x)
#         x = self.conv1x1(x)
#         return x


def data_augmentation(task):
    new_task = []
    # origin

    def f(sample):
        return {
            'input': np.array(sample['input']),
            'output': np.array(sample['output'])
        }
    new_task.extend([f(sample) for sample in task])

    # flip
    def f(sample, t):
        return {
            'input': cv2.flip(np.array(sample['input']), t),
            'output': cv2.flip(np.array(sample['output']), t)
        }
    new_task.extend([f(sample, -1) for sample in task])
    new_task.extend([f(sample, 0) for sample in task])
    new_task.extend([f(sample, 1) for sample in task])

    print("data_augmentation from %d to %d data" % (len(task), len(new_task)))

    return new_task


def solve_task(task, max_steps=10, workspace=Path('/tmp')):
    model = CAModel(10).to(device)
    num_epochs = 100
    criterion = nn.CrossEntropyLoss()
    losses = np.zeros((max_steps - 1) * num_epochs)
    best_loss = 1000

    # task = data_augmentation(task)
    data_x = [inp2img(sample['input']) for sample in task]
    data_y = [np.array(sample['output']) for sample in task]
    data_y_in = [inp2img(sample['output']) for sample in task]

    for num_steps in range(1, max_steps):
        optimizer = torch.optim.Adam(model.parameters(), lr=(0.1 / (num_steps * 2)), weight_decay=1e-4)

        for e in range(num_epochs):
            optimizer.zero_grad()
            loss = 0.0

            for x, y, y_in in zip(data_x, data_y, data_y_in):
                # predict output from input
                x = torch.from_numpy(x).unsqueeze(0).float().to(device)
                y = torch.from_numpy(y).long().unsqueeze(0).to(device)
                y_pred = model(x, num_steps)
                loss += criterion(y_pred, y)

                # predit output from output
                # enforces stability after solution is reached
                y_in = torch.from_numpy(y_in).unsqueeze(0).float().to(device)
                y_pred = model(y_in, 1)
                loss += criterion(y_pred, y)

            loss.backward()
            optimizer.step()
            losses[(num_steps - 1) * num_epochs + e] = loss.item()
            if best_loss > loss.item():
                best_loss = loss.item()
                torch.save({'model': model.state_dict()}, workspace / 'best_model.pth')

    model.load_state_dict(torch.load(workspace / 'best_model.pth')['model'])

    return model, num_steps, losses


@torch.no_grad()
def predict(model, task):
    predictions = []
    for sample in task:
        x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)
        pred = model(x, 100).argmax(1).squeeze().cpu().numpy()
        predictions.append(pred)
    return predictions


def main(task_id, task):
    workspace = prediction_path / task_id
    workspace.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    model, num_steps, losses = solve_task(task['train'], workspace=workspace)
    print("use %d s for training of task %s" % ((time.time() - start_time), task_id))

    plt.plot(losses)
    plt.savefig(workspace / 'loss.png')

    # prediciton
    start_time = time.time()
    predictions = predict(model, task['train'])
    for i in range(len(task['train'])):
        task['train'][i]['prediction'] = predictions[i].tolist()

    predictions = predict(model, task['test'])
    for i in range(len(task['test'])):
        task['test'][i]['prediction'] = predictions[i].tolist()
    print("use %d s for prediction of task %s" % ((time.time() - start_time), task_id))

    result_json = workspace / ("%s.json" % task_id)
    with open(result_json, 'w') as f:
        json.dump(task, f)

    plot_task(task, workspace / "result.jpg")


if __name__ == '__main__':
    paser = ArgumentParser()
    paser.add_argument('id')
    args = paser.parse_args()

    task = train_tasks[args.id]

    if input_output_shape_is_same(task):
        main(args.id, task)
    else:
        print("skip task %s" % args.id)
