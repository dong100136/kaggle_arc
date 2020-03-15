import json
import time
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation, colors, rc
from torch import nn

from arc_train import model_config
from learner import Learner
from itertools import chain

# ------------------------------------------------------
fintune_epoch = 200
fintune_lr = 1e-4
weight_decay = 1e-5

basepath = Path('/root/dataset/abstraction-and-reasoning-challenge')
model_save_path = basepath / 'workspace' / 'model' / 'init_model.pth'

train_path = basepath / 'training'
evaluation_path = basepath / 'evaluation'
test_path = basepath / 'test'
prediction_path = basepath / 'prediction'
# ------------------------------------------------------
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
# train_tasks = {x.stem: read_task(x) for x in train_path.glob("*.json")}
# eval_tasks = {x.stem: read_task(x) for x in evaluation_path.glob("*.json")}
# test_tasks = {x.stem: read_task(x) for x in test_path.glob("*.json")}


# print("found %d train tasks, %d valid tasks, %d test tasks" % (
#     len(train_tasks), len(eval_tasks), len(test_tasks)
# ))
# ----------------------------------------------------------------------------------------------------


def load_model(model, verbose=1):
    if model_save_path.exists():
        if verbose > 0:
            print("load pretrain model")
        save_state_dict = torch.load(model_save_path)
        model.load_state_dict(save_state_dict['model'])
    return model


def solve_task(task, max_steps=10, workspace=Path('/tmp'), verbose=1):
    model = Learner(model_config)
    # model = load_model(model, verbose)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    losses = np.zeros(fintune_epoch)
    optimizer = torch.optim.Adam(model.parameters(), lr=fintune_lr, weight_decay=weight_decay)
    best_loss = 1000

    # task = data_augmentation(task)
    data_x = [torch.Tensor(inp2img(sample['input'])).unsqueeze(0).float().to(device) for sample in task]
    data_y = [torch.Tensor(np.array(sample['output'])).long().unsqueeze(0).to(device) for sample in task]

    for cur_epoch in range(fintune_epoch):
        loss = 0.0

        for x, y in zip(data_x, data_y):
            # predict output from input
            y_pred = model(x)
            loss += criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses[cur_epoch] = loss.item()
        if best_loss > loss.item():
            best_loss = loss.item()
            torch.save({'model': model.state_dict()}, workspace / 'best_model.pth')

    model.load_state_dict(torch.load(workspace / 'best_model.pth')['model'])

    return model, losses


@torch.no_grad()
def predict(model, task):
    predictions = []
    for sample in task:
        x = torch.from_numpy(inp2img(sample["input"])).unsqueeze(0).float().to(device)
        pred = model(x).argmax(1).squeeze().cpu().numpy()
        predictions.append(pred)
    return predictions


def main(task_id, task, verbose=1):
    workspace = prediction_path / task_id
    workspace.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    model, losses = solve_task(task['train'], workspace=workspace, verbose=verbose)
    if verbose > 0:
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

    if verbose > 0:
        print("use %d s for prediction of task %s" % ((time.time() - start_time), task_id))

    result_json = workspace / ("%s.json" % task_id)
    with open(result_json, 'w') as f:
        json.dump(task, f)

    plot_task(task, workspace / "result.jpg")


def load_tasks(task_id):
    path = (train_path / ('%s.json' % task_id))
    if path.exists():
        return read_task(path)

    path = (evaluation_path / ('%s.json' % task_id))
    if path.exists():
        return read_task(path)

    raise Exception("%s task is not exists" % task_id)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('id')
    args = parser.parse_args()

    # task = tasks[args.id]
    task = load_tasks(args.id)

    if input_output_shape_is_same(task):
        main(args.id, task, verbose=False)
    else:
        print("skip task %s" % args.id)
