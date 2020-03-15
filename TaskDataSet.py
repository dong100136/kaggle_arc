import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------------------------------------
basepath = Path("/root/dataset/abstraction-and-reasoning-challenge")
train_path = basepath / 'training'
evaluation_path = basepath / 'evaluation'
test_path = basepath / 'test'
prediction_path = basepath / 'prediction'
# ----------------------------------------------------------------------------------------------------


def getTaskDataLoader(tasks_path, batch_size=10):
    """
    Arguments:
        tasks_path {str} -- [the dir of tasks]

    Returns:
        return Nx(spt_x,spt_y,qry_x,qry_y) tensor
    """
    return DataLoader(
        TaskDataSet(tasks_path),
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=lambda x: x,
        pin_memory=False,
        drop_last=False)


class TaskDataSet(Dataset):
    def __init__(self, tasks_path):
        self.tasks_path = tasks_path
        self.tasks_list = [x.stem for x in tasks_path.glob('*.json')]
        np.random.shuffle(self.tasks_list)

        self.tasks = [self.read_json(x) for x in self.tasks_list]
        self.tasks = [x for x in self.tasks if self.input_output_shape_is_same(x)]
        print("got %d tasks(same size for input and output)" % (len(self.tasks)))

    def read_json(self, task_id):
        path = self.tasks_path / ('%s.json' % task_id)
        with open(path, 'r') as f:
            task = json.load(f)
        return task

    def input_output_shape_is_same(self, task):
        return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task['train']])

    def inp2img(self, inp):
        inp = np.array(inp)
        img = np.full((10, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
        for i in range(10):
            img[i] = (inp == i)
        return img

    def __getitem__(self, index):
        task = self.tasks[index]
        spt_x = [torch.tensor(self.inp2img(sample['input'])).float().unsqueeze(0) for sample in task['train']]
        spt_y = [torch.tensor(np.array(sample['output'])).long().unsqueeze(0) for sample in task['train']]

        qry_x = [torch.tensor(self.inp2img(sample['input'])).float().unsqueeze(0) for sample in task['test']]
        qry_y = [torch.tensor(np.array(sample['output'])).long().unsqueeze(0) for sample in task['test']]

        return spt_x, spt_y, qry_x, qry_y

    def __len__(self):
        return len(self.tasks)


if __name__ == "__main__":
    data = getTaskDataLoader(train_path)
    x = iter(data).next()
    print(x[0][0])
