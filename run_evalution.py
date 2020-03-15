import json
import os
import time
from multiprocessing import Pool, Queue, Process
from pathlib import Path
import numpy as np
from pprint import pprint
from tqdm import tqdm
from functools import partial

# ----------------------------------------------------------------------------------------------------
basepath = Path("/root/dataset/abstraction-and-reasoning-challenge")
train_path = basepath / 'training'
evaluation_path = basepath / 'evaluation'
test_path = basepath / 'test'
prediction_path = basepath / 'prediction'
POOL_SIZE = 20
# ----------------------------------------------------------------------------------------------------


def read_json(task_id):
    path = prediction_path / task_id / ('%s.json' % task_id)
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def calk_score(task):
    return [int(np.equal(sample['output'], sample['prediction']).all()) for sample in task['test']]


def run(task_id):
    os.system("python3 arc_finetune.py %s" % task_id)
    # os.system("python3 model.py %s" % task_id)


def main(task_path):
    train_list = [x.stem for x in task_path.glob('*.json')]
    print("found %d tasks in %s" % (len(train_list), task_path))

    start_time = time.time()
    with Pool(POOL_SIZE) as p:
        p.map(run, train_list)

    print("use %d s for evaluation" % (time.time() - start_time))

    # train_list = [x.stem for x in prediction_path.glob("*")]
    train_list = [x for x in train_list if (prediction_path / x).exists()]
    tasks = [read_json(x) for x in train_list]
    scores = [calk_score(task) for task in tasks]

    solved = {
        task_id: score
        for task_id, score in zip(train_list, scores)
        if any(score)
    }

    print("solved %d tasks of %d tasks(img size is same)" % (len(solved), len(tasks)))
    pprint(solved)
    print([k for k in solved.keys()])


if __name__ == "__main__":
    main(train_path)
    # main(evaluation_path)
