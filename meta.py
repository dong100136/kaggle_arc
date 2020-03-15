import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np

from learner import Learner
from copy import deepcopy


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, exp_config, model, criterion=F.cross_entropy):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = exp_config['update_lr']
        self.meta_lr = exp_config['meta_lr']
        self.update_step = exp_config['update_step']
        self.update_step_test = exp_config['update_step_test']
        self.criterion = criterion

        self.net = model
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def calk_score(self, y_logits, y):
        with torch.no_grad():
            y_pred = F.softmax(y_logits, dim=1).argmax(dim=1)
            correct = torch.eq(y_pred, y).all().int().item()

        return correct

    def run(self, model, samples_x, samples_y, params=None, bn_training=True):
        n_sample = len(samples_x)
        loss = 0
        correct = []
        for x, y in zip(samples_x, samples_y):
            logits = model(x, vars=params, bn_training=bn_training)
            loss += self.criterion(logits, y)
            correct.append(self.calk_score(logits, y))

        return loss / n_sample, sum(correct)

    def forward(self, tasks):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num = len(tasks)
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):
            x_spt, y_spt, x_qry, y_qry = tasks[i]

            n_sample = len(x_spt)

            # 1. run the i-th task and compute loss for k=0
            loss, correct = self.run(self.net, x_spt, y_spt, params=None, bn_training=True)
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                loss_q, correct = self.run(self.net, x_qry, y_qry, params=self.net.parameters(), bn_training=True)
                losses_q[0] += loss_q
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                loss_q, correct = self.run(self.net, x_qry, y_qry, params=fast_weights, bn_training=True)
                losses_q[1] += loss_q
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                loss, correct = self.run(self.net, x_spt, y_spt, params=fast_weights, bn_training=True)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                loss, correct = self.run(self.net, x_qry, y_qry, fast_weights, bn_training=True)
                losses_q[k + 1] += loss
                with torch.no_grad():
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()

        accs = np.array(corrects)
        return accs, loss_q.item()

    def finetunning(self, task):
        assert len(task) == 4
        x_spt, y_spt, x_qry, y_qry = task

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        loss, correct = self.run(net, x_spt, y_spt, params=None, bn_training=True)
        grad = torch.autograd.grad(loss, net.parameters(), allow_unused=True)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            logits_q, correct = self.run(net, x_qry, y_qry, net.parameters(), bn_training=True)
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            logits_q, correct = self.run(net, x_qry, y_qry, fast_weights, bn_training=True)
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            loss, correct = self.run(net, x_spt, y_spt, fast_weights, bn_training=True)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q, correct = self.run(net, x_qry, y_qry, fast_weights, bn_training=True)

            with torch.no_grad():
                corrects[k + 1] = corrects[k + 1] + correct

        del net

        accs = np.array(corrects)

        return accs


def main():
    pass


if __name__ == '__main__':
    main()
