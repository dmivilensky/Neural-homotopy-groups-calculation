import torch
import numpy.linalg as lin
import numpy as np
from torchmetrics import Accuracy


class TrainMetric():
    def __init__(self, oracle):
        self.oracle = oracle
        self.values = []

    def __call__(self, batch):
        raise NotImplementedError

    def total(self):
        raise NotImplementedError


class WeightsNormTrainMetric(TrainMetric):
    def __init__(self, oracle, offset = 1000):
        super().__init__(oracle)
        self.offset = offset

    def __call__(self, batch):
        self.values.append(self.oracle.get_flat_params().detach().cpu().numpy().copy())
        
    def total(self):
        return list(map(
            lambda v: lin.norm(v - self.values[-1]), self.values[:-self.offset]
        ))


class GradientNormTrainMetric(TrainMetric):
    def __init__(self, oracle):
        super().__init__(oracle)

    def __call__(self, batch):
        self.values.append(
            lin.norm(self.oracle.gradient(batch).cpu().numpy())
        )

    def total(self):
        return self.values
        

class PerformanceMetric(TrainMetric):
    def __init__(self, oracle, metric):
        super().__init__(oracle)
        self.metric = metric

    def __call__(self, batch):
        X_, y_ = batch
        y_pred = self.oracle._model(X_)
        self.values.append(self.metric(y_pred.cpu().squeeze(), y_.cpu().squeeze()))

    def total(self):
        return self.values


class LossTrainMetric(TrainMetric):
    def __init__(self, oracle):
        super().__init__(oracle)

    def __call__(self, batch):
        self.values.append(self.oracle.loss_function_val(batch).cpu().detach().item())

    def total(self):
        return self.values
    