import copy
import numpy as np
import torch
from fastai.torch_core import Tensor
from fastai.callbacks import TrackerCallback
from fastai.callback import Callback, add_metrics


class JigsawMetricOld(Callback):

    def on_epoch_begin(self, **kwargs):
        self.targs, self.preds = Tensor([]), Tensor([])

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):
        assert last_output.numel() == last_target.numel(), "Expected same numbers of elements in pred & targ"
        self.preds = torch.cat((self.preds, last_output.cpu()))
        self.targs = torch.cat((self.targs, last_target.cpu()))

    def on_epoch_end(self, last_metrics, **kwargs):
        pass



class JigsawMetric(Callback):
    '''
    example usage:

    evaluator = Evaluator(y_true = np.array(y_train_torch[val_index, 0])
                          ,  y_identity= np.array(identity[val_index]) )

    fm = Final_Metric()
    fm.set_eval(evaluator)

    learn = Learner(databunch
                    ,model
                    ,loss_func=custom_loss
                   , metrics = [fm])

    train_model(learn)
    '''

    def set_eval(self, evaluator):
        self.evaluator = evaluator

    def on_epoch_begin(self, **kwargs):
        self.predictions = []
        self.correct, self.total = 0., 0.

    def on_batch_end(self, last_output, last_target, **kwargs):
        preds = list(last_output[:, 0])
        self.predictions += preds

    def on_epoch_end(self, last_metrics, **kwargs):
        self.predictions = np.array(self.predictions)
        metric = self.evaluator.get_final_metric(self.predictions)
        self.predictions = []

        return add_metrics(last_metrics, metric)


class SWA(TrackerCallback):

    def __init__(self, learn):
        super().__init__(learn)
        self.model = learn.model
        self.swa_model = copy.deepcopy(learn.model)
        self.swa_n = 1

    def on_epoch_end(self, **kwargs):
        self.update_average_model()

    def update_average_model(self):
        # update running average of parameters
        model_params = self.model.parameters()
        swa_params = self.swa_model.parameters()
        for model_param, swa_param in zip(model_params, swa_params):
            swa_param.data *= self.swa_n
            swa_param.data += model_param.data
            swa_param.data /= (self.swa_n + 1)
