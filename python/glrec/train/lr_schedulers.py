from abc import ABC, abstractmethod
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


class LRVisualizer(ABC):
    """The base class for learning rate visualization"""

    @abstractmethod
    def _lr_by_batchnum(self, steps_per_epoch, epochs):
        pass

    def visualize(self, steps_per_epoch, epochs):
        learning_rates = self._lr_by_batchnum(steps_per_epoch, epochs)
        plt.plot(np.arange(steps_per_epoch * epochs), learning_rates)
        plt.xlabel('batch number')
        plt.ylabel('learning rate')


class CyclicLR(Callback, LRVisualizer):
    """This callback implements a cyclical learning rate policy (CLR).

    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper
    (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half
        each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations)
        at each cycle iteration.
    For more detail, please see paper.

    This code was copied from: https://github.com/bckenstler/CLR
    Distributed under MIT license.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000.,
                 mode='triangular', gamma=1., scale_fn=None,
                 scale_mode='cycle'):
        super().__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self, iteration):
        cycle = np.floor(1+iteration/(2*self.step_size))
        x = np.abs(iteration/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + \
                (self.max_lr - self.base_lr) * \
                np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + \
                (self.max_lr - self.base_lr) * \
                np.maximum(0, (1-x))*self.scale_fn(iteration)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.clr(self.clr_iterations))

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(
                K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(
                self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr,
                    self.clr(self.clr_iterations))

    def _lr_by_batchnum(self, steps_per_epoch, epochs):
        ret = np.zeros(shape=(steps_per_epoch * epochs))
        for batch_num in range(steps_per_epoch * epochs):
            ret[batch_num] = self.clr(batch_num)
        return ret


class FineTuningLR(LearningRateScheduler, LRVisualizer):
    """
    Starts small (to not ruin delicate pre-trained weights),
    increases, then decays exponentially.

    # Arguments:
        lr_start: the initial learning rate
        lr_max: the peak learning rate
        lr_min: the lowest learning rate at the end
        lr_rampup_epochs: number of epochs before peak
        lr_sustain_epochs: number of epochs at the peak
        lr_exp_decay: exponential lr decay parameter
        verbose: int. 0: quiet, 1: update messages.
    """
    def __init__(self,
                 lr_start=0.00001,
                 lr_max=0.00005,
                 lr_min=0.00001,
                 lr_rampup_epochs=5,
                 lr_sustain_epochs=0,
                 lr_exp_decay=.8,
                 verbose=0):

        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_rampup_epochs = lr_rampup_epochs
        self.lr_sustain_epochs = lr_sustain_epochs
        self.lr_exp_decay = lr_exp_decay

        super().__init__(self.schedule, verbose)

    def schedule(self, epoch):
        if epoch < self.lr_rampup_epochs:
            lr = (self.lr_max - self.lr_start) / \
                    self.lr_rampup_epochs * epoch + self.lr_start
        elif epoch < self.lr_rampup_epochs + self.lr_sustain_epochs:
            lr = self.lr_max
        else:
            lr = (self.lr_max - self.lr_min) * self.lr_exp_decay ** \
                (epoch - self.lr_rampup_epochs - self.lr_sustain_epochs) + \
                self.lr_min
        return lr

    def _lr_by_batchnum(self, steps_per_epoch, epochs):
        ret = np.zeros(shape=(steps_per_epoch * epochs))
        for epoch in range(epochs):
            ret[epoch * steps_per_epoch:
                (epoch + 1) * steps_per_epoch] = self.schedule(epoch)
        return ret
