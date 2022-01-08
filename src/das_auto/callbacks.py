# modified from https://gist.github.com/pippo111/911167d9574250b3e799b4b71bf6cbe8
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

class ReinitWeightOnFalseStart(Callback):
    """Reinitializes model weights when model is not improving right after beginning.

    Stops training after number of trials.

    If model starts improving then behave like
    EarlyStopping callback and stops training
    after number of epochs with no improvement.

    Monitored value is 'val_loss'.

    # Arguments
        patience: number of epochs with no improvement
            after which weights will be reinitialized.
        trials: number of attempts to reinitialize weights
            after which training will be stopped.
        checks: number of epochs with no improvement after
            initial starts with promising weights
            after which training will be stopped.
        verbose: verbosity mode.
    """
    def __init__(self,
                 patience=3,
                 trials=1,
                 checks=10,
                 verbose=0):
        self.patience = patience
        self.trials = trials
        self.checks = checks
        self.verbose = verbose

    def reset_weights(self):
        session = K.get_session()
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)

    def on_train_begin(self, logs={}):
        self._monitor = True
        self._restarted = True
        self._best = np.Inf
        self._alltime_best = np.Inf
        self._wait = 0
        self._tries = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get('val_loss')

        print('tries: ', self._tries, ', wait: ', self._wait)
        print('current: ', current, ', best: ', self._best)
        print('alltime best', self._alltime_best)

        if self._monitor:
            if current >= self._best:
                self._wait += 1

                if self._wait == self.patience:
                    if self._tries == self.trials:
                        if self.verbose > 0:
                            print('\nReinitWeightOnFalseStart: Model is not improving from the start. Stopping.')
                        self.model.stop_training = True
                        return

                    if self.verbose > 0:
                        print('\nReinitWeightOnFalseStart: Reinitializing model weights.')
                    self.reset_weights()
                    self._restarted = True
                    self._wait = 0
                    self._best = np.Inf
                    self._tries += 1

            else:
                self._alltime_best = current

                if self._restarted:
                    self._best = current
                    self._restarted = False
                else:
                    if self.verbose > 0:
                        print('\nReinitWeightOnFalseStart: Looks like model is improving. Stop monitoring.')
                    self._monitor = False
                    self._wait = 0

        else:
            if current >= self._alltime_best:
                self._wait += 1

                if self._wait > self.checks:
                    if self.verbose > 0:
                        print('\nReinitWeightOnFalseStart: Model is not improving from the start. Stopping.')
                    self.model.stop_training = True
                    return
            else:
                self._alltime_best = current
                self._wait = 0