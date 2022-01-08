import numpy as np
import time
import matplotlib.pyplot as plt
import flammkuchen
import das.utils
import das.augmentation
import das.io, das.data, das.evaluate, das.tracking, das.tcn, das.kapre
from typing import List, Optional, Tuple, Dict, Any
import logging
import das_auto.models, das_auto.data, das_auto.layer_utils
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import defopt


def train(*,
          data_dir: str,
          x_suffix: str = '',
          y_suffix: str = '',
          save_dir: str = './',
          save_prefix: Optional[str] = None,
          save_name: Optional[str] = None,
          model_name: str = 'tcn',
          nb_filters: int = 16,
          kernel_size: int = 16,
          nb_conv: int = 3,
          use_separable: List[bool] = False,
          nb_hist: int = 1024,
          ignore_boundaries: bool = True,
          batch_norm: bool = True,
          nb_pre_conv: int = 0,
          pre_nb_dft: int = 64,
          pre_kernel_size: int = 3,
          pre_nb_filters: int = 16,
          pre_nb_conv: int = 2,
          upsample: bool = True,
          nb_lstm_units: int = 0,
          verbose: int = 1,
          batch_size: int = 32,
          nb_epoch: int = 400,
          margin: int = 1,
          learning_rate: Optional[float] = None,
          reduce_lr: bool = False,
          reduce_lr_patience: int = 5,
          fraction_data: Optional[float] = None,
          seed: Optional[int] = None,
          batch_level_subsampling: bool = False,
          augmentations: str = 'test_augmentations.yaml',
          wandb_api_token: Optional[str] = None,
          wandb_project: Optional[str] = None,
          wandb_entity: Optional[str] = None,
          log_messages: bool = False,
          nb_stacks: int = 2,
          with_y_hist: bool = True,
          balance: bool = False,
          version_data: bool = True,
          _qt_progress: bool = False) -> Tuple[tensorflow.keras.Model, Dict[str, Any]]:

    # FIXME THIS IS NOT GREAT:
    sample_weight_mode = None
    data_padding = 0
    if with_y_hist:  # regression
        return_sequences = True
        stride = nb_hist
        y_offset = 0
        sample_weight_mode = 'temporal'
        if ignore_boundaries:
            data_padding = int(np.ceil(
                kernel_size * nb_conv))  # this does not completely avoid boundary effects but should minimize them sufficiently
            stride = stride - 2 * data_padding
    else:  # classification
        return_sequences = False
        stride = 1  # should take every sample, since sampling rates of both x and y are now the same
        y_offset = int(round(nb_hist / 2))

    if save_prefix is None:
        save_prefix = ''

    if len(save_prefix):
        save_prefix = save_prefix + '_'

    params = locals()

    d = das.io.load(data_dir)
    params.update(d.attrs)  # add metadata from data.attrs to params for saving
    params.update({
        'nb_freq': d['train']['x'].shape[1],
        'nb_channels': d['train']['x'].shape[-1],
        'nb_classes': len(params['class_names'])
    })

    data_gen = das.data.AudioSequence(d['train']['x'], d['train']['y'], shuffle=True, nb_repeats=1, **params)
    val_gen = das.data.AudioSequence(d['val']['x'], d['val']['y'], shuffle=False, **params)

    augs = das.augmentation.Augmentations.from_yaml(augmentations)

    train_gen = das_auto.data.PairGen(data_gen, augs)

    # TODO freeze augs for validation
    val_gen = das_auto.data.PairGen(val_gen, augs)

    model = das_auto.models.model_dict['siamese'](**params)
    model.summary()

    if save_name is None:
        save_name = time.strftime('%Y%m%d_%H%M%S')
    save_name = '{0}/{1}{2}'.format(save_dir, save_prefix, save_name)

    das.utils.save_params(params, save_name)
    checkpoint_save_name = save_name + "_model.h5"  # this will overwrite intermediates from previous epochs

    callbacks = [
        ModelCheckpoint(checkpoint_save_name, save_best_only=True, save_weights_only=False, monitor='val_loss', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10),
        #  ReduceLROnPlateau(patience=10, verbose=1),
    ]

    if wandb_api_token and wandb_project:  # could also get those from env vars!
        del params['wandb_api_token']
        wandb = das.tracking.Wandb(wandb_project, wandb_api_token, wandb_entity, params)
        if wandb:
            callbacks.append(wandb.callback())

    fit_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=nb_epoch,
        verbose=verbose,
        callbacks=callbacks,
    )

    logging.info(f'Evaluating {checkpoint_save_name}.')
    custom_objects = {
        'Spectrogram': das.kapre.time_frequency.Spectrogram,
        'TCN': das.tcn.tcn_new.TCN,
        'ContrastiveLossCOLA': das_auto.layer_utils.ContrastiveLossCOLA,
    }

    # conf_mat, report = das.evaluate.evaluate(save_name, custom_objects=custom_objects)
    # logging.info(conf_mat)
    # logging.info(report)

    # if wandb_api_token and wandb_project:  # could also get those from env vars!
    #     wandb.log_test_results(report)

    # save_filename = "{0}_results.h5".format(save_name)
    # logging.info('saving to ' + save_filename + '.')
    # d = {
    #     'fit_hist': fit_history.history,
    #     'confusion_matrix': conf_mat,
    #     'classification_report': report,
    #     'x_test': d['test']['x'],
    #     'y_test': d['test']['y'],
    #     # 'labels_test': labels_test,
    #     # 'labels_pred': labels_pred,
    #     'params': params,
    # }

    # flammkuchen.save(save_filename, d)
    return model, params


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    defopt.run(train)
