"""Code for training networks."""
import time
import logging
import flammkuchen
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import tensorflow.keras as keras
import tensorflow.keras.layers as kl

import das_auto.train_pre, das_auto.models, das_auto.layer_utils
import das.train, das.tcn, das.kapre
import tensorflow as tf

import defopt
from glob import glob

from das import data, models, utils, predict, io, evaluate, tracking

from typing import Optional


def train(load_name: str,
          *,
          data_dir: str = None,
          save_dir: str = None,
          save_prefix: Optional[str] = None,
          save_name: Optional[str] = None,
          verbose: int = 1,
          nb_epoch: int = 400,
          fraction_data: float = None,
          seed: int = None,
          freeze: bool = False,
          learning_rate: float = 0.0001,
          wandb_api_token: Optional[str] = None,
          wandb_project: Optional[str] = None,
          wandb_entity: Optional[str] = None,
          reduce_lr: bool = False):

    if save_prefix is None:
        save_prefix = ''

    if len(save_prefix):
        save_prefix = save_prefix + '_'

    params_given = locals()

    logging.info(f'loading old params and network from {load_name}.')
    model, params = utils.load_model_and_params(load_name,
                                                custom_objects={
                                                    'Spectrogram': das.kapre.time_frequency.Spectrogram,
                                                    'TCN': das.tcn.tcn_new.TCN,
                                                    'ContrastiveLossCOLA': das_auto.layer_utils.ContrastiveLossCOLA
                                                })
    if data_dir is None:
        data_dir = params['data_dir']
    if save_dir is None:
        save_dir = params['save_dir']
    if save_name is None:
        save_name = params['save_name']
    params.update(params_given)  # override loaded params with given params
    params['data_dir'] = data_dir
    params['save_dir'] = save_dir
    params['save_name'] = save_name
    logging.info(f"loading data from {params['data_dir']}")

    d = io.load(params['data_dir'], x_suffix=params['x_suffix'], y_suffix=params['y_suffix'])
    params.update(d.attrs)  # add metadata from data.attrs to params for saving

    if fraction_data is not None:  # train on a subset
        if fraction_data > 1.0:  # seconds
            logging.info(
                f"{fraction_data} seconds corresponds to {fraction_data / d.attrs['samplerate_x_Hz']} of the training data.")
            fraction_data = fraction_data / d.attrs['samplerate_x_Hz']
        logging.info(f"Using {fraction_data} of data for training and validation.")
        min_nb_samples = params['nb_hist'] * (params['batch_size'] + 2)  # ensure the generator contains at least one full batch
        first_sample_train, last_sample_train = data.sub_range(d['train']['x'].shape[0],
                                                               fraction_data,
                                                               min_nb_samples,
                                                               seed=seed)
        first_sample_val, last_sample_val = data.sub_range(d['val']['x'].shape[0], fraction_data, min_nb_samples, seed=seed)
    else:
        first_sample_train, last_sample_train = 0, None
        first_sample_val, last_sample_val = 0, None

    # TODO clarify nb_channels, nb_freq semantics - always [nb_samples,..., nb_channels] -  nb_freq is ill-defined for 2D data
    params.update({
        'nb_freq': d['train']['x'].shape[1],
        'nb_channels': d['train']['x'].shape[-1],
        'nb_classes': len(params['class_names']),
        'first_sample_train': first_sample_train,
        'last_sample_train': last_sample_train,
        'first_sample_val': first_sample_val,
        'last_sample_val': last_sample_val,
    })

    logging.info('preparing data')
    data_gen = data.AudioSequence(d['train']['x'],
                                  d['train']['y'],
                                  shuffle=True,
                                  first_sample=first_sample_train,
                                  last_sample=last_sample_train,
                                  nb_repeats=100,
                                  **params)

    val_gen = data.AudioSequence(d['val']['x'],
                                 d['val']['y'],
                                 shuffle=False,
                                 first_sample=first_sample_val,
                                 last_sample=last_sample_val,
                                 **params)
    logging.info('Training data:')
    logging.info(data_gen)
    logging.info('Validation data:')
    logging.info(val_gen)

    nb_classes = d['train']['y'].shape[1]
    if freeze or nb_classes != model.output_shape[-1]:

        sample_weight_mode = params['sample_weight_mode']
        nb_pre_conv = params['nb_pre_conv']
        upsample = True
        loss = 'categorical_crossentropy'

        encoder = model.layers[2]
        new_model = keras.Model(encoder.inputs, encoder.output, name='encoder')
        # freeze layers
        if freeze:
            for layer in new_model.layers:
                layer.trainable = False

        x = new_model.output
        x = kl.Dense(nb_classes * 3, name='dense_new1')(x)
        x = kl.Dense(nb_classes, name='dense_new2')(x)
        x = kl.Activation('softmax', name='activation_new')(x)
        if nb_pre_conv > 0 and upsample:
            x = kl.UpSampling1D(size=2**nb_pre_conv, name='upsampling_new')(x)
        output_layer = x
        model = keras.models.Model(new_model.inputs, output_layer, name='TCN_new')
        model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate, clipnorm=1.),
                      loss=loss,
                      sample_weight_mode=sample_weight_mode)

    logging.info(model.summary())
    if save_name is None:
        save_name = time.strftime('%Y%m%d_%H%M%S')
    save_name = '{0}/{1}{2}'.format(save_dir, save_prefix, save_name)

    utils.save_params(params, save_name)
    checkpoint_save_name = save_name + "_model.h5"  # this will overwrite intermediates from previous epochs

    callbacks = [
        ModelCheckpoint(checkpoint_save_name, save_best_only=True, save_weights_only=False, monitor='val_loss', verbose=1),
        EarlyStopping(monitor='val_loss', patience=20),
    ]
    if reduce_lr:
        callbacks.append(ReduceLROnPlateau(patience=5, verbose=1))

    if wandb_api_token and wandb_project:  # could also get those from env vars!
        del params['wandb_api_token']
        wandb = das.tracking.Wandb(wandb_project, wandb_api_token, wandb_entity, params)
        if wandb:
            callbacks.append(wandb.callback())

    # TRAIN NETWORK
    logging.info('start training')
    fit_history = model.fit(
        data_gen,
        epochs=nb_epoch,
        steps_per_epoch=min(len(data_gen), 1000),
        verbose=verbose,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    logging.info(f'Evaluating {checkpoint_save_name}.')
    conf_mat, report = das.evaluate.evaluate(save_name)
    logging.info(conf_mat)
    logging.info(report)

    if wandb_api_token and wandb_project:  # could also get those from env vars!
        wandb.log_test_results(report)

    save_filename = "{0}_results.h5".format(save_name)
    logging.info('saving to ' + save_filename + '.')
    d = {
        'fit_hist': fit_history.history,
        'confusion_matrix': conf_mat,
        'classification_report': report,
        'x_test': d['test']['x'],
        'y_test': d['test']['y'],
        'params': params,
    }

    flammkuchen.save(save_filename, d)
    return model, params


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    defopt.run(train)