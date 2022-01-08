import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.losses import Loss
from . import layer_utils
import das.models


model_dict = dict()


def _register_as_model(func):
    """Adds func to model_dict Dict[modelname: modelfunc]. For selecting models by string."""
    model_dict[func.__name__] = func
    return func


def get_encoder(model, layer_index=-4):
    intermediate_rep = layers.Concatenate(axis=-1)(model.layers[layer_index].input)
    embedding_network = tf.keras.Model(inputs=model.input,
                                       outputs=intermediate_rep)
    return embedding_network


def get_siamese(embedding_network):
    input_1 = layers.Input(shape=embedding_network.input_shape[1:])
    input_2 = layers.Input(shape=embedding_network.input_shape[1:])

    encoder_1 = embedding_network(input_1)
    encoder_2 = embedding_network(input_2)

    merge_layer = layers.Lambda(layer_utils.pairwise_euclidean_distance)([encoder_1, encoder_2])
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
    siamese = tf.keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    return siamese


@_register_as_model
def siamese(*args, **kwargs):
    encoder = get_encoder(das.models.model_dict['tcn'](*args, **kwargs))
    siamese = get_siamese(encoder)
    siamese.compile(loss=layer_utils.ContrastiveLossCOLA(margin=kwargs['margin']), optimizer="Adam")
    return siamese

