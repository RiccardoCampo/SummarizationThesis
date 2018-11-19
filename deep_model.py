import os
import pickle
import keras.backend as K

from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.engine.saving import load_model
from keras.layers import Masking, Bidirectional, LSTM, Lambda, Activation, Dense


def build_model(doc_size, vector_size, loss_function, dense_layers, output_activation):
    """
    Initialize and compile a model for the specific dimensions.

    :param doc_size: maximum document size (number of sentences).
    :param vector_size: size of the sentence vector representation.
    :param loss_function: loss function.
    :param dense_layers: number of dense layers after the BLSTM.
    :param output_activation: output activation.
    :return: compiled model.
    """
    inputs = Input(shape=(doc_size, vector_size))
    mask = Masking(mask_value=0.0)(inputs)

    blstm = Bidirectional(LSTM(1, return_sequences=True), merge_mode="ave")(mask)
    blstm = Lambda(lambda x: K.squeeze(x, -1))(blstm)

    # If more than one dense layer is present between each dense layer a "relu" is placed to propagate the gradient.
    # If no dense layer is used the specified activation is directly applied.
    if dense_layers > 0:
        blstm_act = Activation("relu")(blstm)
        dense = Dense(doc_size)(blstm_act)
        for i in range(1, dense_layers):
            dense_act = Activation("relu")(dense)
            dense = Dense(doc_size)(dense_act)
        output = Activation(output_activation)(dense)
    else:
        output = Activation(output_activation)(blstm)

    # output = Lambda(crop)([output, inputs])

    model = Model(inputs=inputs, outputs=output)
    model.compile('adam', loss=loss_function, metrics=['accuracy'])

    print(model.summary())

    return model


def train_model(model, model_name, doc_matrix, score_matrix, initial_epoch,
                epochs, batch_size=1, val_size=0, save_model=False):
    """
    Train a pre-compiled model with the provided inputs.

    :param model: model to train.
    :param model_name: name of the model.
    :param doc_matrix: document matrix.
    :param score_matrix: score matrix.
    :param initial_epoch: initial epoch., useful to produce a more significant graph.
    :param epochs: number of epochs.
    :param batch_size: batch size.
    :param val_size: size of the validation set.
    :param save_model: if True the model will be stored.
    """
    if val_size > 0:
        set_size = int(doc_matrix.shape[0] - val_size)
    else:
        set_size = int(doc_matrix.shape[0])

    x_train = doc_matrix[:set_size, :, :]
    x_test = doc_matrix[set_size:, :, :]
    y_train = score_matrix[:set_size, :]
    y_test = score_matrix[set_size:, :]

    log_path = os.getcwd() + "/results/logs/" + model_name
    tensorboard = TensorBoard(log_dir=log_path)

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=[x_test, y_test],
                        callbacks=[tensorboard],
                        initial_epoch=initial_epoch)

    history_path = os.getcwd() + "/results/histories/" + model_name + ".hst"
    history = history.history
    if os.path.isfile(history_path):
        with open(history_path, "rb") as file:
            old_history = pickle.load(file)
        for key in history.keys():
            old_history[key].extend(history[key])
        history = old_history

    with open(history_path, "wb") as dest_file:
        pickle.dump(history, dest_file)

    if save_model:
        model.save(os.getcwd() + "/models/" + model_name + ".h5")
        K.clear_session()


def crop(x):
    """
    Crops the output(x[0]) based on the input(x[1]) padding.

    :param x: output(x[0]) and input(x[1]) of the model.
    :return: crop output.
    """
    dense = x[0]
    inputs = x[1]
    vector_size = 134

    # Build a matrix having 1 for every non-zero vector, 0 otherwise.
    padding = K.cast(K.not_equal(inputs, 0), dtype=K.floatx())  # Shape: BxDxV.
    # Transposing the matrix.
    padding = K.permute_dimensions(padding, (0, 2, 1))  # Shape: BxVxD.

    resizing = K.ones((1, vector_size))  # Shape: 1xV.
    padding = K.dot(resizing, padding)  # Shape: Bx1xD
    padding = K.squeeze(padding, 0)
    # Rebuilding the vector with only 1 and 0 (as the dot will produce vector_size and 0s).
    padding = K.cast(K.not_equal(padding, 0), dtype=K.floatx())

    # Multiplying the output by the padding (thus putting to zero the padding documents).
    return dense * padding


def predict_scores(model_name, docs):
    """
    Returns the predicted scores given model name and documents.

    :param model_name: name of the model.
    :param docs: document matrix.
    :return: predicted document scores.
    """
    model = load_model(os.getcwd() + "/models/" + model_name + ".h5")
    return model.predict(docs, batch_size=1)
