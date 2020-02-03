from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, LSTMCell, RNN, AveragePooling1D, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    amir notes: notice here the input is the entire spectrugram of the recording 
    shape is (None, input) , None means i do not know how long the length of the recording is, but i know for example
    that i will be cutting it into 0.5 seconds pieces and extract its spectrugram which has diension of input_dim
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, kernel_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        kernel_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_kernel_size + 1  # kernel = 1 does not make any changes only from kernel > 1 length changes
    return (output_length + stride - 1) // stride



def cnn_output_length_final_model(input_length, kernel_size, border_mode, stride, max_pool_size, dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        kernel_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    defaul_length = cnn_output_length(input_length, kernel_size, border_mode, stride, dilation=1)
    return defaul_length // max_pool_size


def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    rnn_input = input_data
    # TODO: Add recurrent layers, each with batch normalization
    for i in range(recur_layers):
        simp_rnn = GRU(units, activation='relu',
            return_sequences=True, implementation=2, name=str(i))(rnn_input)
        # TODO: Add batch normalization 
        bn_rnn = BatchNormalization()(simp_rnn)
        rnn_input = bn_rnn
        
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(LSTM(units = units, return_sequences=True))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """       
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    filters = 200
    kernel_size = 11
    strides= 2
    padding='valid'
    pool_size = 2
    conv_1d = Conv1D(filters = filters,
                     kernel_size = kernel_size, 
                     strides= strides, 
                     padding=padding,
                     activation='relu',
                     name='conv1d')(input_data)
    pooled_layer = AveragePooling1D(pool_size=pool_size, strides=None, padding='valid')(conv_1d)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(pooled_layer)
    drop_out = Dropout(rate = 0.3, noise_shape=None, seed=None)(bn_cnn)
    rnn_input = drop_out
    # TODO: Add recurrent layers, each with batch normalization
    for i in range(recur_layers):
        simp_rnn = Bidirectional(LSTM(units,
                                      activation='relu',
                                      return_sequences=True,
                                      implementation=2,
                                      dropout=0.3,
                                      recurrent_dropout=0.1,
                                      name='rnn_' + str(i)))(rnn_input)
#         simp_rnn = GRU(units, activation='relu',
#             return_sequences=True, implementation=2, dropout = 0.25, name=str(i))
        # TODO: Add batch normalization 
        bn_rnn = BatchNormalization()(simp_rnn)
        rnn_input = bn_rnn
        
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length_final_model(
        x, kernel_size, padding, strides, pool_size)
    print(model.summary())
    return model