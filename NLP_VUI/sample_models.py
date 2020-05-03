from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Flatten, Concatenate, Dropout, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

K.set_image_dim_ordering('th')

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
    bn_rnn = BatchNormalization(name='bnorm')(simp_rnn)
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
    simp_rnn = SimpleRNN(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bnorm')(simp_rnn)
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

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # Add recurrent layers, each with batch normalization
    simp_rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(input_data)
    bn_rnn = BatchNormalization(name='bnorm')(simp_rnn)
    # layer2
    simp_rnn_2 = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn2')(bn_rnn)
    bn_rnn_2 = BatchNormalization(name='bnorm2')(simp_rnn_2)
    # layer3
    simp_rnn_3 = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn3')(bn_rnn_2)
    bn_rnn_3 = BatchNormalization(name='bnorm3')(simp_rnn_3)
    
    # TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_3)
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
    bidir_rnn = Bidirectional(GRU(units, 
                                  activation='relu', 
                                  return_sequences=True, 
                                  implementation=2, 
                                  name='bdrnn'), 
                              merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29, dropout=0.4):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    
   
    # add max pooling layer
    #max_1d_2 = MaxPooling1D(pool_size=kernel_size, strides=1, padding='same', name='maxpool_1d_2')(bn_cnn3)
    #bn_cnn4 = BatchNormalization(name='bn_conv_4')(max_1d_2)
 
    # Add a recurrent layer
#     bidir_rnn_1 = Bidirectional(GRU(units, 
#                                   activation='relu', 
#                                   return_sequences=True, 
#                                   implementation=2, 
#                                   name='bdrnn1',
#                                   recurrent_dropout=0.2,
#                                   dropout=dropout), 
#                               merge_mode='concat')(bn_cnn1)
#     bnorm_rnn_1 = BatchNormalization(name='bnorm_rnn_1')(bidir_rnn_1)
    
#     bidir_rnn_2 = Bidirectional(GRU(units, 
#                                   activation='relu', 
#                                   return_sequences=True, 
#                                   implementation=2, 
#                                   name='bdrnn2',
#                                   recurrent_dropout=0.2,
#                                   dropout=dropout), 
#                               merge_mode='concat')(bnorm_rnn_1)
#     bnorm_rnn_2 = BatchNormalization(name='bnorm_rnn_2')(bidir_rnn_2)
    
#     bidir_rnn_3 = Bidirectional(GRU(units, 
#                                   activation='relu', 
#                                   return_sequences=True, 
#                                   implementation=2, 
#                                   name='bdrnn3',
#                                   recurrent_dropout=0.2,
#                                   dropout=dropout), 
#                               merge_mode='concat')(bnorm_rnn_2)
#     bnorm_rnn_3 = BatchNormalization(name='bnorm_rnn_3')(bidir_rnn_3)
    
    
#     # TODO: Specify the layers in your network
#     # Conv1
    conv_1d_1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1')(input_data)
    max_pool_1 = MaxPooling1D(kernel_size,strides=1,padding='same', name='maxpool_1d')(conv_1d_1)
    
    # Add batch normalization
    bn_cnn1 = BatchNormalization(name='bn_conv_1')(max_pool_1)
    # add max pooling layer
#     conv_1d_2 = Conv1D(filters, kernel_size, 
#                      strides=conv_stride, 
#                      padding=conv_border_mode,
#                      activation='relu',
#                      name='conv2')(max_pool_)
#     max_pool_2 = MaxPooling1D(kernel_size, strides=1,padding='same', name='maxpool_1d_2')(conv_1d_2)
    
#     # Add batch normalization
#     bn_cnn2 = BatchNormalization(name='bn_conv_2')(max_pool_2)
  
    
    simp_rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(bn_cnn1)
    bn_rnn = BatchNormalization(name='bnorm')(simp_rnn)
    # layer2
    simp_rnn_2 = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn2')(bn_rnn)
    bn_rnn_2 = BatchNormalization(name='bnorm2')(simp_rnn_2)
    # layer3
    simp_rnn_3 = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn3')(bn_rnn_2)
    bn_rnn_3 = BatchNormalization(name='bnorm3')(simp_rnn_3)
    
    simp_rnn_4 = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn4')(bn_rnn_3)
    bn_rnn_4 = BatchNormalization(name='bnorm4')(simp_rnn_4)
    # layer3
    simp_rnn_5 = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn5')(bn_rnn_4)
    bn_rnn_5 = BatchNormalization(name='bnorm5')(simp_rnn_5)
    
#     #bn_cnn_1 = BatchNormalization(name='bn_conv_1d_1')(max_pool_1)
#     #Conv2
#     conv_1d_2 = Conv1D(filters//2, kernel_size-1, 
#                      strides=conv_stride, 
#                      padding=conv_border_mode,
#                      activation='relu',
#                      name='conv2')(max_pool_1)
#     #avg_pool_1 = GlobalAveragePooling1D()(conv_1d_2)
#     max_pool_2 = MaxPooling1D(kernel_size-1, strides=1,padding='same', name='maxpool_2d')(conv_1d_2)
#     #flat_1 = Dense(1024, activation='relu')(max_pool_1)
#     #flat_1 = Flatten(name='flat')(avg_pool_1)
#     #drop = Dropout(0.5)(conv_1d)
#     #flat_1 = Dense(1024, activation='relu')(avg_pool_1)
#     #bn_cnn_2 = BatchNormalization(name='bn_conv_1d_2')(max_pool_2)
#     #concatenated_tensor = Concatenate()([max_pool_1, max_pool_2])
#     #flat_1 = Dense(1024, activation='relu')(concatenated_tensor)
#     #flat_1 = Flatten(name='flat')(max_pool_1)
#     #intermediate_drop = (Dropout(0.5))(flat_1)
#     #RNN
       
#     bidir_rnn_1 = Bidirectional(GRU(units, 
#                                   activation='relu', 
#                                   return_sequences=True, 
#                                   implementation=2, 
#                                   name='bdrnn1',
#                                   recurrent_dropout=0.2,
#                                   dropout=dropout), 
#                               merge_mode='concat')(max_pool_1)
#     bnorm_rnn_1 = BatchNormalization(name='bnorm_rnn_1')(bidir_rnn_1)
    
# #     bidir_rnn_2 = Bidirectional(GRU(units, 
# #                                   activation='relu', 
# #                                   return_sequences=True, 
# #                                   implementation=2, 
# #                                   name='bdrnn2',
# #                                   recurrent_dropout=0.2,
# #                                   dropout=dropout), 
# #                               merge_mode='concat')(bnorm_rnn_1)
# #     bnorm_rnn_2 = BatchNormalization(name='bnorm_rnn_2')(bidir_rnn_2)
    
# #     bidir_rnn_3 = Bidirectional(GRU(units, 
# #                                   activation='relu', 
# #                                   return_sequences=True, 
# #                                   implementation=2, 
# #                                   name='bdrnn3',
# #                                   recurrent_dropout=0.2,
# #                                   dropout=dropout), 
# #                               merge_mode='concat')(bnorm_rnn_2)
# #     bnorm_rnn_3 = BatchNormalization(name='bnorm_rnn_3')(bidir_rnn_3)
    
    dense_1 = TimeDistributed(Dense(1024, activation='relu'))(simp_rnn_5)
    dense_2 = TimeDistributed(Dense(256, activation='relu'))(dense_1)
    time_dense = TimeDistributed(Dense(output_dim))(dense_2)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model