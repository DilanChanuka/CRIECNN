from keras.layers import Convolution1D, Dropout, add, Activation, BatchNormalization, Bidirectional, LSTM, AveragePooling1D, Input, Dense, Flatten, Embedding
from tensorflow.keras.layers import Concatenate
from keras.models import Model
from keras_self_attention import SeqSelfAttention

def ConvolutionBlock(input, f, k):
    input_conv = Convolution1D(filters=f, kernel_size=k, padding='same')(input) 
    input_bn = BatchNormalization(axis=-1)(input_conv)
    input_at = Activation('relu')(input_bn)
    input_dp = Dropout(0.4)(input_at)    
    return input_dp

def MultiScale(input):
    A = ConvolutionBlock(input, 64, 1)
    B = ConvolutionBlock(input, 64, 1)
    C = ConvolutionBlock(input, 64, 1)
    D = ConvolutionBlock(input, 64, 1)
    B = ConvolutionBlock(B, 64, 3)        
    C = ConvolutionBlock(C, 64, 5)
    C = ConvolutionBlock(C, 64, 5)    
    D = ConvolutionBlock(D, 64, 7)
    D = ConvolutionBlock(D, 64, 7) 
    D = ConvolutionBlock(D, 64, 7)      
    merge = Concatenate(axis=-1)([A, B, C, D])    
    shortcut_y = Convolution1D(filters=256, kernel_size=1, padding='same')(input)
    shortcut_y = BatchNormalization()(shortcut_y)
    result = add([shortcut_y, merge])
    result = Activation('relu')(result)
    return result

def createECNN(embedding_matrix, sequence_len = 101):
    profile_input = Input(shape=(sequence_len, ), name='profile_input')
    embedding = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False)(profile_input)
    profile = Convolution1D(filters=128, kernel_size=3, padding='same')(embedding)
    profile = BatchNormalization(axis=-1)(profile)
    profile = Activation('relu')(profile)
    property_input = Input(shape=(sequence_len, 64), name='property_input')
    propertys = Convolution1D(filters=128, kernel_size=3, padding='same')(property_input)
    propertys = BatchNormalization(axis=-1)(propertys)
    propertys = Activation('relu')(propertys)
    sequence_input = Input(shape=(sequence_len, 84), name='sequence_input')
    sequence = Convolution1D(filters=128, kernel_size=3, padding='same')(sequence_input)
    sequence = BatchNormalization(axis=-1)(sequence)
    sequence = Activation('relu')(sequence)
    main_input = Input(shape=(sequence_len, 4),name='main_input')
    main = Convolution1D(filters=128, kernel_size=3, padding='same')(main_input)
    main = BatchNormalization(axis=-1)(main)
    main = Activation('relu')(main)
    mergeInput = Concatenate(axis=-1)([sequence, profile, propertys, main])   
    overallResult = MultiScale(mergeInput)
    overallResult = AveragePooling1D(pool_size=5)(overallResult)
    overallResult = Dropout(0.3)(overallResult)
    overallResult = Bidirectional(LSTM(120,return_sequences=True))(overallResult)
    overallResult = SeqSelfAttention(attention_activation='sigmoid', name='Attention')(overallResult)  
    overallResult = Flatten()(overallResult)
    overallResult = Dense(101, activation='relu')(overallResult)    
    ss_output = Dense(2, activation='softmax', name='ss_output')(overallResult)
    return Model(inputs=[sequence_input, profile_input, property_input, main_input], outputs=[ss_output])