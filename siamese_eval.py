import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, TimeDistributed, Lambda, concatenate, BatchNormalization, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.layers import concatenate
from keras import backend as K

seq_dim = 20

epoch = 10
batch_size = 64
dropout = 0.1
embedding_dim = 200
embedding_mat = np.load('/home/abhishek/Desktop/Quora-Duplicate-Question-Pairs/data/preprocessed/embedding_mat.npy')
nb_words = embedding_mat.shape[0] -1

q1 = Input(shape=(seq_dim,))
q2 = Input(shape=(seq_dim,))

e1 = Embedding(nb_words+1, embedding_dim, input_length=seq_dim, weights = [embedding_mat], trainable=False)(q1)

e1 = TimeDistributed(Dense(embedding_dim, activation='relu'))(e1)
e1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim,))(e1)

e2 = Embedding(nb_words+1, embedding_dim, input_length=seq_dim, weights = [embedding_mat], trainable=False)(q2)
e2 = TimeDistributed(Dense(embedding_dim, activation='relu'))(e2)
e2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim,))(e2)

merge = concatenate([e1, e2])

merge = Dense(100, activation='relu')(merge)
merge = Dropout(dropout)(merge)
merge = BatchNormalization()(merge)

merge = Dense(100, activation='relu')(merge)
merge = Dropout(dropout)(merge)
merge = BatchNormalization()(merge)

merge = Dense(100, activation='relu')(merge)
merge = Dropout(dropout)(merge)
merge = BatchNormalization()(merge)

merge = Dense(100, activation='relu')(merge)
merge = Dropout(dropout)(merge)
merge = BatchNormalization()(merge)

is_dup_ = Dense(1, activation='sigmoid')(merge)
is_dup_ = BatchNormalization()(is_dup_)

model = Model(inputs=[q1,q2], outputs=[is_dup_])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'mse'])

model.load_weights('parameters.h5')

q1_test = np.load('/home/abhishek/Desktop/Quora-Duplicate-Question-Pairs/data/datasets/q1_test_.npy')
q2_test = np.load('/home/abhishek/Desktop/Quora-Duplicate-Question-Pairs/data/datasets/q2_test_.npy')
y_test =np.load('/home/abhishek/Desktop/Quora-Duplicate-Question-Pairs/data/datasets/y_test.npy') 

evals = model.evaluate([q1_test, q2_test], y_test, verbose=0)
print('loss = {0:.4f}, accuracy = {1:.4f}, mean_square_error = {2:.4f}'.format(evals[0],evals[1], evals[2]))