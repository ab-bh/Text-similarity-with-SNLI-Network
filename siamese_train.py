## useful links:
# www.keras.io
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, TimeDistributed, Lambda, concatenate, BatchNormalization, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.layers import concatenate
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split as tts

q1_data = np.load('q1_train.npy')
q2_data = np.load('q2_train.npy')
is_dup = np.load('is_dup.npy')
embedding_mat = np.load('embedding_mat.npy')
nb_words = embedding_mat.shape[0] -1


epoch = 10
batch_size = 64
dropout = 0.1
embedding_dim = 200
seq_dim = q1_data.shape[1]


X = np.stack((q1_data, q2_data), axis=1)
y = is_dup


X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2)
q1_train = X_train[:,0]
q2_train = X_train[:,1]
q1_test = X_test[:,0]
q2_test = X_test[:,1]


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



#model.summary() # to check network structure




checkpointer = ModelCheckpoint(filepath='parameters.h5', verbose=1, save_best_only=True, monitor= 'val_acc', mode='max', period=2)
tfboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
history = model.fit([q1_train, q2_train],
          y_train,
          batch_size=batch_size,
          epochs=epoch,
          verbose=1,
          callbacks=[checkpointer, tfboard],
          shuffle=True,
          validation_split=0.1
         )




print (history.history.keys()) # check paramters which were evaluated

# some plots for evaluation of the network

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()