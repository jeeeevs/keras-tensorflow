from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
from time import time
from keras.models import load_model


m = .3
c = 7
x_train = np.linspace(1, 100, 1000)
y_train = m * x_train + c
x_test = np.linspace(3, 200, 400)
y_test = m * x_test + c

output_dim = 1
input_dim = 1
model = Sequential()
model.add(Dense(output_dim, input_dim=input_dim))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer='rmsprop')
model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=1, validation_data=(x_test, y_test),
          callbacks=[tensorboard, checkpointer])
loss = model.evaluate(x_test, y_test, batch_size=16)
print("LOSS", loss)
 model.save('my_model_lin_reg.h5')

#loading the model and predicting
# model = load_model('my_model_lin_reg.h5')
# x = model.predict(np.array([100]))
# print(x)

