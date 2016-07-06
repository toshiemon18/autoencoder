import numpy as np
from keras.datasets import mnist
from keras.models  import Model
from keras.layers  import Input, Dense
from keras.optimizers import Adadelta
from keras.utils   import np_utils
from keras.utils.visualize_util import model_to_dot
from keras.utils.visualize_util import plot
from IPython.display import SVG
from keras import backend as K
from keras.callbacks import EarlyStopping


from matplotlib import pyplot as plt
plt.style.use("ggplot")

# 表示
def draw_digit(data, row, col, n):
    size = int(np.sqrt(data.shape[0]))
    plt.subplot(row, col, n)
    plt.imshow(data.reshape(size, size))
    plt.gray()

epochs = 2
batch_size = 258
input_dim = 28
input_unit_size = input_dim ** 2
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], input_unit_size)
X_train = X_train.astype('float32')

# モデルの定義
inputs = Input(shape=(input_unit_size,))
x = Dense(144, activation='relu')(inputs)
outputs = Dense(input_unit_size)(x)
model = Model(input=inputs, output=outputs)
model.compile(loss='mse', optimizer='adam')

# トレーニング
early_stopping = EarlyStopping(monitor='loss', patience=5)
model.fit(X_train, X_train, callbacks=[early_stopping],
          nb_epoch=epochs, batch_size=batch_size)

# オリジナルの入力データを描画
show_size = 10
# total = 0
# plt.figure(figsize=(20, 20))
# for i in range(show_size):
#     for j in range(show_size):
#         draw_digit(X_train[total], show_size, show_size, total+1)
#         total += 1
# plt.show()

# 隠れ層の学習状況を描画
# get_layer_output = K.function([model.layers[0].input],
#                               [model.layers[1].output])
# hidden_outputs = get_layer_output([X_train[0:show_size**2]])[0]
#
# total = 0
# plt.figure(figsize=(20, 20))
# for i in range(show_size):
#     for j in range(show_size):
#         draw_digit(hidden_outputs[total], show_size, show_size, total+1)
#         total+=1
# plt.show()

# デコードした出力層の出力を描画
get_layer_output = K.function([model.layers[0].input],
                              [model.layers[2].output])
last_outputs = get_layer_output([X_train[0:show_size**2]])[0]

total = 0
plt.figure(figsize=(20, 20))
for i in range(show_size):
    for j in range(show_size):
        draw_digit(last_outputs[total], show_size, show_size, total+1)
        total+=1
plt.show()
