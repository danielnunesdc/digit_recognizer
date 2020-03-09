
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sns.set()
np.random.seed(2)

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# exibindo os cabeçalhos
df_train.head()
df_test.head()

# Forma dos dados
print(f'Forma do conjunto de dados de Treino: {df_train.shape}')
print(f'Forma do conjunto de dados de teste: {df_test.shape}')


training_array = np.array(df_train, dtype='float32')
testing_array = np.array(df_test, dtype='float32')

w_grid = 5
l_gird = 5

# Plotando a predição manual, guiada pelas classes
fig, axes = plt.subplots(l_gird, w_grid, figsize=(15, 15))
axes = axes.ravel()

n_training = len(training_array)

for i in np.arange(0, l_gird * w_grid):
    index = np.random.randint(0, n_training)
    axes[i].imshow(training_array[index, 1:].reshape(28, 28))
    axes[i].set_title(int(training_array[index, 0]), fontsize=8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)

# Colhendo a informação do digito
x_train = df_train.drop(["label"], axis=1).values

# O dataset já se encontra em formato 28x28, porém a lib espera receber a conf das dimensões, por tanto
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = df_test.values.reshape((df_test.shape[0], 28, 28, 1))

print(f'Forma do conjunto de dados de Treino: {x_train.shape}')
print(f'Forma do conjunto de dados de teste: {x_test.shape}')

y_train = df_train["label"].values
y_train = np_utils.to_categorical(y_train)

# Mostrando imagens randomicamente
for i in range(0, 6):
    random_num = np.random.randint(0, len(x_train))
    img = x_train[random_num]
    plt.subplot(3, 2, i+1)
    plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.subplots_adjust(top=1.4)
plt.show()

"""
Diminui a probabilidade de inconsistências mediante a pequenas alterações
deteminando os valores entre 0 e 1, considerando que a variação de um pixel vai de 0 a 255. 
"""
x_train = x_train / 255
x_test = x_test / 255

# Definindo a CNN
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# optimizer = Adam()
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

# early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

batch_size = 32
epochs = 10

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.2,
                    verbose=1,
                    callbacks=[learning_rate_reduction])

# exibe o quadro com modelo do histórico
model_history = pd.DataFrame(history.history)
print(model_history)

# plotar os gráficos
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(history.history['accuracy'])+1
epoch_list = list(range(1, max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
plt.show()

# Testes de predições
predictions = model.predict_classes(x_test)

plt.figure(figsize=(7, 14))
for i in range(0, 8):
    random_num = np.random.randint(0, len(x_test))
    img = x_test[random_num]
    plt.subplot(6, 4, i+1)
    plt.margins(x=20, y=20)
    plt.title('Predição: ' + str(predictions[random_num]))
    plt.imshow(img.reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.show()

submission = pd.DataFrame({'ImageID': pd.Series(range(1, 28001)), 'Label': predictions})
submission.to_csv("results/submission.csv", index=False)




