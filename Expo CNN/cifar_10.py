import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 ya esta dividido en entrenamiento y prueba
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Definimos los nombres de las 10 clases
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Normalizar los valores de los píxeles para que estén entre 0 y 1
# Las imágenes originales están en el rango 0-255
train_images = train_images / 255.0
test_images = test_images / 255.0

# Verifiquemos la forma de los datos
# Verás (50000, 32, 32, 3) -> 50,000 imágenes de 32x32 píxeles CON 3 CANALES (RGB)
print("Forma de las imágenes de entrenamiento:", train_images.shape)
print("Forma de las etiquetas de entrenamiento:", train_labels.shape)

# Esto nos ayuda a ver la complejidad de las imágenes
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # Las etiquetas de CIFAR-10 son arrays, por eso necesitamos el [0]
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = models.Sequential()

# BLOQUE 1
# input_shape es (32, 32, 3) para imágenes a color
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
# atchNormalization estabiliza el aprendizaje
model.add(BatchNormalization())
# Agregamos capa Conv2D para aprender más patrones
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
# Dropout "apaga" el 25% de las neuronas para evitar sobreajuste
model.add(Dropout(0.25))

# BLOQUE 2 (Aumentamos el número de filtros a 64)
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# BLOQUE 3 (Aumentamos el número de filtros a 128)
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# BLOQUE CLASIFICADOR
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
# Un Dropout más alto (50%) antes de la capa final
model.add(Dropout(0.5))
# Capa final con 10 neuronas (una por clase).
# No usamos 'softmax' aquí porque usaremos from_logits=True en la pérdida.
model.add(Dense(10))

# Imprimimos un resumen del modelo para ver la arquitectura
model.summary()

# Usamos 'adam' y 'SparseCategoricalCrossentropy'
# 'from_logits=True' es importante porque nuestra última capa no tiene softmax
model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("\n--- Iniciando Entrenamiento ---")
# Entrenamos por más épocas (ej. 25) porque el problema es más difícil
# Usamos un batch_size de 64
history = model.fit(train_images, train_labels,
                    epochs=25,
                    batch_size=64,
                    validation_data=(test_images, test_labels))
print("--- Entrenamiento Finalizado ---")

print("\n--- Evaluando Modelo ---")
# 1. Graficar la precisión y pérdida del entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión (entrenamiento)')
plt.plot(history.history['val_accuracy'], label = 'Precisión (validación)')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.title('Precisión del Modelo')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida (entrenamiento)')
plt.plot(history.history['val_loss'], label = 'Pérdida (validación)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend(loc='upper right')
plt.title('Pérdida del Modelo')
plt.show()

# 2. Evaluar la precisión final con el conjunto de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nPrecisión final en el conjunto de prueba: {test_acc*100:.2f}%")

# Creamos un modelo de probabilidad que añade la capa Softmax
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i][0], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)

plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

predicted_labels = np.argmax(predictions, axis=1)

cm = confusion_matrix(test_labels, predicted_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(10, 10))

disp.plot(cmap=plt.cm.RdPu, ax=ax, xticks_rotation='vertical')
plt.show()

