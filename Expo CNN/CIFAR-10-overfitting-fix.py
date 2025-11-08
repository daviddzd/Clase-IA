"""
pip install tensorflow numpy matplotlib scikit-learn seaborn
"""
# Importaciones necesarias
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Definir los nombres de las clases de CIFAR-10 para usarlos después en los gráficos
class_names = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

# Cargar el dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# ----- Preprocesamiento de los Datos -----

# 1. Normalización de las imágenes:
# Los pixeles de las imágenes vienen en un rango de 0 a 255.
# Dividimos entre 255 para que los valores queden en un rango de 0 a 1.
# Esto ayuda al modelo a entrenar más rápido y de forma más estable.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 2. One-Hot Encoding de las etiquetas:
# Las etiquetas (y_train, y_test) son números del 0 al 9.
# Las convertimos a un formato "one-hot", que es un vector de 10 posiciones.
# Por ejemplo, la etiqueta '3' (gato) se convierte en [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
# Es el formato que necesita nuestra capa de salida con activación 'softmax'.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Crear el modelo secuencial
model = Sequential()

# --- Parte 1: La Base Convolucional ---

# Capa 1: Conv2D + MaxPooling2D
# Conv2D: Aplica 32 filtros para encontrar características básicas.
# 'relu' es una función de activación que ayuda a la red a aprender patrones no lineales.
# 'input_shape' solo se pone en la primera capa.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2))) # Reduce el tamaño de la imagen para enfocarse en las características más importantes.

# Capa 2: Conv2D + MaxPooling2D
# Agregamos más filtros para que la red aprenda características más complejas.
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# --- Parte 2: El Clasificador ---

# Flatten: Convierte el mapa de características 2D en un vector 1D para poder conectarlo a las capas densas.
model.add(Flatten())

# Capa Densa (totalmente conectada)
# 64 neuronas que aprenden a combinar las características extraídas.
model.add(Dense(64, activation='relu'))

# Capa de Salida
# 10 neuronas, una para cada clase (avión, perro, gato, etc.).
# 'softmax' convierte las salidas en probabilidades, para que la suma de todas sea 1.
# La neurona con la probabilidad más alta será la predicción del modelo.
model.add(Dense(10, activation='softmax'))

# Imprimir un resumen de la arquitectura del modelo
model.summary()

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
print("Entrenando el modelo...")
history = model.fit(x_train, y_train,
                    epochs=15, # Puedes empezar con 15 y luego probar con más, como 25.
                    batch_size=64,
                    validation_data=(x_test, y_test))

# ----- Visualización del Aprendizaje -----

# 1. Gráfica de Precisión (Accuracy)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# 2. Gráfica de Pérdida (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.show()

# ----- Evaluación Final y Matriz de Confusión -----

# Evaluar el modelo con los datos de prueba
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'\nPérdida en el conjunto de prueba: {loss:.4f}')
print(f'Precisión en el conjunto de prueba: {accuracy:.4f}')

# Generar predicciones para la matriz de confusión
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Crear la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.xlabel('Clase Predicha')
plt.ylabel('Clase Verdadera')
plt.show()