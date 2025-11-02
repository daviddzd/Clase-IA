"""
Red Neuronal Multicapa para la función XOR usando TensorFlow y Keras
modulos necesarios: tensorflow, numpy, matplotlib
"""

import numpy as np # para operaciones numéricas

# Datos XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Definimos nuestra red neuronal multicapa usando Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

modelo = Sequential([
    Dense(4, activation='relu', input_dim=2),
    Dense(1, activation='sigmoid')
])

# Creamos el optimizador Adam con una tasa de aprendizaje personalizada
from tensorflow.keras.optimizers import Adam

# Tasa de aprendizaje deseada
learning_rate = 0.01
adam_optimizer = Adam(learning_rate=learning_rate)

modelo.compile(
    optimizer=adam_optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Entrenamos el modelo
history = modelo.fit(
    X, y,
    epochs=50,
    verbose=0, 
)

# Graficamos la función de pérdida
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.title('Evolución de la pérdida (Loss)')
plt.xlabel('Época')
plt.ylabel('Binary Cross-Entropy')
plt.show()

# Evaluamos el modelo
loss, accuracy = modelo.evaluate(X, y)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Hacemos predicciones finales
pred = (modelo.predict(X) > 0.5).astype(int)
for i, (inp, p) in enumerate(zip(X, pred)):
    print(f"{inp} -> {p[0]}")
