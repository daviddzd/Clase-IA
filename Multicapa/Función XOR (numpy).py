""" 
Red Neuronal Multicapa para la función XOR usando NumPy
modulos necesarios: numpy
"""

import numpy as np #para operaciones numéricas

# Datos de entrada y salida
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inicialización de pesos y sesgos y definimos la arquitectura de la red
np.random.seed(1)
input_neurons = 2
hidden_neurons = 4
output_neurons = 1

# La red tiene una capa ode entrada, una capa oculta y una capa de salida
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

#Definimos las funciones de activación y sus derivadas
# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

# Hiperparámetros
learning_rate = 0.1 # Definimos la tasa de aprendizaje
epochs = 10000 # Número de épocas para el entrenamiento

# Entrenamiento de la red neuronal
for epoch in range(epochs):
    # Paso hacia adelante (Forward pass)
    input_layer = X
    hidden_layer_input = np.dot(input_layer, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)

    # Cálculo del error
    error = y - output
    mse = np.mean((error) ** 2)
    # Mostrar el progreso
    if (epoch + 1) % 100 == 0:
        _mse = "{:.20f}".format(mse)
        print(f'Época {epoch + 1}, Loss function (MSE): {_mse}')
    
    # Retropropagación (Backpropagation)
    d_output = error * sigmoid_derivative(output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Actualización de pesos y sesgos
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += input_layer.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Resultados finales después del entrenamiento
input_layer = X
hidden_layer_input = np.dot(input_layer, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
predictions = sigmoid(output_layer_input)

# Imprimimos los resultados finales
print("Resultados finales:")
for i in range(0, len(X)):
    print(f"Input: {X[i]}, Target: {y[i]}, Predictions {predictions[i]}")
    
mse = ((y - predictions) ** 2).mean()
print(f"Error Cuadrático Medio (MSE): {mse}")
