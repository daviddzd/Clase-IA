import numpy as np
# Entradas para el perceptron 
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
# Salidas
Y = np.array([0, 0, 0, 1])
# Pesos para las entradas
weights = np.array([1.0, 0.5])
# Tasa de aprendizaje
lr = 0.01
# Epocas
epochs = 100
# Sesgo
bias =  1.0

class Perceptron:
    def __init__(self, lr, epochs, weights, bias):
        """
            Constructor del perceptron:
            Guarda las variables
            lr -> tasa de aprendizaje
            epochs -> numero de epocas
            weights -> vector de pesos iniciales
            bias -> sesgo inicial
        """
        self.lr = lr
        self.epochs = epochs
        self.weights = weights
        self.bias = bias

    def fit(self, X, Y):
        """
            Realiza el entrenamiento del Perceptron.
        """
        # Recorre el dataset la cantidad indicada en epocas
        for epoch in range(self.epochs):
            for j in range(X.shape[0]):
                # Calcula la salida del perceptrón para la entrada actual
                y_pred = self.activation_function(np.dot(self.weights, X[j]) + self.bias)
                # Calcula el error
                loss = Y[j] - y_pred
                # Actualiza los pesos y el sesgo
                self.weights += self.lr * loss * X[j]  
                self.bias += self.lr * loss
            print(f"Epoch {epoch}, Optimized Weights are {self.weights}, and bias is {self.bias}")
        # Imprime los valores finales de los parámetros aprendidos            
        print(f"Optimized Weights are {self.weights} and bias is {self.bias}")

    def activation_function(self, activation):
        """
            Función de activacion escalon
        """
        return 1 if activation >= 0 else 0

    def prediction(self, X):
        """
            Calcula la salida del Perceptron para cada fila de entradas X.
        """
        # Calcula producto punto + bias para todas las entradas
        sum_ = np.dot(X, self.weights) + self.bias
        # Mensaje input y su predicción
        for i, s in enumerate(sum_):
            print(f"Input: {X[i]}, Predictions: {self.activation_function(sum_[i])}")
        # Devuelve un array con todas las predicciones
        return np.array([self.activation_function(s) for s in sum_])
    
# Crear una instancia del perceptrón
p = Perceptron(lr=lr, epochs=epochs, weights=weights, bias=bias)
# Entrenar el modelo
p.fit(X, Y)
# Usar el modelo entrenado para realizar predicciones
predictions = p.prediction(X)           

