import numpy as np

# ******************************************************
# CAMBIO 1: Entradas (X) con una sola columna
# Solo tenemos dos casos: entrada 0 y entrada 1
X = np.array([[0],
              [1]])

# ******************************************************
# CAMBIO 2: Salidas (Y) para la operación NOT
# [0] -> 1
# [1] -> 0
Y = np.array([1, 0])

# ******************************************************
# CAMBIO 3: Pesos iniciales ajustados
# Ahora es un array con un solo peso, ya que solo hay una entrada
weights = np.array([1.0]) 

# Tasa de aprendizaje
lr = 0.01
# Epocas
epochs = 100
# Sesgo (Bias)
bias = 1.0

class Perceptron:
    def __init__(self, lr, epochs, weights, bias):
        """
            Constructor del perceptron:
            Guarda las variables
        """
        self.lr = lr
        self.epochs = epochs
        # Aseguramos que los pesos se manejen correctamente como un vector
        self.weights = weights.reshape(-1) 
        self.bias = bias

    def fit(self, X, Y):
        """
            Realiza el entrenamiento del Perceptron.
        """
        for epoch in range(self.epochs):
            for j in range(X.shape[0]):
                # Nota: X[j] es ahora un array 1D que representa la única entrada
                # np.dot(self.weights, X[j]) sigue funcionando correctamente
                y_pred = self.activation_function(np.dot(self.weights, X[j]) + self.bias)
                loss = Y[j] - y_pred
                
                # Actualiza los pesos (el peso único)
                self.weights += self.lr * loss * X[j]  
                # Actualiza el sesgo
                self.bias += self.lr * loss
                
        print(f"Optimized Weight is {self.weights[0]:.4f} and bias is {self.bias:.4f}")

    def activation_function(self, activation):
        """
            Función de activacion escalón (Heaviside step function)
        """
        # Devuelve 1 si la suma ponderada + bias es >= 0, sino devuelve 0
        return 1 if activation >= 0 else 0

    def prediction(self, X):
        """
            Calcula la salida del Perceptron para cada fila de entradas X.
        """
        # Calcula el producto punto (multiplicación por el único peso) + bias
        sum_ = np.dot(X, self.weights) + self.bias
        
        # Muestra la entrada y su predicción
        for i, s in enumerate(sum_):
            # Usamos X[i][0] para mostrar el valor de entrada individual
            print(f"Input: {X[i][0]}, Prediction: {self.activation_function(s)}")
            
        return np.array([self.activation_function(s) for s in sum_])
    
# Crear una instancia del perceptrón
p = Perceptron(lr=lr, epochs=epochs, weights=weights, bias=bias)

# Entrenar el modelo
p.fit(X, Y)

# Usar el modelo entrenado para realizar predicciones
print("\n--- Resultados de Predicción ---")
predictions = p.prediction(X)