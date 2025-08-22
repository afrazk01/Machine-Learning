# Implementing Machine learning model from scratch: Linear Regression
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        # This is the constructor for the Linear Regression model
        # learning_rate: The step size for weight updates
        # n_iterations: Number of iterations for gradient descent

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # n_samples: Number of samples in the dataset (which is the number of rows in X)
        # n_features: Number of features in the dataset (which is the number of columns in X)
        # X: Input features, shape (n_samples, n_features) (
        # y: Target values, shape (n_samples,)
        n_samples, n_features = X.shape
        # Initialize weights and bias to zeros which are the parameters we will learn 
        # n_features is given to determine the size of the weights vector
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # predict the target values using the current weights and bias
            # the formula for prediction is y = X * weights + bias
            # np.dot(X, self.weights) computes the dot product of X and weights
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            # dw: Gradient of the loss with respect to weights
            # db: Gradient of the loss with respect to bias
            # and the formula for gradients is: 1 / number of samples * dot product of X transpose and (predicted - actual)
            # and for bias it is: 1 / number of samples * sum of (predicted - actual)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    