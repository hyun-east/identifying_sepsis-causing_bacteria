import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
t1 = time.time()

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


# 손실 함수 및 그 도함수
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss


def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true


# 레이어
class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)
        return input_gradient


# 신경망
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.activations = []
        self.loss = []

    def add_layer(self, input_size, output_size, activation, activation_derivative):
        self.layers.append(Layer(input_size, output_size))
        self.activations.append((activation, activation_derivative))

    def forward(self, input_data):
        output = input_data
        for layer, (activation, _) in zip(self.layers, self.activations):
            output = activation(layer.forward(output))
        return output

    def backward(self, output_gradient, learning_rate):
        for layer, (_, activation_derivative) in reversed(list(zip(self.layers, self.activations))):
            if activation_derivative is not None:
                output_gradient = activation_derivative(layer.output) * output_gradient
            output_gradient = layer.backward(output_gradient, learning_rate)

    def compute_loss(self, X, Y):
        output = self.forward(X)
        return cross_entropy_loss(Y, output)

    def compute_loss_gradient(self, X, Y):
        output = self.forward(X)
        return cross_entropy_derivative(Y, output)

    def train(self, X_train, Y_train, epochs, batch_size, learning_rate):
        print('training start')
        for epoch in range(epochs):
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                Y_batch = Y_train[i:i + batch_size]
                loss_gradient = self.compute_loss_gradient(X_batch, Y_batch)
                self.backward(loss_gradient, learning_rate)
            loss = self.compute_loss(X_train, Y_train)
            print(f"Epoch {epoch + 1}, Loss: {loss}")
            self.loss.append(loss)


df = pd.read_csv('data.csv')


# 레이블 인코딩
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['bacteria'])

# 원-핫 인코딩
onehot_encoder = OneHotEncoder()
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded).toarray()


# 입력 데이터 표준화
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('bacteria', axis=1))
Y = onehot_encoded

# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# 신경망 생성 및 레이어 추가
network = NeuralNetwork()
network.add_layer(X_train.shape[1], 32, relu, relu_derivative)
network.add_layer(32, 64, relu, relu_derivative)
network.add_layer(64,64, relu, relu_derivative)
network.add_layer(64, 32, relu, relu_derivative)

network.add_layer(32, 3, softmax, None)

# 학습
network.train(X_train, Y_train, epochs=1000, batch_size=16, learning_rate=0.0005)

# 테스트 데이터로 평가
predictions = network.forward(X_test)
predictions = np.argmax(predictions, axis=1)
Y_test_labels = np.argmax(Y_test, axis=1)
print(f"Test Accuracy: {accuracy_score(Y_test_labels, predictions)}")
print(time.time()-t1,'s')

a = np.arange(0, 1000)
plt.plot(a, network.loss)
plt.show()
