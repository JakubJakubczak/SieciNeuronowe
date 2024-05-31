import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

 # funkcja tworzy sieć dwuwarstwową
 # i wypełnia jej macierze wag wartościami losowymi
 # z zakresu od -0.1 do 0.1
 # parametry: S – liczba wejść do sieci / liczba wejść warstwy 1
 # K1 – liczba neuronów w warstwie 1
 # K2 – liczba neuronów w warstwie 2 / liczba wyjść sieci
 # wynik: W1 – macierz wag warstwy 1 sieci
 # W2 – macierz wag warstwy 2 sieci
def init2(S, K1, K2):
    W1 = np.random.uniform(low=-0.1, high=0.1, size=(S+1, K1))
    W2 = np.random.uniform(low=-0.1, high=0.1, size=(K1 + 1, K2))
    return W1,W2

# Sigmoid activation function
def sigmoid(x,B):
    return 1 / (1 + np.exp(-B * x))

def sigmoid_derivative(x, B):
    return B * sigmoid(x, B) * (1 - sigmoid(x, B))
 # funkcja symuluje działanie sieci dwuwarstwowej
 # parametry: W1 – macierz wag pierwszej warstwy sieci
 # W2 – macierz wag drugiej warstwy sieci
 # X – wektor wejść do sieci
 # sygnał podany na wejście ( sieci / warstwy 1 )
 # wynik: Y1 – wektor wyjść warstwy 1 ( przyda się podczas uczenia )
 # Y2 – wektor wyjść warstwy 2 / sieci
 # sygnał na wyjściu sieci
def dzialaj2 (W1,W2,X, B=5):
    # dodac bias do wejsc
    bias = -1
    X1_rozszerzone =  np.insert(X, 0, bias)
    #wyliczyc U1 - mnozenie wejsc z wagami
    U1 = W1.T.dot(X1_rozszerzone)
    # funkcja sigmoid
    Y1 = sigmoid(U1,B)
    #teraz wyjscia Y1 sa wejsciami i dodajemy tam -1 do wejsc
    X2 = Y1
    X2_rozszerzone =  np.insert(X2, 0, bias)
    #wyliczamy U2
    U2 = W2.T.dot(X2_rozszerzone)
    #nakładamy funkcje sigmoid
    Y2 = sigmoid(U2, B)

    return Y1, Y2, X1_rozszerzone, X2_rozszerzone
 # funkcja uczy sieć dwuwarstwową
 # na podanym ciągu uczącym (P,T)
 # przez zadaną liczbę kroków (n)
 # parametry: W1przed – macierz wag warstwy 1 przed uczeniem
 # P – ciąg uczący – przykłady - wejścia
 # T - ciąg uczący – żądane wyjścia
 # dla poszczególnych przykładów
 # n - liczba kroków
 # wynik: W1po – macierz wag warstwy 1 po uczeniu
 # W2po – macierz wag warstwy 2 po uczeniu
def ucz2(W1przed, W2przed,P,T,n,learning_rate=0.01, B=5):
    W1 = W1przed.copy()
    W2 = W2przed.copy()

    for epoch in range(n):
        for i in range(len(P)):
            X = P[i]
            target = T[i]

            # Forward pass
            Y1, Y2, X1_rozszerzone, X2_rozszerzone = dzialaj2(W1, W2, X, B)

            # Calculate error
            error = target - Y2

            # Backpropagation
            delta2 = error * sigmoid_derivative(Y2, B)
            delta1 = W2[1:].dot(delta2) * sigmoid_derivative(Y1, B)

            # Update weights
            W2 += learning_rate * np.outer(X2_rozszerzone, delta2)
            W1 += learning_rate * np.outer(X1_rozszerzone, delta1)

    return W1, W2


# Function to test the neural network
def test2(W1, W2, P, T, B=5):
    correct_predictions = 0
    total_predictions = len(P)

    for i in range(total_predictions):
        _, Y2,_,_ = dzialaj2(W1, W2, P[i], B)
        prediction = 1 if Y2 >= 0.5 else 0
        if prediction == T[i]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy, correct_predictions, total_predictions

P = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

T = np.array([0,1,1,0])
W1, W2 = init2(2,2,1)
print("Initial weights:")
print("W1:", W1)
print("W2:", W2)
print("First input example:", P[0])
Y1, Y2,_,_ = dzialaj2(W1, W2, P[0])
print("Output of layer 1:", Y1)
print("Output of layer 2:", Y2)

# Train the network
W1_trained, W2_trained = ucz2(W1, W2, P, T, 100000)

# Test the network
accuracy, correct_predictions, total_predictions = test2(W1_trained, W2_trained, P, T)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Correct Predictions: {correct_predictions}")
print(f"Total Predictions: {total_predictions}")