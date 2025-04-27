import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from tabulate import tabulate


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

#input is a sigmoid function
def sigmoid_derivative(x, B):
    #B * sigmoid(x, B) * (1 - sigmoid(x, B))
    return B * x * (1 - x)

# ReLU activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)
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
    # print("X:  ", X)
    bias = -1
    X1_rozszerzone =  np.insert(X, 0, bias).reshape(-1, 1)
    # print("X1: ",X1_rozszerzone)
    #wyliczyc U1 - mnozenie wejsc z wagami
    U1 = W1.T.dot(X1_rozszerzone)
    # funkcja sigmoid
    Y1 = sigmoid(U1, B)
    #teraz wyjscia Y1 sa wejsciami i dodajemy tam -1 do wejsc
    X2 = Y1
    X2_rozszerzone =  np.insert(X2, 0, bias).reshape(-1, 1)
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
def ucz2(W1przed, W2przed,P,T,n,learning_rate=0.01,batch_size=4, B=5):
    W1 = W1przed.copy()
    W2 = W2przed.copy()
    S2 = W2.shape[0]
    liczbaPrzykladow = P.shape[1]
    mse_errors_layer1 = []
    mse_errors_layer2 = []
    ce_accuracies = []
    dW1pokaz = 0
    dW2pokaz = 0
    total_error_layer1 = 0
    total_error_layer2 = 0

    for krok_uczenia in range(1, n + 1):
        correct_predictions = 0
        for krok_pokazu in range(batch_size):
            # losuj numer przykładu
            nrPrzykladu = np.random.randint(liczbaPrzykladow, size=1)

            X = P[:, nrPrzykladu]
            Y1, Y2, X1, X2 = dzialaj2(W1, W2, X)

            prediction = 1 if Y2 >= 0.5 else 0
            if prediction == T[:, nrPrzykladu]:
                correct_predictions += 1

            D2 = T[:, nrPrzykladu] - Y2
            E2 = B * D2 * Y2 * (1 - Y2)

            D1 = W2[1:S2, :] * E2
            E1 = B * D1 * Y1 * (1 - Y1)

            # zliczanie błędu średniokwadratowego
            total_error_layer1 += np.sum(D1 ** 2 / 2)
            total_error_layer2 += np.sum(D2 ** 2 / 2)

            dW1 = learning_rate* X1 * E1.T
            dW2 = learning_rate * X2 * E2.T

            dW1pokaz += dW1
            dW2pokaz += dW2

        # zastosuj poprawkę do wag sieci; reset zmiennych pomocniczych
        W1 += dW1pokaz / batch_size
        W2 += dW2pokaz / batch_size

        dW1pokaz = 0
        dW2pokaz = 0

        mse_layer1 = total_error_layer1 / ( batch_size)
        mse_layer2 = total_error_layer2 /  (batch_size)
        mse_errors_layer1.append(mse_layer1)
        mse_errors_layer2.append(mse_layer2)

        total_error_layer1 = 0
        total_error_layer2 = 0

        accuracy = correct_predictions / ( batch_size)
        ce_accuracies.append(accuracy)


    return W1, W2, mse_errors_layer1, mse_errors_layer2, ce_accuracies


# Function to test the neural network
def test2(W1, W2, P, T, B=5):
    correct_predictions = 0
    total_predictions = T.shape[1]
    tab = []
    print("Wartości oczekiwane")
    print(f"{T}")
    for i in range(T.shape[1]):
        _, Y2,_,_ = dzialaj2(W1, W2, P[:, [i]], B)

        prediction = 1 if Y2 >= 0.5 else 0
        if prediction == T[:, i]:
            correct_predictions += 1
        tab.append(Y2)
    tab_np = np.array(tab)
    print("Wyniki")
    print(f"{tab_np}")
    accuracy = correct_predictions / total_predictions
    return accuracy, correct_predictions, total_predictions


def plot_errors(mse_errors_layer1, mse_errors_layer2, ce_accuracies):
    epochs = range(len(mse_errors_layer1))

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, mse_errors_layer1, 'b', label='MSE Layer 1')
    plt.title('Mean Squared Error Layer 1')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, mse_errors_layer2, 'r', label='MSE Layer 2')
    plt.title('Mean Squared Error Layer 2')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, ce_accuracies, 'g', label='CE')
    plt.title('Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


P = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
T = np.array([[0, 1, 1, 0]])
W1, W2 = init2(2,2,1)
print("Initial weights:")
print("W1:", W1)
print("W2:", W2)
print("First input example:", P[0])
Y1, Y2,_,_ = dzialaj2(W1, W2, P[:, [0]])
print("Output of layer 1:", Y1)
print("Output of layer 2:", Y2)

# Train the network
n_steps = 10000
learning_rate = 0.1
W1_trained, W2_trained, mse_errors_layer1, mse_errors_layer2,ce_accuracies = ucz2(W1, W2, P, T, n_steps,learning_rate)

# Test the network
accuracy, correct_predictions, total_predictions = test2(W1_trained, W2_trained, P, T)

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Correct Predictions: {correct_predictions}")
print(f"Total Predictions: {total_predictions}")

# pokazEfektNaukiSieci()
# Plot errors and accuracy
plot_errors(mse_errors_layer1, mse_errors_layer2, ce_accuracies)

# sigmoid derivative
# sam gradient