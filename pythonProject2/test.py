import sys
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


beta = 5


def init2(S, K1, K2):

    W1 = np.random.uniform(low=-0.1, high=0.1, size=(S + 1, K1))
    W2 = np.random.uniform(low=-0.1, high=0.1, size=(K1 + 1, K2))
    return W1, W2

def sigmoid(x,B):
    return 1 / (1 + np.exp(-B * x))

#input is a sigmoid function
def sigmoid_derivative(x, B):
    #B * sigmoid(x, B) * (1 - sigmoid(x, B))
    return B * x * (1 - x)

def dzialaj2(W1, W2, X):
    """
    funkcja symuluje działanie sieci dwuwarstwowej

    parametry: W1 – macierz wag pierwszej warstwy sieci
               W2 – macierz wag drugiej warstwy sieci
               X –  wektor wejść do sieci
                    sygnał podany na wejście (sieci / warstwy 1)

    wynik:     Y1 – wektor wyjść warstwy 1 (przyda się podczas uczenia)
               Y2 – wektor wyjść warstwy 2 / sieci
                    sygnał na wyjściu sieci
    """
    global beta

    # dodac bias do wejsc
    # print("X:  ", X)
    bias = -1
    X1_rozszerzone = np.insert(X, 0, bias).reshape(-1, 1)
    # print("X1: ", X1_rozszerzone)
    # wyliczyc U1 - mnozenie wejsc z wagami
    U1 = W1.T.dot(X1_rozszerzone)
    # funkcja sigmoid
    Y1 = sigmoid(U1, beta)
    # teraz wyjscia Y1 sa wejsciami i dodajemy tam -1 do wejsc
    X2 = Y1
    X2_rozszerzone = np.insert(X2, 0, bias).reshape(-1, 1)
    # wyliczamy U2
    U2 = W2.T.dot(X2_rozszerzone)
    # nakładamy funkcje sigmoid
    Y2 = sigmoid(U2, beta)

    return Y1, Y2,X1_rozszerzone,X2_rozszerzone

def ucz2(W1przed, W2przed, P, T, n, m, e, k):
    """
    funkcja uczy sieć dwuwarstwową
    na podanym ciągu uczącym (P,T)
    przez zadaną liczbę kroków (n)

    parametry: W1przed – macierz wag warstwy 1 przed uczeniem
               W1przed – macierz wag warstwy 2 przed uczeniem
               P – ciąg uczący – przykłady - wejścia
               T – ciąg uczący – żądane wyjścia
                   dla poszczególnych przykładów
               n – liczba epok uczenia
               m – maksymalna liczba kroków uczenia
               e – błąd, który sieć ma osiągnąć
               k – liczba pokazów w kroku

    wynik:     W1po – macierz wag warstwy 1 po uczeniu
               W2po – macierz wag warstwy 2 po uczeniu
    """
    liczbaPrzykladow = P.shape[1]

    W1 = W1przed
    W2 = W2przed

    S2 = W2.shape[0]

    # inicjalizacja zmiennych
    wspUcz = 0.1
    blad2poprzedni = 0
    dW1 = 0
    dW2 = 0
    global beta
    plot_data2 = {}
    plot_data1 = {}
    dW1pokaz = 0
    dW2pokaz = 0
    blad1pokaz = 0
    blad2pokaz = 0

    for krok_uczenia in range(1, n + 1):
        for krok_pokazu in range(k):
            # losuj numer przykładu
            nrPrzykladu = np.random.randint(liczbaPrzykladow, size=1)

            # podaj przykład na wejścia i oblicz wyjścia
            X = P[:, nrPrzykladu]
            # X1 = np.vstack((-1, X))
            Y1, Y2,X1,X2 = dzialaj2(W1, W2, X)

            # X2 = np.vstack((-1, Y1))

            # oblicz błędy na wyjściach warstw
            D2 = T[:, nrPrzykladu] - Y2
            E2 = beta * D2 * Y2 * (1 - Y2)

            D1 = W2[1:S2, :] * E2
            E1 = beta * D1 * Y1 * (1 - Y1)

            # zliczanie błędu średniokwadratowego
            blad1pokaz += np.sum(D1 ** 2 / 2)
            blad2pokaz += np.sum(D2 ** 2 / 2)
            # print(X1.shape[0], X1.shape[1])
            # print(E1.shape[0], E1.shape[1])
            # oblicz poprawki wag (momentum)
            dW1 = wspUcz * X1 * E1.T
            dW2 = wspUcz * X2 * E2.T

            dW1pokaz += dW1
            dW2pokaz += dW2

        # zastosuj poprawkę do wag sieci; reset zmiennych pomocniczych
        W1 += dW1pokaz / k
        W2 += dW2pokaz / k

        dW1pokaz = 0
        dW2pokaz = 0

        # uśrednienie błędu średniokwadratowego otrzymanego podczas pokazu i zapisanie go; reset zmiennych pomocniczych
        blad1 = blad1pokaz / k
        blad2 = blad2pokaz / k
        plot_data2[krok_uczenia] = blad2
        plot_data1[krok_uczenia] = blad1

        blad1pokaz = 0
        blad2pokaz = 0

    return W1, W2, plot_data1, plot_data2


################################################## funkcje dodatkowe, zdefiniowane w celu poprawienia czytelności w 'main'

def testujSiec():
    Y1, Y2a,_,_ = dzialaj2(W1po, W2po, P[:, [0]])
    Y1, Y2b,_,_ = dzialaj2(W1po, W2po, P[:, [1]])
    Y1, Y2c,_,_ = dzialaj2(W1po, W2po, P[:, [2]])
    Y1, Y2d,_,_ = dzialaj2(W1po, W2po, P[:, [3]])
    Y = [Y2a, Y2b, Y2c, Y2d]

    Ypo = np.array([[]])
    for i in Y:
        Ypo = np.append(Ypo, i, axis=1)

    return Ypo


def pokazEfektNaukiSieci():
    Ypo = testujSiec()
    print("Wartosci oczekiwane:\n", tabulate(T, tablefmt='fancy_grid'), sep='')
    print("Wyniki po nauczniu sieci:\n", tabulate(Ypo, tablefmt='fancy_grid'), sep='')


def wykresBleduSredniokwadratowego():
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(list(plot_data1.keys()), list(plot_data1.values()))
    ax[0].set_xlim(1, len(plot_data1.keys()))
    ax[0].grid()
    ax[0].set_title("MSE warstwa 1")
    ax[0].set_ylabel('Wartość błędu')
    ax[0].set_xlabel('Krok uczenia')

    ax[1].plot(list(plot_data2.keys()), list(plot_data2.values()))
    ax[1].set_xlim(1, len(plot_data2.keys()))
    ax[1].grid()
    ax[1].set_title("MSE warstwa 2")
    ax[1].set_ylabel('Wartość błędu')
    ax[1].set_xlabel('Krok uczenia')

    fig.tight_layout()


def wykresZmiennaWaga2D():
    blad = []
    zakres_zmiany_wagi = [x / 10 for x in range(-10, 11)]  # -1 do +1 co 0.1
    pierwotna_wartosc_wagi = W2po[0]

    # testowanie sieci dla kolejnych wartości wagi
    for aktualna_wartosc_wagi in zakres_zmiany_wagi:
        W2po[0] = aktualna_wartosc_wagi

        Ypo = testujSiec()

        # obliczenie błędu średniokwadratowego dla aktualnej wartości wagi
        odchylenie_od_wart_oczekiwanej = T - Ypo
        blad.append(np.sum(odchylenie_od_wart_oczekiwanej ** 2 / 2))

    # przywrócenie pierwotnej wartości po zakończonym testowaniu
    W2po[0] = pierwotna_wartosc_wagi

    fig, ax = plt.subplots(1, 1)

    ax.plot(zakres_zmiany_wagi, blad)
    ax.grid()
    ax.set_title("Jedna zmienna waga")
    ax.set_ylabel('Wartość błędu')
    ax.set_xlabel('Wartość wagi')





################################################# 'main'

if __name__ == '__main__':
    # przygotowanie zmiennych
    P = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
    T = np.array([[0, 1, 1, 0]])
    ilosc_petli_nauczania = 5000
    maks_ilosc_krokow_nauczania = 3500
    blad_do_osiagniecia = 0.0003
    liczba_pokazow = 10

    # stworzenie i uczenie sieci
    W1przed, W2przed = init2(2, 6, 1)
    W1po, W2po, plot_data1, plot_data2 = ucz2(W1przed, W2przed, P, T, ilosc_petli_nauczania,
                                              maks_ilosc_krokow_nauczania, blad_do_osiagniecia, liczba_pokazow)
    print(W2po)
    # wizualizacja
    pokazEfektNaukiSieci()
    wykresBleduSredniokwadratowego()
    wykresZmiennaWaga2D()

    # wykresZmiennaWaga3D()

    plt.show()
