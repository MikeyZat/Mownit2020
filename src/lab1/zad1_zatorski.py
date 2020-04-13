import numpy as np
import time
from matplotlib import pyplot as plt


# ZAD 1

# 1.1
def ex_1(number, n):
    x = np.float32(number)
    numbers = np.repeat(x, n)
    numbers_sum = np.float32(0)
    for el in numbers:
        numbers_sum += el
    return numbers_sum


# 1.2
def calculate_erros(expected_sum, actual_sum):
    abs_error = abs(expected_sum - actual_sum)
    non_abs_error = (abs_error / actual_sum) * 100
    return abs_error, non_abs_error


def ex_2(number, n):
    expected_sum = np.float32(number * n)
    actual_sum = ex_1(number, n)
    abs_error, non_abs_error = calculate_erros(expected_sum, actual_sum)

    # print(f'Expected {expected_sum}, got {actual_sum} \n'
    #       f'Abs error {abs_error}, relative error {non_abs_error}%')

    return abs_error, non_abs_error


# 1.3
def ex_3(number, n):
    x = np.float32(number)
    relative_erros = []
    numbers = np.repeat(x, n)
    numbers_sum = np.float32(0)
    i = 0
    for el in numbers:
        numbers_sum += el
        i += 1
        if i % 25000 == 0:
            _, non_abs_error = calculate_erros(np.float(number * i), numbers_sum)
            relative_erros.append(non_abs_error)
    return relative_erros


def draw_plot(plot_data):
    plt.plot(plot_data)
    plt.show()


# 1.4
def ex_4(number, n):
    x = np.float32(number)
    numbers = np.repeat(x, n)
    numbers_sum = rec_sum(numbers)
    return numbers_sum


def rec_sum(numbers):
    size = numbers.shape[0]
    if size == 0:
        return 0
    if size == 1:
        return numbers[0]

    return np.float32(rec_sum(numbers[:size // 2]) + rec_sum(numbers[size // 2:]))


def zadanie_1():
    number = 0.53125
    N = 10 ** 7
    print("Suma 10^7 liczb pojedynczej precyzji, v = 0.53125")
    print(ex_1(number, N))
    print("Błąd bezwzględny oraz błąd względny")
    print(ex_2(number, N))
    print("Błąd względny jest tak duży, ponieważ z każdą następną "
          "iteracją dodajemy od siebie dwie skrajnie różne liczby:\n"
          "Sumę (liczba rzędu 10^7) oraz aktualny składnik (0.53125)")
    plot_data = ex_3(number, N)
    print("Tak rośnie błąd względny w kolejnych krokach:")
    print(plot_data)
    draw_plot(plot_data)
    print("Błąd względny przez pewien czas wynosi 0 (ponieważ dodawane liczby nie różnią się jeszcze tak bardzo),\n"
          "a od pewnego momentu wykres przypomina krzywą logarytmu. Ten moment następuję kiedy suma jest na tyle\n"
          "duża, że dodanie do niej małego składnika powoduje 'ucięcie' mniej znaczących bitów.")
    print("Suma policzona rekurencyjnie:")
    recursive_sum = ex_4(number, N)
    print(recursive_sum)
    # 1.5
    print("Błąd bezwzględny i względny sumy rekurencyjnej")
    print(calculate_erros(number * N, recursive_sum))
    print("Błąd znacznie zmalał, ponieważ w każdym kroku sumujemy ze sobą dwie identyczne liczby")
    # podpunkt 1.6 jest zrealizowany w zadaniu 2.3
    # 1.7
    number = np.float32(0.7312731)
    M = 10 ** 3
    print(f"liczba - {number}, N = {M}")
    print(calculate_erros(number * M, ex_4(number, M)))


zadanie_1()

# ZAD 2


def kahan_sum(number, n):
    tab = np.repeat(np.float32(number), n)

    tab_sum = np.float32(0)
    err = np.float32(0)

    for el in tab:
        y = np.float32(el - err)
        tmp = np.float32(tab_sum + y)
        err = (tmp - tab_sum) - y
        tab_sum = tmp

    return tab_sum


def kahan_errors(number, n):
    expected_sum = number * n
    actual_sum = kahan_sum(number, n)
    return calculate_erros(expected_sum, actual_sum)


def benchmark(number, n, algorithm, name):
    start = time.time()
    algorithm(number, n)
    end = time.time()
    print(f"{name} took {end - start}s to execute")


def zadanie_2():
    number = 0.53125
    N = 10 ** 7

    # 2.1
    print("Suma z zad1 policzona alogrytmem Kahana")
    print(kahan_sum(number, N))
    print("Błąd bezwzględny i względny sumy kahana dla tych samych danych co w zad1")
    print(kahan_errors(number, N))
    # 2.2
    print("Algorytm Kahana używa zmiennej err do odzyskiwania bitów mniej znaczących liczby dodawanej do sumy.\n"
          "Pozwala to na zmniejszenie niedokładności podczas sumowania dużych liczb z małymi.")
    # 2.3
    print("Porównanie czasów")
    benchmark(number, N, ex_1, "Normal sum")
    benchmark(number, N, ex_4, "Recursive sum")
    benchmark(number, N, kahan_errors, "Kahan sum")


zadanie_2()

# ZAD 3

def rieman_function(s, n, precision):
    sum = precision(0)
    for k in range(1, n + 1):
        sum += precision(
            1 / (k ** s)
        )
    return sum


def rieman_function_reverse(s, n, precision):
    sum = precision(0)
    for k in range(n, 0, -1):
        sum += precision(
            1 / (k ** s)
        )
    return sum


def dirichlet_function(s, n, precision):
    sum = precision(0)
    for k in range(1, n + 1):
        sum += precision(
            ((-1) ** (k - 1)) * (1 / (k ** s))
        )
    return sum


def dirichlet_function_reverse(s, n, precision):
    sum = precision(0)
    for k in range(n, 0, -1):
        sum += precision(
            ((-1) ** (k - 1)) * (1 / (k ** s))
        )
    return sum


def print_results(list_1, list_2):
    zip_list = zip(list_1, list_2)
    difference = []
    for sublist_1, sublist_2 in zip_list:
        zip_sublist = zip(sublist_1, sublist_2)
        difference.append(
            [abs(el_1 - el_2) for el_1, el_2 in zip_sublist]
        )
    return difference


def zadanie_3():
    s = np.array([2, 3.6667, 5, 7.2, 10])
    n = np.array([50, 100, 200, 500, 1000])
    s_float = np.float32(s)
    s_double = np.float64(s)

    def print_results(float_list, double_list):
        for s_i, s_el in enumerate(s):
            for n_i, n_el in enumerate(n):
                print(f"s = {s_el}, n = {n_el}")
                print(f"Float 32 - {float_list[s_i, n_i]}, Float 64 - {double_list[s_i, n_i]}"
                      f", difference -  {abs(float_list[s_i, n_i] - double_list[s_i, n_i])}")

    results_rieman_float = np.array([
        [rieman_function(s_el, n_el, np.float32) for s_el in s_float]
        for n_el in n
    ], dtype="float32")

    results_rieman_double = np.array([
        [rieman_function(s_el, n_el, np.float64) for s_el in s_double]
        for n_el in n
    ], dtype="float64")

    print("-----------------------------\n\nRiemann sumując w przód float32 vs float64\n")
    print_results(results_rieman_float, results_rieman_double)

    results_rieman_reverse_float = np.array([
        [rieman_function_reverse(s_el, n_el, np.float32) for s_el in s_float]
        for n_el in n
    ], dtype="float32")

    results_rieman_reverse_double = np.array([
        [rieman_function_reverse(s_el, n_el, np.float64) for s_el in s_double]
        for n_el in n
    ], dtype="float64")

    print("-----------------------------\n\nRiemann sumując wstecz float32 vs float64\n")
    print_results(results_rieman_reverse_float, results_rieman_reverse_double)

    results_dirichlet_float = np.array([
        [dirichlet_function(s_el, n_el, np.float32) for s_el in s_float]
        for n_el in n
    ], dtype="float32")

    results_dirichlet_double = np.array([
        [dirichlet_function(s_el, n_el, np.float64) for s_el in s_double]
        for n_el in n
    ], dtype="float64")

    print("-----------------------------\n\nDirichlet sumując w przód float32 vs float64\n")
    print_results(results_dirichlet_float, results_dirichlet_double)

    results_dirichlet_reverse_float = np.array([
        [dirichlet_function_reverse(s_el, n_el, np.float32) for s_el in s_float]
        for n_el in n
    ], dtype="float32")

    results_dirichlet_reverse_double = np.array([
        [dirichlet_function_reverse(s_el, n_el, np.float64) for s_el in s_double]
        for n_el in n
    ], dtype="float64")

    print("-----------------------------\n\nDirichlet sumując wstecz float32 vs float64\n")
    print_results(results_dirichlet_reverse_float, results_dirichlet_reverse_double)

    # interpretacja
    print("-----------------------\n\n"
          "W przypadku funkcji dzeta Riemenna widzimy, że sumowanie wstecz generuje wyraźnie mniejszy błąd.\n"
          "Dzieje się tak dlatego, że sumując wprost dodajemy coraz mniejsze liczby do coraz większej sumy\n"
          ", natomiast sumując wstecz dodajemy coraz większe liczby do coraz większej sumy => różnica wielkości\n"
          "sumy i dodawanej liczby są mniejsze => generuje to mniejszy błąd.")

    print("\nW przypadku funkcji eta Dirichleta również możemy zaobserwować podobny efekt.\n"
          "Jednak tutaj nie jest on tak duży, dzieje się tak dlatego, że suma nie rośnie liniowo w kolejnych\n"
          "iteracjach (przez to, że raz odejmujemy a raz dodajemy daną liczbę). W tym przypadku efekt błędu\n"
          "generowanego poprzez dodanie dużej liczby do małej jest nieco 'stłumiony'.")


zadanie_3()


# ZAD 4


def rec_image(x, r, precision):
    return precision(r) * precision(x) * (precision(1.0) - precision(x))


def bifurcation_diagram(r, X, n, margin, precision, op=.25):
    fig, ax = plt.subplots(1, X.shape[0], figsize=(16, 5))

    for x_i, x in enumerate(X):
        x_0 = x
        for i in range(n):
            x = rec_image(x, r, precision)
            if i >= (n - margin):
                ax[x_i].plot(r, x, 'k', alpha=op)
        ax[x_i].set(title=f'Diagram bifurkacyjny dla x_0 = {x_0}', xlabel='r', ylabel='x_n')
    plt.show()


def ex_a(x, n):
    bifurcation_diagram(np.linspace(1.0, 4.0, n), x, n, 100, np.float32)


def ex_b(x, n):
    bifurcation_diagram(np.linspace(3.75, 3.8, n), x, n, 3, np.float32)
    bifurcation_diagram(np.linspace(3.75, 3.8, n), x, n, 3, np.float64)


# dodany epsilon ponieważ dla niektórych danych nigdy nie dochodziło do 0
def ex_c(r, x, eps=np.float(10 ** -12)):
    zero = np.float32(0)
    i = 0
    while abs(x - zero) > eps:
        x = r * x * (1 - x)
        i += 1
    return i


def zadanie_4():
    N = 10**3
    x = np.array([0.123, 0.321, 0.51241, 0.8236])

    # 4.a
    ex_a(x, N)
    print("Można zauważyć, że diagram bifurkacyjny wygląda podobnie dla różnych wartości x_0.\n"
          "Dla r z przedziału [1,3] ciąg zbiega do jednej wartości, dla r = [3; ~3.5] ciag zbiega\n"
          "do dwóch wartości, natomiast dla r = [3.5, 4] widzimy, że ciąg nie zbiega do niczego konkretnego,\n"
          "jego wartości są dosyć chaotycznie rozłożone.")

    # 4.b
    print("Pierwszy zestaw wykresów - pojedyncza precyzja\n"
          "Drugi zestaw wykresów - podwójna precyzja")
    ex_b(x, N)
    print("-----------------\n\n"
          "Niewiele możemy dowiedzieć się na temat rozłożenia kolejnych wyrazów ciągu jednak\n"
          "ewidetnie diagramy dla pojedynczej i podwójnej precyzji NIE SĄ RÓWNE.\n"
          "Oznacza to, że precyzja realnie wpływa na proces wyliczania i zapamiętywania\n"
          "kolejnych wyrazów tego ciagu.")


    # 4.c
    print("---------------\n\n")
    for x_el in np.float32(x):
        print(f"Ilość iteracji potrzebna dla osiągnięcia zera dla x_0 = {x_el}")
        print(ex_c(np.float32(4), x_el))

    print("-----------------\n\n"
          "Wbrew pozorom dla najmniejszego x_0, liczba iteracji potrzebnych do zejścia do 0 była największa.\n"
          "Wiele zależy w tym wypadku od konkretnego ułożenia bitów a nie od wielkości liczby początkowej.\n"
          "Co więcej dla niektórych wartości (gdyby nie epsilon) ciąg nigdy nie osiągnąłby wartości 0.\n"
          "Jest to spowodowane skończoną precyzją komputera i generowanej przez ten fakt niedokładności arytmetycznej.")


zadanie_4()
