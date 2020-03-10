import numpy as np
import time
from pandas import DataFrame
from matplotlib import pyplot as plt


# IEEE - 754

# float - 32 bity - 1 znak - 8 cecha - 23 mantysa
# double - 64 bity -

# ZAD 1

def ex_1(number, n):
    x = np.float32(number)
    numbers = np.repeat(x, n)
    numbers_sum = np.float32(0)
    for el in numbers:
        numbers_sum += el
    return numbers_sum


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
    print(ex_1(number, N))
    print(ex_2(number, N))
    plot_data = ex_3(number, N)
    print(plot_data)
    draw_plot(plot_data)
    print(ex_4(number, N))


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

    print(kahan_sum(number, N))
    print(kahan_errors(number, N))
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


def get_difference(list_1, list_2):
    zip_list = zip(list_1, list_2)
    difference = []
    for sublist_1, sublist_2 in zip_list:
        zip_sublist = zip(sublist_1, sublist_2)
        difference.append(
            [abs(el_1 - el_2) for el_1, el_2 in zip_sublist]
        )
    return difference


def formatted_table(s, n, data):
    return DataFrame(data, s, n).rename_axis('s', axis=0).rename_axis('n', axis=1)


def zadanie_3():
    s = np.array([2, 3.6667, 5, 7.2, 10])
    n = np.array([50, 100, 200, 500, 1000])
    s_float = np.float32(s)
    s_double = np.float64(s)

    results_rieman_float = [
        [rieman_function(s_el, n_el, np.float32) for s_el in s_float]
        for n_el in n
    ]

    results_rieman_double = [
        [rieman_function(s_el, n_el, np.float64) for s_el in s_double]
        for n_el in n
    ]

    results_rieman_reverse_float = [
        [rieman_function_reverse(s_el, n_el, np.float32) for s_el in s_float]
        for n_el in n
    ]

    results_rieman_reverse_double = [
        [rieman_function_reverse(s_el, n_el, np.float64) for s_el in s_double]
        for n_el in n
    ]

    results_dirichlet_float = [
        [dirichlet_function(s_el, n_el, np.float32) for s_el in s_float]
        for n_el in n
    ]

    results_dirichlet_double = [
        [dirichlet_function(s_el, n_el, np.float64) for s_el in s_double]
        for n_el in n
    ]

    results_dirichlet_reverse_float = [
        [dirichlet_function_reverse(s_el, n_el, np.float32) for s_el in s_float]
        for n_el in n
    ]

    results_dirichlet_reverse_double = [
        [dirichlet_function_reverse(s_el, n_el, np.float64) for s_el in s_double]
        for n_el in n
    ]

    print("\nDifference between float and double\n")
    print(f"Rieman \n"
          f"{formatted_table(s, n, get_difference(results_rieman_float, results_rieman_double))}")
    print(f"Dirichlet \n"
          f"{formatted_table(s, n, get_difference(results_dirichlet_float, results_dirichlet_double))}")

    print("\nDifference between straight and reverse adding\n")
    print(f"Rieman \n"
          f"{formatted_table(s, n, get_difference(results_rieman_reverse_float, results_rieman_float))}")
    print(f"Dirichlet\n"
          f" {formatted_table(s, n, get_difference(results_dirichlet_reverse_float, results_dirichlet_float))}")
    # print(
    #     results_rieman_float, "\n",
    #     results_rieman_reverse_float, "\n",
    #     results_rieman_double, "\n",
    #     results_rieman_reverse_double, "\n",
    #     results_dirichlet_float, "\n",
    #     results_dirichlet_reverse_float, "\n",
    #     results_dirichlet_double, "\n",
    #     results_dirichlet_reverse_double, "\n"
    # )


zadanie_3()


# ZAD 4


def rec_image(x_0, r, n):
    x_n = x_0
    for i in range(0, n):
        x_n = r * x_n * (1 - x_n)

    return x_n


def ex_a(r, x, n):
    results = [
        [rec_image(x_0, r_0, n) for x_0 in x]
        for r_0 in r
    ]

    table = DataFrame(results, r, x).rename_axis('r', axis=0).rename_axis('x', axis=1)
    return table, results


def ex_b(r, x, n):
    r_double = np.float64(r[:3])
    r_float = np.float32(r[:3])
    x_double = np.float64(x)
    x_float = np.float32(x)

    table_float, _ = ex_a(r_float, x_float, n)
    table_double, _ = ex_a(r_double, x_double, n)

    return table_float, table_double


def ex_c(r, x, eps=np.float(10**-12)):
    zero = np.float32(0)
    i = 0
    while abs(x - zero) > eps:
        x = r * x * (1 - x)
        i += 1
    return i


def zadanie_4():
    N = 10 ** 6
    r = np.array([3.75, 3.77, 3.79, 1.523, 2.321, 3.222, 3.125])
    x = np.array([0.123, 0.321, 0.51241, 0.8236])
    r_float = np.float32(r)
    x_float = np.float32(x)

    table, data = ex_a(r_float, x_float, N)
    print(table)
    print(data)

    table_float, table_double = ex_b(r, x, N)
    print('\n results for float\n')
    print(table_float)
    print('\n results for double\n')
    print(table_double)

    ex_c_results = [ex_c(np.float(4), x_el) for x_el in x_float]
    print(ex_c_results)


zadanie_4()
