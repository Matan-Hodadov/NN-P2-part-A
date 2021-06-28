import numpy as np
import matplotlib.pyplot as plt
import SOM


def get_data(bottom, upper, size, step, y_bottom, y_upper, x_left, x_right):
    data = []
    i = 0
    while i < size:
        d = np.array(np.random.uniform(bottom, upper, 2))
        normal_x = (int)(d[0]*step)-step
        normal_x = size-1 if normal_x >= size else normal_x
        if x_left[0] <= d[0] <= x_right[0] and y_bottom[0] <= d[1] <= y_upper[normal_x]:
            data.append([d[0], d[1]])
            i += 1
    return data


def full_hand():
    som = SOM.SOM()
    
    y_upper_bounds_by_x = np.zeros(SIZE)
    y_bottom_bounds_by_x = np.zeros(SIZE)
    x_left_bounds_by_y = np.zeros(SIZE)
    x_right_bounds_by_y = np.zeros(SIZE)
    # first hand
    x = np.linspace(1, 9, SIZE)
    y_bottom_bounds_by_x = np.ones(SIZE)

    y = np.linspace(1, 5, SIZE)
    x_left_bounds_by_y = np.ones(SIZE)

    y = np.linspace(1, 5, SIZE)
    x_right_bounds_by_y = 9 * np.ones(SIZE)

    x = np.linspace(1, 2, STEP)
    y_upper_bounds_by_x[0: STEP] = 7 * x - 2

    x = np.linspace(2, 3, STEP)
    y_upper_bounds_by_x[STEP: 2 * STEP] = -4 * x + 20

    x = np.linspace(3, 4, STEP)
    y_upper_bounds_by_x[2 * STEP: 3 * STEP] = 5 * x - 7

    x = np.linspace(4, 5, STEP)
    y_upper_bounds_by_x[3 * STEP: 4 * STEP] = -5 * x + 33

    x = np.linspace(5, 6, STEP)
    y_upper_bounds_by_x[4 * STEP: 5 * STEP] = 4 * x - 12

    x = np.linspace(6, 7, STEP)
    y_upper_bounds_by_x[5 * STEP: 6 * STEP] = -4 * x + 36

    x = np.linspace(7, 8, STEP)
    y_upper_bounds_by_x[6 * STEP: 7 * STEP] = 2 * x - 6

    x = np.linspace(8, 9, SIZE-7*STEP)
    y_upper_bounds_by_x[7 * STEP:] = -5 * x + 50

    plt.plot(X_RANGE, y_upper_bounds_by_x, 'r')
    plt.plot(X_RANGE, y_bottom_bounds_by_x, 'r')
    plt.plot(x_left_bounds_by_y, Y_RANGE, 'r')
    plt.plot(x_right_bounds_by_y, Y_RANGE, 'r')

    data = get_data(1, 13, 1000, STEP, y_bottom_bounds_by_x, y_upper_bounds_by_x, x_left_bounds_by_y,
                    x_right_bounds_by_y)

    map = som.som(data, 500, 0.5, 5, 6, 2, show_prog=True, desc="30 neurons 250 of 500 iter", limit=False)

    plt.title("30 neurons 500 iter")
    plt.plot(X_RANGE, y_upper_bounds_by_x, 'r')
    plt.plot(X_RANGE, y_bottom_bounds_by_x, 'r')
    plt.plot(x_left_bounds_by_y, Y_RANGE, 'r')
    plt.plot(x_right_bounds_by_y, Y_RANGE, 'r')
    som.plot_map(map, data)


def modded_hand():
    som = SOM.SOM()

    y_upper_bounds_by_x = np.zeros(SIZE)
    y_bottom_bounds_by_x = np.zeros(SIZE)
    x_left_bounds_by_y = np.zeros(SIZE)
    x_right_bounds_by_y = np.zeros(SIZE)
    # first hand
    x = np.linspace(1, 9, SIZE)
    y_bottom_bounds_by_x = np.ones(SIZE)

    y = np.linspace(1, 5, SIZE)
    x_left_bounds_by_y = np.ones(SIZE)

    y = np.linspace(1, 5, SIZE)
    x_right_bounds_by_y = 9 * np.ones(SIZE)

    x = np.linspace(1, 2, STEP)
    y_upper_bounds_by_x[0: STEP] = 7 * x - 2

    x = np.linspace(2, 3, STEP)
    y_upper_bounds_by_x[STEP: 2 * STEP] = -4 * x + 20

    x = np.linspace(3, 5, 2*STEP)
    y_upper_bounds_by_x[2 * STEP: 4 * STEP] = 8 * np.ones(2 * STEP)

    x = np.linspace(5, 6, STEP)
    y_upper_bounds_by_x[4 * STEP: 5 * STEP] = 4 * x - 12

    x = np.linspace(6, 7, STEP)
    y_upper_bounds_by_x[5 * STEP: 6 * STEP] = -4 * x + 36

    x = np.linspace(7, 8, STEP)
    y_upper_bounds_by_x[6 * STEP: 7 * STEP] = 2 * x - 6

    x = np.linspace(8, 9, SIZE-7*STEP)
    y_upper_bounds_by_x[7 * STEP:] = -5 * x + 50

    plt.plot(X_RANGE, y_upper_bounds_by_x, 'r')
    plt.plot(X_RANGE, y_bottom_bounds_by_x, 'r')
    plt.plot(x_left_bounds_by_y, Y_RANGE, 'r')
    plt.plot(x_right_bounds_by_y, Y_RANGE, 'r')

    data = get_data(1, 13, 1000, STEP, y_bottom_bounds_by_x, y_upper_bounds_by_x, x_left_bounds_by_y, x_right_bounds_by_y)

    map = som.som(data, 500, 0.5, 5, 6, 2, show_prog=True, desc="30 neurons 250 of 500 iter", limit=False)

    plt.title("30 neurons 500 iter")
    plt.plot(X_RANGE, y_upper_bounds_by_x, 'r')
    plt.plot(X_RANGE, y_bottom_bounds_by_x, 'r')
    plt.plot(x_left_bounds_by_y, Y_RANGE, 'r')
    plt.plot(x_right_bounds_by_y, Y_RANGE, 'r')
    som.plot_map(map, data)#, x_plots = [x_left_bounds_by_y, x_right_bounds_by_y])

    plt.show()


if __name__ == '__main__':
    SIZE = 1000
    X_MIN = 1
    X_MAX = 9
    X_RANGE = np.linspace(X_MIN, X_MAX, SIZE)
    Y_RANGE = np.linspace(X_MIN, 5, SIZE)
    STEP = 1000 // (X_MAX - X_MIN)

    full_hand()
    modded_hand()

    """
    # second hand
    x = np.linspace(1, 9, size)
    plt.plot(x, np.ones(size), 'b')

    x = np.linspace(1, 2, size)
    plt.plot(x, 7*x-2, 'b')

    x = np.linspace(2, 3, size)
    plt.plot(x, -4 * x + 20, 'b')

    x = np.linspace(3, 5, size)
    plt.plot(x, 8 * np.ones(size), 'b')

    x = np.linspace(5, 6, size)
    plt.plot(x, 4 * x - 12, 'b')

    x = np.linspace(6, 7, size)
    plt.plot(x, -4 * x + 36, 'b')

    x = np.linspace(7, 8, size)
    plt.plot(x, 2 * x - 6, 'b')

    x = np.linspace(8, 9, size)
    plt.plot(x, -5 * x + 50, 'b')

    y = np.linspace(1, 5, size)
    plt.plot(np.ones(size), y, 'b')

    y = np.linspace(1, 5, size)
    plt.plot(9*np.ones(size), y, 'b')

    plt.show()
    """

