import numpy as np
import matplotlib.pyplot as plt


def closest_node(data, t, map):
    m_rows = len(map)
    m_cols = len(map[0])
    vec = data[t]
    min =1.7976931348623158E+308
    x = 0
    y = 0
    for i in range(m_rows):
        for j in range(m_cols):
            dist = euc_dist(vec, map[i][j])
            if dist < min:
                min = dist
                x = i
                y = j
    return x, y


def euc_dist(v1, v2):
    sum = 0
    for i in range(len(v1)):
        sum = sum+pow((v1[i]-v2[i]),2)
    res = np.sqrt(sum)
    return res


def man_dist(n1_x, n1_y, n2_x, n2_y):
    return abs(n1_x - n2_x) + abs(n1_y - n2_y)


def som(data: np.array, max_iter: int, learning_rate: float, rows: int, cols: int, features: int):
    if max_iter > len(data):
        print("number of iteration is bigger then the data size")
        return
    range_max = rows+cols
    map = np.random.rand(rows, cols, features)
    for iter in range(max_iter):
        bmu_row, bmu_col = closest_node(data, iter, map)

        pct_left = 1.0 - ((iter * 1.0) / max_iter)
        curr_range = (int)(pct_left * range_max)
        curr_rate = pct_left * learning_rate

        for i in range(rows):
            for j in range(cols):
                if man_dist(bmu_row, bmu_col, i, j) < curr_range:
                    map[i][j] = map[i][j] + curr_rate * (data[iter] - map[i][j])

    return map


def plot_map(map):
    x = []
    y = []
    for rows in range(len(map)):
        for cols in range(len(map[0])):
            x.append(map[rows][cols][0])
            y.append(map[rows][cols][1])

    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    # prepare the data
    size = 1000
    data = []
    for i in range(size):
        d = np.array(np.random.uniform(0, 1, 2))
        data.append(d)
    print(data)
    map = som(data, 500, 0.5, 15, 15, 2)
    # plot_map(map)

    # change data
    data = []
    i = 0
    while i < size:
        d = np.array(np.random.uniform(-1, 1, 2))
        if 1 <= d[0]**2 + d[1]**2 <= 2:
            data.append([d[0], d[1]])
            i += 1
    print(data)

    map = som(data, 500, 0.5, 15, 15, 2)
    plot_map(map)