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


def som(data: np.array, max_iter: int, learning_rate: float, rows: int, cols: int, features: int, show_prog=False, desc="half iter"):
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

        if show_prog and iter == max_iter//2:
            plt.gca().set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
            plt.title(desc)
            plot_map(map, data)

    return map


def plot_map(map, data):
    # plot data
    x = []
    y = []
    for cell in data:
        x.append(cell[0])
        y.append(cell[1])
    plt.scatter(x, y)

    # plot neurons
    x = []
    y = []
    for rows in range(len(map)):
        for cols in range(len(map[0])):
            x.append(map[rows][cols][0])
            y.append(map[rows][cols][1])
    plt.scatter(x, y)
    plt.show()


def get_data(bottom, upper, lim, size, nu=0, orig=False):
    data = []
    i = 0
    while i < size:
        if nu == 0:
            d = np.array(np.random.uniform(-lim, lim, 2))
        elif nu == 1:
            d = np.array(np.random.standard_normal(2))
        elif nu == 2:
            d = np.array(np.random.laplace(np.mean([upper, bottom]), .075, 2))
            data.append(d)
            i += 1
            continue
        if bottom <= d[0] ** 2 + d[1] ** 2 <= upper:
            if orig:
                data.append([d[0], d[1]])
            else:
                data.append([d[0]/2+.5, d[1]/2+.5])
            i += 1
    return data


def part_a_uniform(size, bot, top):
    lim = 1.5
    data = get_data(bot, top, lim, size)

    map = som(data, 500, 0.5, 3, 5, 2)
    plt.gca().set(xlim=(-bot-0.1, top+.1), ylim=(-bot-0.1, top+.1))
    plt.title("15 neurons 500 iter")
    plot_map(map, data)

    map = som(data, 1000, 0.5, 3, 5, 2)
    plt.gca().set(xlim=(-bot-0.1, top+.1), ylim=(-bot-0.1, top+.1))
    plt.title("15 neurons 1000 iter")
    plot_map(map, data)

    map = som(data, 500, 0.5, 10, 20, 2)
    plt.gca().set(xlim=(-bot-0.1, top+.1), ylim=(-bot-0.1, top+.1))
    plt.title("200 neurons 500 iter")
    plot_map(map, data)

    map = som(data, 1000, 0.5, 10, 20, 2)
    plt.gca().set(xlim=(-bot-.1, top+.1), ylim=(-bot-.1, top+.1))
    plt.title("200 neurons 1000 iter")
    plot_map(map, data)


def part_a_nonuniform1(size, bot, top):
    # change data
    lim = 1.5
    data = get_data(bot, top, lim, size, nu=1)

    map = som(data, 500, 0.5, 3, 5, 2)
    plt.gca().set(xlim=(-.1, 1.1), ylim=(-.1, 1.1))
    plt.title("standard normal 15 neurons 500 iter")
    plot_map(map, data)

    map = som(data, 1000, 0.5, 3, 5, 2)
    plt.gca().set(xlim=(-.1, 1.1), ylim=(-.1, 1.1))
    plt.title("standard normal 15 neurons 1000 iter")
    plot_map(map, data)

    map = som(data, 500, 0.5, 10, 20, 2)
    plt.gca().set(xlim=(-.1, 1.1), ylim=(-.1, 1.1))
    plt.title("standard normal 200 neurons 500 iter")
    plot_map(map, data)

    map = som(data, 1000, 0.5, 10, 20, 2)
    plt.gca().set(xlim=(-.1, 1.1), ylim=(-.1, 1.1))
    plt.title("standard normal 200 neurons 1000 iter")
    plot_map(map, data)


def part_a_nonuniform2(size, bot, top):
    lim = 1.5
    data = get_data(bot, top, lim, size, nu=2)

    map = som(data, 500, 0.5, 3, 5, 2)
    plt.gca().set(xlim=(-.1, 1.1), ylim=(-.1, 1.1))
    plt.title("laplace 15 neurons 500 iter")
    plot_map(map, data)

    map = som(data, 1000, 0.5, 3, 5, 2)
    plt.gca().set(xlim=(-.1, 1.1), ylim=(-.1, 1.1))
    plt.title("laplace 15 neurons 1000 iter")
    plot_map(map, data)

    map = som(data, 500, 0.5, 10, 20, 2)
    plt.gca().set(xlim=(-.1, 1.1), ylim=(-.1, 1.1))
    plt.title("laplace 200 neurons 500 iter")
    plot_map(map, data)

    map = som(data, 1000, 0.5, 10, 20, 2)
    plt.gca().set(xlim=(-.1, 1.1), ylim=(-.1, 1.1))
    plt.title("laplace 200 neurons 1000 iter")
    plot_map(map, data)


def part_a_2(size, bot, top):
    lim = 1.5
    data = get_data(bot, top, lim, size, orig=True)

    map = som(data, 500, 0.5, 5, 6, 2, show_prog=True, desc="30 neurons 250 of 500 iter")
    plt.gca().set(xlim=(-lim, lim), ylim=(-lim, lim))
    plt.title("30 neurons 500 iter")
    plot_map(map, data)

    map = som(data, 1000, 0.5, 5, 6, 2)
    plt.gca().set(xlim=(-lim, lim), ylim=(-lim, lim))
    plt.title("30 neurons 1000 iter")
    plot_map(map, data)

    map = som(data, 500, 0.5, 10, 20, 2)
    plt.gca().set(xlim=(-lim, lim), ylim=(-lim, lim))
    plt.title("200 neurons 500 iter")
    plot_map(map, data)

    map = som(data, 1000, 0.5, 10, 20, 2)
    plt.gca().set(xlim=(-lim, lim), ylim=(-lim, lim))
    plt.title("200 neurons 1000 iter")
    plot_map(map, data)


if __name__ == '__main__':
    size = 1000
    part_a_uniform(size, 0, 1)
    part_a_nonuniform1(size, 0, 1)
    part_a_nonuniform2(size, 0, 1)
    part_a_2(size, 1, 2)
