import numpy as np
import matplotlib.pyplot as plt


class SOM:
    def __init__(self):
        pass

    def closest_node(self, data, t, map):
        m_rows = len(map)
        m_cols = len(map[0])
        vec = data[t]
        min_val = 1.7976931348623158E+308
        x = 0
        y = 0
        for i in range(m_rows):
            for j in range(m_cols):
                dist = self.euc_dist(vec, map[i][j])
                if dist < min_val:
                    min_val = dist
                    x = i
                    y = j
        return x, y

    def euc_dist(self, v1, v2):
        sum = 0
        for i in range(len(v1)):
            sum = sum+pow((v1[i]-v2[i]),2)
        res = np.sqrt(sum)
        return res

    def man_dist(self, n1_x, n1_y, n2_x, n2_y):
        return abs(n1_x - n2_x) + abs(n1_y - n2_y)

    def som(self, data: np.array, max_iter: int, learning_rate: float, rows: int, cols: int, features: int, show_prog=False, desc="half iter", limit=True):
        if max_iter > len(data):
            print("number of iteration is bigger then the data size")
            return
        range_max = rows+cols
        map = np.random.rand(rows, cols, features)
        for iter in range(max_iter):
            bmu_row, bmu_col = self.closest_node(data, iter, map)

            pct_left = 1.0 - ((iter * 1.0) / max_iter)
            curr_range = (int)(pct_left * range_max)
            curr_rate = pct_left * learning_rate

            for i in range(rows):
                for j in range(cols):
                    if self.man_dist(bmu_row, bmu_col, i, j) < curr_range:
                        map[i][j] = map[i][j] + curr_rate * (data[iter] - map[i][j])

            if show_prog and iter == max_iter//2:
                if limit:
                    plt.gca().set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
                plt.title(desc)
                self.plot_map(map, data)

        return map

    def plot_map(self, map, data):
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
