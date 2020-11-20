import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc


class Ellipse:
    # coefficients of the ellipse
    a = 100
    b = 1

    # graph max
    max_x = 20
    max_y = 6

    # drawing size
    figsize=(20, 6)

    # hyper parameters
    learning_rate = 0.95
    beta = 0.5               # for exponential_weighted_averages
    initial_point = np.array([-19., 5.])
    steps = 50

    bias_correction = True

    def forward(self, points):
        return points[:, 0] ** 2 / self.a + points[:, 1] ** 2 / self.b

    def backward(self, point):
        return np.array([point[0] * 2 / self.a, point[1] * 2 / self.b])

    def plot_contour(self, axis):
        x_min, x_max = - self.max_x, self.max_x
        y_min, y_max = - self.max_y, self.max_y
        h = 100
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, h), np.linspace(y_min, y_max, h))
        data = np.c_[xx.ravel(), yy.ravel()]

        Z = self.forward(data)
        Z = np.array(Z).reshape(xx.shape)

        cont = axis.contour(xx, yy, Z, colors=[(0.7,0.7,0.7)], levels=[0.2, 1,5,10,15,20,30])
        cont.clabel(fmt='%1.1f', fontsize=14)

    def plot_gradient_descent(self, axis):
        p = self.initial_point
        points = np.array(p)

        for i in range(self.steps):
            p = self.next(p)
            points = np.append(points, p)

        points = points.reshape(-1, 2)
        N = points.shape[0]

        ewa = exponential_weighted_averages(points, self.beta, self.bias_correction)

        lines1 = [[points[i], points[i+1]] for i in range(N - 1)]
        lines2 = [[ewa[i], ewa[i+1]] for i in range(N - 1)]

        lc1 = mc.LineCollection(lines1, colors=[(0,1,0)], linewidths=4)
        lc2 = mc.LineCollection(lines2, colors=(1,0,0))
        axis.add_collection(lc1)
        axis.add_collection(lc2)

    def next(self, p):
        df = self.backward(p)
        return p - self.learning_rate * df

    def plot(self):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        plt.ylabel('y')
        plt.xlabel('x')

        self.plot_contour(ax)
        self.plot_gradient_descent(ax)

        plt.show()

def exponential_weighted_averages(steps, beta, bias_correction = True):
    v = np.zeros((1,2))
    avg = np.empty(0)

    for i, p in enumerate(steps):
        v = v * beta + (1 - beta) * p
        correction = (1 - beta ** (i + 1)) if bias_correction else 1
        avg = np.append(avg, v / correction)

    return avg.reshape(-1, 2)


def main():
    ellipse = Ellipse()
    ellipse.plot()

main()
