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

    # buffer for exponentially weighted averages
    v = np.array([0, 0])

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

    def plot_gradient_descent(self, axis, colors, linewidths, *, momentum = False):
        p = self.initial_point
        points = np.array(p)

        for i in range(self.steps):
            p = self.next(p, i + 1, momentum=momentum)
            points = np.append(points, p)

        points = points.reshape(-1, 2)
        N = points.shape[0]

        lines1 = [[points[i], points[i+1]] for i in range(N - 1)]
        lc1 = mc.LineCollection(lines1, colors=colors, linewidths=linewidths)
        axis.add_collection(lc1)

    def next(self, p, current_step, *, momentum):
        df = self.backward(p)
        if momentum:
            correction = (1 - self.beta ** current_step) if self.bias_correction else 1
            self.v = self.v * self.beta + df * (1 - self.beta) / correction
            df = self.v

        return p - self.learning_rate * df

    def plot(self):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        plt.ylabel('y')
        plt.xlabel('x')

        self.plot_contour(ax)
        self.plot_gradient_descent(ax, [(0,1,0)], 4)
        self.plot_gradient_descent(ax, [(1,0,0)], 1, momentum = True)

        plt.show()

def main():
    ellipse = Ellipse()
    ellipse.plot()

main()
