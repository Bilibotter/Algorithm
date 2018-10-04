import random
import numpy as np
from matplotlib import pyplot as plt

grad = None


class PLA():
    def fit(self, arr):
        """
        :param arr:
        :return: 若返回True则在阈值内可收敛，False则不能
        """
        sign = lambda x: 1 if x > 0 else -1
        err = []
        n, m = arr.shape
        self.n = n
        self.m = m
        m -= 1
        threshold = 1000
        weights = [1 for i in range(m)]
        for row in arr:
            h = np.dot(weights, row[:m])
            if sign(h) != sign(row[-1]):
                weights = weights + row[-1]*row[:m]
                err.append(row)

        while err:
            row = err.pop(0)
            h = np.dot(weights, row[:m])
            if sign(h) != sign(row[-1]):
                weights = weights + row[-1] * row[:m]
                err.append(row)
                if not threshold:
                    self.weights = weights
                    return False
        self.weights = weights
        return True

    def predict(self, arr):
        sign = lambda x: 1 if x > 0 else -1
        m = self.m - 1
        return np.array([sign(np.dot(self.weights, row[:m])) for row in arr])


# 随意选择，参数是为了图好看
def dataSet(w_1, w_2, turns=200):
    global grad
    print('Fact:', w_1, w_2)
    r = lambda: random.choice((1, -1))*np.random.normal(8, 3, 1)[0]
    w_x = lambda x_1, x_2: w_1*x_1+w_2*x_2
    grad = (w_1, w_2)
    for i in range(turns):
        x = r()
        y = r()
        f = w_x(x, y)
        if f > 0:
            yield [x, y, 1]
        elif f != 0:
            yield [x, y, -1]


R = lambda: random.randint(-10, 10)
w_1, w_2, w_3 = R(), R(), R()
DS = np.array([vec for vec in dataSet(w_1, w_2)])
vm = PLA()
print('Separable?', vm.fit(DS))
vm.predict(DS)
p_preds = vm.predict(DS)
weights = vm.weights
print('Hypothesis:', weights)
positive = [[], []]
negative = [[], []]
for row, pred in zip(DS, p_preds):
    if row[-1] != pred:
        print('wrr', row)
    if pred == 1:
        positive[0].append(row[0])
        positive[-1].append(row[1])
    else:
        negative[0].append(row[0])
        negative[-1].append(row[1])
y_2_x = lambda gradint, x: -(gradint[0]/gradint[-1])*x
f_line = [[], []]
h_line = [[], []]
for i in (-22, 22):
    f_line[0].append(i)
    f_line[-1].append(y_2_x(grad, i))
    h_line[0].append(i)
    h_line[-1].append(y_2_x(weights, i))
plt.scatter(positive[0], positive[-1], c='b')
plt.scatter(negative[0], negative[-1], c='r', marker='^')
plt.plot(f_line[0], f_line[-1], c='b')
plt.plot(h_line[0], h_line[-1], c='r')
plt.show()
