import random
import numpy as np
from matplotlib import pyplot as plt

grad = None
b = None


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
        constant = 0
        threshold = 1000
        weights = [1 for i in range(m)]
        threshold = 1000
        while threshold > 0:
            for row in arr:
                h = np.dot(weights, row[:m])
                if sign(h+constant) != sign(row[-1]):
                    threshold -= 1
                    p = random.uniform(0, 1)
                    weights = weights + p * row[-1] * row[:m]
                    constant = -1 * (h * p + constant * (1 - p))
                    err.append(row)
            if not err:
                self.weights = weights
                self.constant = constant
                return True
            while err:
                row = err.pop(0)
                h = np.dot(weights, row[:m])
                if sign(h+constant) != sign(row[-1]):
                    threshold -= 1
                    p = random.uniform(0, 1)
                    weights = weights + p * row[-1] * row[:m]
                    constant = -1 * (h + constant) / 2
                    err.append(row)
                    if threshold < 0:
                        self.weights = weights
                        self.constant = constant
                        return False

        return True

    def predict(self, arr):
        sign = lambda x: 1 if x > 0 else -1
        m = self.m - 1
        return np.array([sign(np.dot(self.weights, row[:m])+self.constant) for row in arr])

    def check(self, arr, weights, constant):
        sign = lambda x: 1 if x > 0 else -1
        m = self.m - 1
        return np.array([sign(np.dot(weights, row[:m]) + constant) for row in arr])


# 随意选择，参数是为了图好看
def dataSet(w_1, w_2, turns=200):
    global b
    global grad
    constant = 20
    b = constant
    print('Fact:', w_1, w_2, constant)
    r = lambda: random.choice((1, -1))*np.random.normal(constant+8, 3, 1)[0]
    w_x = lambda x_1, x_2: w_1*x_1+w_2*x_2
    grad = (w_1, w_2)
    constant *= -1
    for i in range(turns):
        x = r()
        y = r()
        f = w_x(x, y)
        if f > constant:
            yield [x, y, 1]
        elif f != constant:
            yield [x, y, -1]


R = lambda: random.randint(-10, 10)
w_1, w_2, w_3 = R(), R(), R()
DS = np.array([vec for vec in dataSet(w_1, w_2)])
vm = PLA()
print('Separable?', vm.fit(DS))
vm.predict(DS)
p_preds = vm.predict(DS)
weights = vm.weights
constant = vm.constant
print('Hypothesis:', weights, constant)
positive = [[], []]
negative = [[], []]
errs = 0
for row, pred in zip(DS, p_preds):
    if row[-1] != pred:
        print(weights*row[:vm.m-1]+constant, pred, row)
        errs += 1
    if pred == 1:
        positive[0].append(row[0])
        positive[-1].append(row[1])
    else:
        negative[0].append(row[0])
        negative[-1].append(row[1])
print(errs)
print('{:.2%}'.format(errs/200))
y_2_x = lambda gradint, x, c: -(gradint[0]/gradint[-1])*x-c/gradint[-1]
f_line = [[], []]
h_line = [[], []]
for i in (-22, 22):
    f_line[0].append(i)
    f_line[-1].append(y_2_x(grad, i, b))
    h_line[0].append(i)
    h_line[-1].append(y_2_x(weights, i, constant))
plt.scatter(positive[0], positive[-1], c='b')
plt.scatter(negative[0], negative[-1], c='r', marker='^')
plt.plot(f_line[0], f_line[-1], c='b')
plt.plot(h_line[0], h_line[-1], c='r')
plt.show()
