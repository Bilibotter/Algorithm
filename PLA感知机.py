import numpy as np
from matplotlib import pyplot as plt


class PLA():
    def dataSet(self, num=200, *args):
        if not args:
            b = np.random.normal(10, 10, 1)[0]
            wegihts = np.random.normal(1, 1, 3)
        else:
            *wegihts, b = args
        self.f_weights = wegihts
        self.f_b = b
        DS = []
        dim = len(wegihts)
        for i in range(num):
            point = np.random.normal(b+5, 2, dim).tolist()
            # 加入点的正负标签
            point.append(np.sign(np.dot(point, wegihts)+b))
            DS.append(point)
        return DS


p = PLA()
DS = p.dataSet()
# point = []*4 为浅拷贝
points = [[] for i in range(4)]
for point in DS:
    span = 0 if point[-1] > 0 else 2
    points[span].append(point[0])
    points[span+1].append(point[1])

plt.scatter(points[0], points[1], c='b')
plt.scatter(points[-2], points[-1], c='r', marker='^')
plt.plot((-p.f_b/p.f_weights[0], 0), (0, -p.f_b/p.f_weights[-1]))
plt.show()
