


import numpy as np
import matplotlib.pyplot as plt
from sortedcontainers import SortedList
from future.utils import iteritems
from sklearn.metrics import fbeta_score, make_scorer

from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles



# обучающее множество (выборка)
class KNN(object):

    # конструктор
    def __init__(self, k):
        self.k = k

    # обучение, тренировка
    def fit(self, X, y):
        self.X = X
        self.y = y

    # предсказание, вывод (определение метки класса для нового примера)
    def predict(self, XT):
        y = np.zeros(len(XT))

        # для каждого примера XT[i]
        # вычислить расстояния от XT[i] до self.X[j]
        for i, x in enumerate(XT):
            sl = SortedList()
            # по обучающей выборке
            for j, x_train in enumerate(self.X):
                diff = x - x_train
                dist = diff.dot(diff)
                if (len(sl) < self.k):
                    sl.add((dist, self.y[j]))
                else:
                    if (dist < sl[-1][0]):
                        del sl[-1]
                        sl.add((dist, self.y[j]))

            # для _k_ ближайших к XT[i] соседей:
            # определить наиболее часто встречающийся класс (C),
            # положить yt[i] = (C)

            votes = {}
            for _, v in sl:
                votes[v] = votes.get(v, 0) + 1

            max_votes = 0
            max_votes_class = -1

            for v, count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v

            y[i] = max_votes_class

        return y

    def score(self, testX, testY):
        return 0.0

knn_cls = KNN(1)



X, y = make_moons(n_samples=200, random_state=42)
plt.scatter(X[:, 0], X[:, 1], s=100, c=y, alpha=0.5)
plt.show()

X_train = X[:160]
X_test = X[160:]
y_train = y[:160]
y_test = y[160:]
plt.scatter(X_train[:, 0], X_train[:, 1], s=100, c=y_train, alpha=0.5)
plt.show()
plt.scatter(X_test[:, 0], X_test[:, 1], s=100, c=y_test, alpha=0.5)
plt.show()

#качество Тестовой выборке
knn_cls.fit(X_train, y_train)
pred = knn_cls.predict(X_test)
print(np.mean(y_test == pred))

#качество обучающей выборке
pred_train = knn_cls.predict(X_train)
print(np.mean(pred_train == y_train))


knn_cls = KNN(5)

X, y = make_circles(n_samples=150,factor=0.5,noise=0.1)
plt.scatter(X[:, 0], X[:, 1], s=100, c=y, alpha=0.5)
plt.show()

X_train = X[:120]
X_test = X[120:]
y_train = y[:120]
y_test = y[120:]
plt.scatter(X_train[:, 0], X_train[:, 1], s=100, c=y_train, alpha=0.5)
plt.show()
plt.scatter(X_test[:, 0], X_test[:, 1], s=100, c=y_test, alpha=0.5)
plt.show()

#качество Тестовой выборке
knn_cls.fit(X_train, y_train)
pred = knn_cls.predict(X_test)
print(np.mean(y_test == pred))

#качество обучающей выборке
pred_train = knn_cls.predict(X_train)
print(np.mean(pred_train == y_train))


knn_cls = KNN(3)

X, y = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0)


X_train = X[:160]
X_test = X[160:]
y_train = y[:160]
y_test = y[160:]
plt.scatter(X_train[:, 0], X_train[:, 1], s=100, c=y_train, alpha=0.5)
plt.show()
plt.scatter(X_test[:, 0], X_test[:, 1], s=100, c=y_test, alpha=0.5)
plt.show()
#качество Тестовой выборке
knn_cls.fit(X_train, y_train)
pred = knn_cls.predict(X_test)
print(np.mean(y_test == pred))

#качество обучающей выборке
pred_train = knn_cls.predict(X_train)
print(np.mean(pred_train == y_train))