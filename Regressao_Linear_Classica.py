import pandas as pd
import numpy as np

x = np.random.default_rng().normal(0, 1, 1000)
y = np.random.default_rng().normal(0, 3, 1000)


class regressao_linear():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        if len(self.x) != len(self.y):
            raise Exception("As arrays precisam ter o mesmo tamanho")

    def train(self):
        self.n = len(self.x)
        self.soma_xy = sum(xi * yi for xi, yi in zip(self.x, self.y))
        self.soma_x = sum(self.x)
        self.soma_y = sum(self.y)
        self.soma_x_quadrado = sum(xi ** 2 for xi in self.x)

        self.coeficiente = (self.n * self.soma_xy - self.soma_x * self.soma_y) / (
            self.n * self.soma_x_quadrado - self.soma_x ** 2
        )
        self.intercepto = (self.soma_y - self.coeficiente * self.soma_x) / self.n

        print("Treino Finalizado")

    def predict(self, valor):
        return self.intercepto + self.coeficiente * valor


reg = regressao_linear(x, y)
reg.train()
print(reg.predict(0.1))
