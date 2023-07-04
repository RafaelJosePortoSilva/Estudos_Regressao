import pandas as pd
import numpy as np
from statistics import mean

x = np.random.default_rng().normal(0,1,1000)
y = np.random.default_rng().normal(0,3,1000)



class regressao_linear():
    def __init__(self,x,y):
        self.x=x
        self.y=y
        if len(self.x) != len(self.y):
            raise "As arrays precisam ter o mesmo tamanho"
    
    def train(self):
        self.num = len(x)
        self.xy_sum = []
        self.x_sum = []
        self.y_sum = []
        self.x_quad = []

        for self.n in range(self.num):
            self.xy_sum.append(self.x[self.n]*self.y[self.n])
            self.x_sum.append(self.x[self.n])
            self.y_sum.append(self.y[self.n])
            self.x_quad.append(self.x[self.n]**2)
        
        self.soma_xy = sum(self.xy_sum)
        self.soma_x = sum(self.x_sum)
        self.soma_y = sum(self.y_sum)
        self.soma_x_quadrado = sum(self.x_quad)

        self.intercepto = ((self.n*self.soma_xy)-(self.soma_x*self.soma_y))/(self.n*self.soma_x_quadrado-self.soma_x**2)
        self.coeficiente = mean(self.y) - self.intercepto*mean(self.x)

        print('Treino Finalizado')
    
    def predict(self,valor):
        return self.intercepto + self.coeficiente*valor
    



reg = regressao_linear(x,y)
reg.train()
print(reg.predict(0.1))