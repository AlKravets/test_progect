import numpy as np
import matplotlib.pyplot as plt
import math

# Решить задачу условной оптимизации!!!! Функция штрафа


# целевая функция. Ищем минимум Функция Растригина
def f (arr , n= 2) -> float:
    s =  float(0)    
    A = 10
    for i in range(n):
        s += arr[i]**2 - A*math.cos(2*math.pi*arr[i])
    return A*n + s


# Функция Сферы
def f1 (arr , n= 2) -> float:
    s =  float(0)    
    for i in range(n):
        s += arr[i]**2
    return s

# Функция Гольдман-Прайса
def f3 (arr , n= 3) -> float:
    return  (1+ (arr[0] + arr[1] + 1)**2*(19 - 14*arr[0] +3*arr[0]**2 - 14*arr[1] +6*arr[0]*arr[1] +3*arr[1]**2 ))*(30 + (2*arr[0] -3*arr[1])**2* (18 - 32*arr[0]+ 12*arr[0]**2+ 48*arr[1] - 36*arr[0]*arr[1] + 27*arr[1]**2))

def f4 (arr , n= 2) -> float:
    return ((1-arr[0])**2 + (1-arr[1])**2 + 100*(arr[1]-arr[2]**2)**2 + 100*(arr[2] - arr[1]**2)**2)
#arr = np.array([[3.4  ,4.1], [2,4]])

# границы
a, b = -5, 5

k_k = 8

# Размерность
n=2

# Кол-во точек
s = 100


# Начальное распределение точек
X = np.around(np.random.rand(s,n)*(b-a) + a, decimals=k_k)

#print(X)

# Наилучшее положение точки
P = X

#print(P)

# Найденное оптимальное положение точки
G = X[0]

for i in range(s):
    if f(G) > f(X[i]):
        G = X[i]

print(f(G))

# ускорение
a1 , a2 = 1.5 , 1.5
w=  1
w_1 = 0.4

# Скорость (начальная)
V = np.around(a1*(np.random.rand(s,n)*2 -1)*(P-X)+a2*(np.random.rand(s,n)*2 -1)*(G-X), decimals=k_k)

#print ('\n', V)

# кол-во итераций алгоритма
l = 1000

raa  = 0
for i in range(l):
    #print(i, ' ', G, sep ='' )
    raa = float(i)
    X = X+V
    for j in range(s):
        if X[j][0] >b or X[j][0] < a or X[j][1] > b or X[j][1] < a:
            X[j] = P[j]
        if f(G) > f(X[j]):
            G = X[j]
        if f(P[j]) > f(X[j]):
            P[j] = X[j]
    
    V = np.around((w - raa/l*(w-w_1))*V + a1*(np.random.rand(s,n)*2 -1)*(P-X)+a2*(np.random.rand(s,n)*2 -1)*(G-X), decimals=k_k)

print(raa)
print(f(G))
print(G) 

raa = 0

g1 , g2= -0.1 , 0.1
g11 , g22  = -0.1 , 0.1

for j in range(s):
    if X[j][0] >g1 and X[j][0] < g2 and X[j][1] > g11 and X[j][1]  < g22:
        raa = raa + 1
print(raa)



'''fig = plt.figure()
print (fig.axes)

print(type(fig))

plt.scatter(1.0, 1.0)

print(fig.axes)



plt.show() '''


