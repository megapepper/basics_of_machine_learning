from math import exp

import pandas as pd
from sklearn.metrics import roc_auc_score

# 1. Загрузите данные из файла data-logistic.csv.
# Это двумерная выборка, целевая переменная на которой принимает значения -1 или 1.
data = pd.read_csv('data-logistic.csv',header=0)
y=data['1']
x1=data['2']
x2=data['3']
len=y.shape[0]

# 2. Убедитесь, что выше выписаны правильные формулы для градиентного спуска.
# 3.Реализуйте градиентный спуск для обычной и L2-регуляризованной (с коэффициентом регуляризации 10)
# логистической регрессии. Используйте длину шага k=0.1. В качестве начального приближения используйте вектор (0, 0).
k=0.1
c=10
e=10**(-5)
l=e+1
n=0
w1=0
w2=0
# 4. Запустите градиентный спуск и доведите до сходимости
# (евклидово расстояние между векторами весов на соседних итерациях должно быть не больше 1e-5).
#L1-регуляризация
while (n<=10000 and l>e):
    n=n+1
    sum1=0
    sum2=0
    for i in range(len):
        sum1=sum1+y[i]*x1[i]*(1-1/(1+exp(-y[i]*(w1*x1[i]+w2*x2[i]))))
        sum2=sum2+y[i]*x2[i]*(1-1/(1+exp(-y[i]*(w1*x1[i]+w2*x2[i]))))
    W1=w1+k/len*sum1
    W2=w2+k/len*sum2
    l=((W1-w1)**2+(W2-w2)**2)**(0.5)
    w1=W1
    w2=W2
a1=[0 for i in range(len)]
for i in range(len):
    a1[i] = 1 / (1 + exp((-w1*x1[i] - w2*x2[i])*y[i]))

l=e+1
n=0
w1=0
w2=0
#L2-регуляризация
while n<=10000 and l>=e:
    n=n+1
    sum1=0
    sum2=0
    for i in range(len):
        sum1=sum1+y[i]*x1[i]*(1-1/(1+exp(-y[i]*(w1*x1[i]+w2*x2[i]))))
        sum2=sum2+y[i]*x2[i]*(1-1/(1+exp(-y[i]*(w1*x1[i]+w2*x2[i]))))
    W1=w1+k/len*sum1-k*c*w1
    W2=w2+k/len*sum2-k*c*w2
    l=((W1-w1)**2+(W2-w2)**2)**(0.5)
    w1=W1
    w2=W2
a2=[0 for i in range(len)]
for i in range(len):
    a2[i] = 1 / (1 + exp((-w1*x1[i] - w2*x2[i])*y[i]))

# 5. Какое значение принимает AUC-ROC на обучении без регуляризации и при ее использовании?
qual1=roc_auc_score(y,a1)
print('AUC-ROC not reg:',round(qual1,3))
qual2=roc_auc_score(y,a2)
print('AUC-ROC with reg:',round(qual2,3))
