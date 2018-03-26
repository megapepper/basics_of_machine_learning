#как меняется оценка качества от нормализации признаков
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Загрузите обучающую и тестовую выборки из файлов train.csv и test.csv.
# Целевая переменная записана в первом столбце, признаки — во втором и третьем.
X_train=pd.read_csv('train.csv')
X_test=pd.read_csv('test.csv')
train1=X_train['1']
train2=X_train.drop(['1'],axis=1)
test1=X_test['1']
test2=X_test.drop(['1'],axis=1)

# 2. Обучите персептрон со стандартными параметрами и random_state=241.
Perc=Perceptron(random_state=241)

# 3. Подсчитайте качество (долю правильно классифицированных объектов, accuracy)
# полученного классификатора на тестовой выборке.
Perc.fit(train2,train1)
rep=Perc.predict(test2)
qual=accuracy_score(test1,rep)
print('no_normaliz_qual=',qual)

# 4. Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.
S=StandardScaler()
train2_scaled = S.fit_transform(train2)
test2_scaled = S.transform(test2)

# 5. Обучите персептрон на новой выборке. Найдите долю правильных ответов на тестовой выборке.
Perc.fit(train2_scaled,train1)
rep_scaled=Perc.predict(test2_scaled)
qual_scaled=accuracy_score(test1,rep_scaled)
print('normaliz_qual=',qual_scaled)
print('norm_qual-no_norm_qual=',round(qual_scaled-qual,3))
