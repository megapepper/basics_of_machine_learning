#РАЗМЕР СЛУЧАЙНОГО ЛЕСА
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# 1. Загрузите данные из файла abalone.csv.
# Это датасет, в котором требуется предсказать возраст ракушки (число колец) по физическим измерениям.
data = pd.read_csv('abalone.csv',header=0)

# 2. Преобразуйте признак Sex в числовой: значение F должно перейти в -1, I — в 0, M — в 1.
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

# 3. Разделите содержимое файлов на признаки и целевую переменную.
# В последнем столбце записана целевая переменная, в остальных — признаки.
n=data.shape[1]
y_index=data.columns[n-1]
y=data[y_index]
X=data.drop(y_index,axis=1)

# 4. Обучите случайный лес (sklearn.ensemble.RandomForestRegressor) с различным числом деревьев: от 1 до 50
# Для каждого из вариантов оцените качество работы полученного леса на кросс-валидации по 5 блокам.
# В качестве меры качества воспользуйтесь коэффициентом детерминации (sklearn.metrics.r2_score).
razb=KFold(shuffle=True,random_state=1, n_splits=5)
Qual=[0 for i in range(51)]
flag=0
for n in range(1,51):
    clf = RandomForestRegressor(n_estimators=n,random_state=1)
    sum=0
    for train, test in razb.split(X):
        X_train=X.filter(train,axis=0)
        y_train=y.filter(train,axis=0)
        clf.fit(X_train,y_train)
        rep=clf.predict(X.filter(test,axis=0))
        qual=r2_score(y.filter(test,axis=0),rep)
        sum=sum+qual
    Qual[n]=sum/5

    # 5. Определите, при каком минимальном количестве деревьев случайный лес показывает качество на кросс-валидации выше 0.52.
    # Это количество и будет ответом на задание.
    if flag==0:
        if Qual[n]>0.52:
            flag=1
            N=n
print('qual>0.52, min_tree_count=',N)
