#выбор оптимальной метрики
import numpy
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# 1. Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
# Результатом вызова данной функции является объект, у которого признаки записаны в поле data,
# а целевой вектор — в поле target.
Boston=load_boston()

# 2. Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.tandardScaler.
S=StandardScaler()
data_scaled=S.fit_transform(Boston.data)

razb=KFold(n_splits=5,shuffle=True,random_state=42)
Max_qual=0
ind=0

# 3. Переберите разные варианты параметра
# метрики p по сетке от 1 до 10 с таким шагом, чтобы всего было протестировано 200 вариантов
for i in numpy.linspace(1,10,200):
    sum=0
    for train, test in razb.split(data_scaled):
        kNR=KNeighborsRegressor(n_neighbors=5,weights='distance',p=i)
        train2=data_scaled[train]
        train1=Boston.target[train]
        kNR.fit(train2,train1)
        rep=kNR.predict(data_scaled[test])

        # 4. Определите, при каком p качество на кросс-валидации оказалось оптимальным.
        # Обратите внимание, что cross_val_score возвращает массив показателей качества по блокам;
        # необходимо максимизировать среднее этих показателей
        qual=numpy.mean(cross_val_score(kNR,data_scaled[test],Boston.target[test],cv=5))
        sum=sum+qual
    if sum/5>Max_qual:
        Max_qual=sum/5
        P=i
print('P_opt=',round(P,1),'Max_quality=',round(Max_qual,2))
