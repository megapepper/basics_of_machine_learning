#метод K ближайших соседей, выбор оптимального по качеству К, как влияет нормализация признаков
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('wine.txt')
# Извлеките из данных признаки и классы.
# Класс записан в первом столбце (три варианта), признаки — в столбцах со второго по последний.
data1=data['Class']
data2=data.drop(['Class'],axis=1)

razb=KFold(n_splits=5,shuffle=True,random_state=42)

qual=[0 for i in range(50)]
qual_scaled=[0 for i in range(50)]
for k in range(1,50):
    sum=0
    sum_scaled=0
    kNN_classify=KNeighborsClassifier(n_neighbors=k)
    for train, test in razb.split(data2):
        train2=data2.filter(train,axis=0)
        train1=data1.filter(train,axis=0)
        kNN_classify.fit(train2,train1)
        rep=kNN_classify.predict(data2.filter(test,axis=0))
        sum=sum+accuracy_score(data1.filter(test,axis=0),rep)

        S=StandardScaler()
        train2_scaled=S.fit_transform(train2)
        test_scaled=S.transform(data2.filter(test,axis=0))
        kNN_classify.fit(train2_scaled,train1)
        rep_scaled=kNN_classify.predict(test_scaled)
        sum_scaled=sum_scaled+accuracy_score(data1.filter(test,axis=0),rep_scaled)
    qual[k-1]=sum/5
    qual_scaled[k-1]=sum_scaled/5

MAX_qual=np.max(qual)
MAX_qual_scaled=np.max(qual_scaled)
K_max=np.argmax(qual)+1
K_max_scaled=np.argmax(qual_scaled)+1
print('k=',K_max,';','quality=',round(MAX_qual,2))
print('k_scaled=',K_max_scaled,';','quality=',round(MAX_qual_scaled,2))
