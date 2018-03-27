# Понижение размерности. Метод главных компонент.
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# 1. Загрузите данные close_prices.csv. В этом файле приведены цены акций 30 компаний на закрытии торгов за каждый день периода.
data = pd.read_csv('close_prices.csv')

# 2. На загруженных данных обучите преобразование PCA с числом компоненты равным 10.
pca=PCA(n_components=10)
data_pr=data.drop(['date'],axis=1)
pca.fit(data_pr.values)

# Скольких компонент хватит, чтобы объяснить 90% дисперсии?
disp_sum=0
num_comp=0
for disp in pca.explained_variance_ratio_:
    disp_sum+=disp
    num_comp+=1
    if disp_sum>=0.9:
        break
print('disp>=0.9 with',num_comp,'components')

# 3. Примените построенное преобразование к исходным данным и возьмите значения первой компоненты.
components=pd.DataFrame(pca.transform(data_pr))
comp1=components[0]

# 4. Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv.
# Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
Dow_Jones_data=pd.read_csv('djia_index.csv')
Dow_Jones_index=Dow_Jones_data['^DJI']
#print(Dow_Jones_index)
corr=np.corrcoef(Dow_Jones_index,comp1)
print('correlation DJI and 1 component =',round(corr[0,1],2))

# 5. Какая компания имеет наибольший вес в первой компоненте?
# Укажите ее название с большой буквы.
comp1_ = pd.Series(pca.components_[0])
comp1_top = comp1_.sort_values(ascending=False).head(1).index[0]
name_company = data_pr.columns[comp1_top]
print('company with max 1 component:',name_company)
