import numpy as np
import pandas
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# 1. Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
data = pd.read_csv('titanic.csv')

# 2. Оставьте в выборке четыре признака: класс пассажира (Pclass),
# цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
colons=['Pclass', 'Fare', 'Age', 'Sex']
X = data.loc[:, colons]

# 3. Обратите внимание, что признак Sex имеет строковые значения.
X['Sex'] = X['Sex'].replace(to_replace=['male', 'female'], value=[1, 0])

# 4. Выделите целевую переменную — она записана в столбце Survived.
y=data['Survived']

# 5. В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст.
# Такие записи при чтении их в pandas принимают значение nan.
# Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
for i in (X['Age']):
    if np.isnan(i):
           X=X.dropna(axis=0,how='any')
y = y[X.index.values]

# 6. Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию
# (речь идет о параметрах конструктора DecisionTreeСlassifier).
my_tree = DecisionTreeClassifier(random_state=241)
my_tree.fit(np.array(X.values), np.array(y.values))

# 7. Вычислите важности признаков и найдите два признака с наибольшей важностью
importances = pandas.Series(my_tree.feature_importances_, index=colons)
print( ' '.join(importances.sort_values(ascending = False).head(2).index.values))
