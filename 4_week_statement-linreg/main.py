import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

# 1. Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла salary-train.csv
data_train = pd.read_csv('salary-train.csv')
data_test = pd.read_csv('salary-test-mini.csv')
print(data_test)

# 2. Проведите предобработку:
# Приведите тексты к нижнему регистру (text.lower()).
data_train['FullDescription']=data_train['FullDescription'].map(lambda t: t.lower())

# Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова.
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

# Примените TfidfVectorizer для преобразования текстов в векторы признаков.
# Оставьте только те слова, которые встречаются хотя бы в 5 объектах (параметр min_df у TfidfVectorizer)
vec_pr = TfidfVectorizer(min_df=5)
train_text=vec_pr.fit_transform(data_train['FullDescription'])
test_text=vec_pr.transform(data_test['FullDescription'])

# Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'.
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

# Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
enc = DictVectorizer()
train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

#Объедините все полученные признаки в одну матрицу "объекты-признаки".
# Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными.
# Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack.
M_train=hstack([train_categ,train_text])
M_test=hstack([test_categ,test_text])

# 3. Обучите гребневую регрессию с параметрами alpha=1 и random_state=241.
# Целевая переменная записана в столбце SalaryNormalized.
y_train=data_train['SalaryNormalized']
greb=Ridge(alpha=1,random_state=241)
greb.fit(M_train,y_train)

# 4. Постройте прогнозы для двух примеров из файла salary-test-mini.csv.
# Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.
y_test=greb.predict(M_test)
print(round(y_test[0],2), round(y_test[1],2))
