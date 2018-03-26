import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

# 1. Загрузите файл classification.csv.
# В нем записаны истинные классы объектов выборки (колонка true) и ответы некоторого классификатора (колонка pred).
data1 = pd.read_csv('classification.csv')
len=data1.shape[0]
tr=data1['true']
pr=data1['pred']

# 2. Заполните таблицу ошибок классификации
TN=0
TP=0
FN=0
FP=0
for i in range(len):
    if pr[i]==0:
        if tr[i]==0:
            TN=TN+1
        else:
            FN=FN+1
    else:
        if tr[i]==1:
            TP=TP+1
        else:
            FP=FP+1
print('TP=',TP,'FP=',FP,'FN=',FN,'TN=',TN)

# 3.  Посчитайте основные метрики качества классификатора:
acc=accuracy_score(tr,pr)   # Accuracy (доля верно угаданных)
prec=precision_score(tr,pr) # Precision (точность)
rec=recall_score(tr,pr)     # Recall (полнота)
F1=f1_score(tr,pr)          # F-мера
print('accuracy=',round(acc,2),'pricision=',round(prec,2),'recall=',round(rec,2),'F-мера=',round(F1,2))

# 4.  Имеется четыре обученных классификатора.
# В файле scores.csv записаны истинные классы и значения степени принадлежности положительному классу
# для каждого классификатора на некоторой выборке:
# для логистической регрессии — вероятность положительного класса
# для SVM — отступ от разделяющей поверхности
# для метрического алгоритма — взвешенная сумма классов соседей
# для решающего дерева — доля положительных объектов в листе
#Загрузите этот файл
data2 = pd.read_csv('scores.csv')

# 5. Посчитайте площадь под ROC-кривой для каждого классификатора.
# Какой классификатор имеет наибольшее значение метрики AUC-ROC?
scores = {}
for clf in data2.columns[1:]:
    scores[clf] = roc_auc_score(data2['true'], data2[clf])

print(pd.Series(scores).sort_values(ascending=False).head(1).index[0])

# 6. Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?
scores_PR = {}
for clf in data2.columns[1:]:
    curve = precision_recall_curve(data2['true'], data2[clf])
    curve_df = pd.DataFrame({'precision': curve[0], 'recall': curve[1]})
    scores_PR[clf] = curve_df[curve_df['recall'] >= 0.7]['precision'].max()
print(pd.Series(scores_PR).sort_values(ascending=False).head(1).index[0])
