import pandas as pd
data = pd.read_csv('titanic.csv', index_col='PassengerId')

#1 Сколько мужчин и женщин было на титанике
print('1)','male number=',str(data['Sex'].value_counts()['male']),' ','female number=',str(data['Sex'].value_counts()['female']))

#2 Какая часть людей выжили при крушении титаника
print('2) proportion of survivors=',round(data['Survived'].value_counts()[1]/(data['Survived'].value_counts()[1]+data['Survived'].value_counts()[0])*100,2),'%')

#3 Какая часть людей ехали первм классом
print('3) proportion of 1 class=',round(data['Pclass'].value_counts()[1]/(data['Pclass'].value_counts()[2]+data['Pclass'].value_counts()[1]+data['Pclass'].value_counts()[3])*100,2),'%')

#4 Средний возраст пассажиров и его медиана
a=round(data['Age'].mean(),2)
b=data['Age'].median()
print('4) mean ade=',a," madian age=",b)

#5 Корреляция между количеством детей/родителей и братьев/сестер/супругов
import numpy as np;
def PCC(X, Y):
   # Pearson Correlation Coefficient.
   # Normalise X and Y
   X -= X.mean(0)
   Y -= Y.mean(0)
   # Standardise X and Y
   X /= X.std(0)
   Y /= Y.std(0)
   # Mean product
   return np.mean(X*Y)
A=data['SibSp'];
B=data['Parch'];
a=PCC(A, B);
print('5) correlation of SibSp and Parch=',round(a,2));

#6 Самое распространенное женское имя на Титанике
Nf=0
Nm=0
FName=[]
for i in data['Name'][:]:
    a=i.find('Miss')
    if a==-1:
        a=i.find('Mrs')
    if a!=-1:
        Nf+=1
        str=i[a:]
        p1=str.find(' ')
        p2=str.find(' ',p1+1)
        s1=str.find('(')
        if s1==-1:
            if str[p1]=='(':
                p1+=1
            if p2==-1:
                p2=len(str)
            FName.append(str[p1+1:p2])
        else:
            sp2=str.find(' ',s1+1)
            if sp2==-1:
                sp2=str.find(')',s1+1)
            FName.append(str[s1+1:sp2])
max=0
m=0
for i in FName:
    m=FName.count(i)
    if max<m:
        max=m
        j=i
print('6) the most popular female name:',j)
