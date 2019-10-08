import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import numpy as np
import math
from sklearn.model_selection import train_test_split


from scipy.stats.stats import pearsonr

#wczytanie danych
df_train=pd.read_csv("train.csv")
print(df_train)
df_test=pd.read_csv("test.csv")
print(df_test)
# zamiana oceny jakości na liczby
df_train["Ocena_jakości"].replace(to_replace=dict(Bad=0, Good=1), inplace=True)
print(df_train)
df_test["Ocena_jakości"].replace(to_replace=dict(Bad=0, Good=1), inplace=True)

reg=linear_model.LinearRegression()

#deklaracja modelu regresji liniowej

reg=linear_model.LinearRegression()


#trenowanie modelu

X = df_train[['Ilość_artykułów','Liczba_wersji_językowych','Liczba_wersji_językowych,Liczba_wyświetleń_strony_kategorii_w_2019']]
y = df_train['Ocena_jakości']

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)


reg.fit(X_train,y_train)
#testowanie wspolczynnika determinacji

accuracy=reg.score(X_test,y_test)
print(accuracy)

reg.fit(X,y)
accuracy=reg.score(X,y)
print(accuracy)



# przewidywanie
predictions=reg.predict(df_test[['Ilość_artykułów','Liczba_wersji_językowych','Liczba_wersji_językowych,Liczba_wyświetleń_strony_kategorii_w_2019']] )


for i in range(len(predictions)):
    if predictions[i]>0.5:
        predictions[i]=1

    else:
        predictions[i]=0


df_test['Ocena_jakości']=predictions

print(df_test)

#tabelka korelacji

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)
corr=df_train.corr()

print(corr)