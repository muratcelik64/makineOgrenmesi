# makineOgr1.py

import pandas as pd

df = pd.read_csv("Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# veri setini ilk 5 satırı
df.head()

# veri setinin tamamını (tüm sütunları) gör / transpozunu al
df.T

# Veri setini satır ve sütun sayılarını göster
df.shape

# sütun tipini göster; object karakteri, diğerleri sayısal değeri olduğunu gösterir
df.dtypes

# TotalCharges sütunundaki verileri sayısal tipe çevir
# errors; hata vermemesi için
df.TotalCharges = pd.to_numeric(df.TotalCharges, errors="coerce")

# TotalCharges sütununda;
# isnull(); eksik verileri tespit eder
# sum(); toplam sayısını verir
df.isnull().sum()

# 11 adet eksik veri yerine 0 değerini ata
df.TotalCharges = df.TotalCharges.fillna(0)

df.isnull().sum().sum()

# sütun isimlerini küçük harfe çevir
# boşluk varsa _ işaretini koy
df.columns = df.columns.str.lower().str.replace(" ", "_")

# string_columns isimli değişkene type=object olan sütunları liste olarak ata
string_columns = list(df.dtypes[df.dtypes=="object"].index)

# string_columns değişkeninde oluşturulan listeyi göster
string_columns

# sütundaki değerleri küçük harfe çevir, boşuklara _ işaretini koy.
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(" ", "_")

# veri setindeki değerleri göster
df.head()

# hedef değişken (churn sütunu) değerlerini göster
df.churn.head()

# churn sütunundaki verileri sayısala çevir; 
# yes ise ilk önce true, sonra 1; no ise false, 0 yap
df.churn = (df.churn == "yes").astype(int)

# hedef değişken (churn sütunu) yeni değerlerini göster
df.churn.head()

# veri setini son halini göster
df.head()

# veri setini parçala işlemi için fonksiyon çağır
from sklearn.model_selection import train_test_split

# eğitim ve test verilerini parçala
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)

# verinin performansını ölçmek için; eğitim verisinin bir kısmını validasyon verisi için ayır
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

# churn sütununu değişkene ata
y_train = df_train.churn.values

# churn sütununu validasyon verisi için 
y_val = df_val.churn.values

# churn hedef sütununu veri değişkeninden sil
# hedef değişkeni veri setinden kaldırıldı
del df_train["churn"] 
del df_val["churn"]

# katagarik sütunları değişkene ata
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies', 'contract',
               'paperlessbilling', 'paymentmethod']

# sayısal sütunları değişkene ata
numerical = ['tenure','monthlycharges','totalcharges']

# Değişkenleri kullanarak eğitim setini sözlük yapısına çevir
train_dict = df_train[categorical + numerical].to_dict(orient="records")

# değişkenin ilk değerini göster
train_dict[:1]

# kategorik verileri OneHotEncoder kodlamaya çevir
# DictVectorizer sınıfını impot et
from sklearn.feature_extraction import DictVectorizer

# DictVectorizer sınıfından bir örnek al, matris oluştur
dv = DictVectorizer(sparse = False)

# Alınan örneği eğit, bunun için 
# train_dict isimli eğitim veri setini yaz.
dv.fit(train_dict)

# Eğitim verisini dönüştür. 
# Eğitim veriseti OneHotEncoder kodlama yapıldı.
X_train = dv.transform(train_dict)

# değişkenin ilk değerleri göster
X_train[0]

# sütunların değerlerini göster
dv.get_feature_names_out()

# Model Kurma İşlemine Geçiş
from sklearn.linear_model import LogisticRegression # 

# solver parametresi optimizasyon için kullanılır
# liblinear küçük örnekler için
model = LogisticRegression(solver="liblinear", random_state=42) 

# fit metodu ile modeli eğit (eğitim veri setlerini yazarak)
model.fit(X_train, y_train)

# modelin performansını değerlendir
# validasyon verisini sözlük yapısına çevir
val_dict = df_val[categorical + numerical].to_dict(orient="records")

# kategorik verileri sayısala dönüştür, analize uygun hale getir.
X_val = dv.transform(val_dict)

# kurulan modele göre verileri tahmin et.
y_pred = model.predict_proba(X_val)

# değerlerin ilk 5 tanesini gör
y_pred[:5] 

# modelin performansı: %80
model.score(X_val, y_val)

# modelin eğitim verileri üzerindeki performansı: %80
# skorlar ne kadar 1 e yakınsa model o kadar iyidir.
model.score(X_train, y_train) 

# modele sabit terim ekleme yazıldı ??? ne demek ???
model.intercept_[0] 

# dict; sözlük tipine çevir
# zip; birleştir
# get_feature_names_out; sütun isimlerini al
# model.coef_; model katsayısını eşleştir
dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))

# modelin yeni veriyi tahmin etmesi
customer = { 
    'customerid': '8879-zkjof',
    'gender': 'male',
    'seniorcitizen': 1,
    'partner': 'no',
    'dependents': 'no',
    'tenure': 41,
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'no',
    'deviceprotection': 'yes',
    'techsupport': 'yes',
    'streamingtv': 'yes',
    'streamingmovies': 'yes',
    'contract': 'one_ year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'bank_transfer_(automatic)',
    'monthlycharges': 79.85,
    'totalcharges': 2990.75}

# customer verileri ham veri, 
# bu verileri model kurarken kullandığımız önceki verilere uygun hale getir.
x_new = dv.transform([customer])

# dönüştürülen verileri modele ver
# buna göre; yeni verilerin etiketinin 0 olma ihtimali 0.92; 1 olma ihtimali 0.07 dir.
# müşteriye promosyon uygulamaya gerek yok.
model.predict_proba(x_new)

# yeni bir müşteri verisi al.
customer2 = { 
    'gender': 'female',
    'seniorcitizen': 1,
    'partner': 'no',
    'dependents': 'no',
    'phoneservice': 'yes',
    'multiplelines': 'yes',
    'internetservice': 'fiber_optic',
    'onlinesecurity': 'no',
    'onlinebackup': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'monthlycharges': 85.7,
    'totalcharges': 85.7}

# müşteri dğerlerini modelin anlayacağı dile çevir.
x_new2 = dv.transform([customer2])

# modele göre verileri tahmin et.
# müşterinin abonelikten çıkma ihtimali 0.79
# Promosyon uygulanmalı
model.predict_proba(x_new2)

