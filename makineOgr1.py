
#!/usr/bin/env python
# coding: utf-8

import pandas as pd

df = pd.read_csv("Datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# veri setini ilk 5 satırı
df.head()


# In[4]:


# veri setinin tamamını (tüm sütunları) gör / transpozunu al
df.T


# In[5]:


# Veri setini satır ve sütun sayılarını göster
df.shape


# In[6]:


# sütun tipini göster; object karakteri, diğerleri sayısal değeri olduğunu gösterir
df.dtypes


# In[7]:


# TotalCharges sütunundaki verileri sayısal tipe çevir
# errors; hata vermemesi için
df.TotalCharges = pd.to_numeric(df.TotalCharges, errors="coerce")


# In[8]:


# TotalCharges sütununda;
# isnull(); eksik verileri tespit eder
# sum(); toplam sayısını verir
df.isnull().sum()


# In[9]:


# 11 adet eksik veri yerine 0 değerini ata
df.TotalCharges = df.TotalCharges.fillna(0)


# In[10]:


df.isnull().sum().sum()


# In[11]:


# sütun isimlerini küçük harfe çevir
# boşluk varsa _ işaretini koy
df.columns = df.columns.str.lower().str.replace(" ", "_")


# In[12]:


# string_columns isimli değişkene type=object olan sütunları liste olarak ata
string_columns = list(df.dtypes[df.dtypes=="object"].index)


# In[13]:


# string_columns değişkeninde oluşturulan listeyi göster
string_columns


# In[14]:


# sütundaki değerleri küçük harfe çevir, boşuklara _ işaretini koy.
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(" ", "_")


# In[15]:


# veri setindeki değerleri göster
df.head()


# In[16]:


# hedef değişken (churn sütunu) değerlerini göster
df.churn.head()


# In[17]:


# churn sütunundaki verileri sayısala çevir; 
# yes ise ilk önce true, sonra 1; no ise false, 0 yap
df.churn = (df.churn == "yes").astype(int)


# In[18]:


# hedef değişken (churn sütunu) yeni değerlerini göster
df.churn.head()


# In[19]:


# veri setini son halini göster
df.head()


# In[20]:


# veri setini parçala işlemi için fonksiyon çağır
from sklearn.model_selection import train_test_split


# In[21]:


# eğitim ve test verilerini parçala
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)


# In[22]:


# verinin performansını ölçmek için; eğitim verisinin bir kısmını validasyon verisi için ayır
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)


# In[23]:


# churn sütununu değişkene ata
y_train = df_train.churn.values


# In[24]:


# churn sütununu validasyon verisi için 
y_val = df_val.churn.values


# In[25]:


# churn hedef sütununu veri değişkeninden sil
# hedef değişkeni veri setinden kaldırıldı
del df_train["churn"] 
del df_val["churn"]


# In[26]:


# katagarik sütunları değişkene ata
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies', 'contract',
               'paperlessbilling', 'paymentmethod']


# In[27]:


# sayısal sütunları değişkene ata
numerical = ['tenure','monthlycharges','totalcharges']


# In[28]:


# Değişkenleri kullanarak eğitim setini sözlük yapısına çevir
train_dict = df_train[categorical + numerical].to_dict(orient="records")


# In[29]:


# değişkenin ilk değerini göster
train_dict[:1]


# In[30]:


# kategorik verileri OneHotEncoder kodlamaya çevir
# DictVectorizer sınıfını impot et
from sklearn.feature_extraction import DictVectorizer


# In[31]:


# DictVectorizer sınıfından bir örnek al, matris oluştur
dv = DictVectorizer(sparse = False)


# In[32]:


# Alınan örneği eğit, bunun için 
# train_dict isimli eğitim veri setini yaz.
dv.fit(train_dict)


# In[33]:


# Eğitim verisini dönüştür. 
# Eğitim veriseti OneHotEncoder kodlama yapıldı.
X_train = dv.transform(train_dict)


# In[34]:


# değişkenin ilk değerleri göster
X_train[0]


# In[35]:


# sütunların değerlerini göster
dv.get_feature_names_out()


# In[36]:


# Model Kurma İşlemine Geçiş
from sklearn.linear_model import LogisticRegression # 


# In[37]:


# solver parametresi optimizasyon için kullanılır
# liblinear küçük örnekler için
model = LogisticRegression(solver="liblinear", random_state=42) 


# In[54]:


# fit metodu ile modeli eğit (eğitim veri setlerini yazarak)
model.fit(X_train, y_train)


# In[56]:


# modelin performansını değerlendir
# validasyon verisini sözlük yapısına çevir
val_dict = df_val[categorical + numerical].to_dict(orient="records")


# In[40]:


# kategorik verileri sayısala dönüştür, analize uygun hale getir.
X_val = dv.transform(val_dict)


# In[57]:


# kurulan modele göre verileri tahmin et.
y_pred = model.predict_proba(X_val)


# In[58]:


# değerlerin ilk 5 tanesini gör
y_pred[:5] 


# In[43]:


# modelin performansı: %80
model.score(X_val, y_val)


# In[44]:


# modelin eğitim verileri üzerindeki performansı: %80
# skorlar ne kadar 1 e yakınsa model o kadar iyidir.
model.score(X_train, y_train) 


# In[60]:


# modele sabit terim ekleme yazıldı ??? ne demek ???
model.intercept_[0] 


# In[61]:


# dict; sözlük tipine çevir
# zip; birleştir
# get_feature_names_out; sütun isimlerini al
# model.coef_; model katsayısını eşleştir
dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))


# In[62]:


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


# In[65]:


# customer verileri ham veri, 
# bu verileri model kurarken kullandığımız önceki verilere uygun hale getir.
x_new = dv.transform([customer])


# In[67]:


# dönüştürülen verileri modele ver
# buna göre; yeni verilerin etiketinin 0 olma ihtimali 0.92; 1 olma ihtimali 0.07 dir.
# müşteriye promosyon uygulamaya gerek yok.
model.predict_proba(x_new)


# In[66]:


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


# In[51]:


# müşteri dğerlerini modelin anlayacağı dile çevir.
x_new2 = dv.transform([customer2])


# In[52]:


# modele göre verileri tahmin et.
# müşterinin abonelikten çıkma ihtimali 0.79
# Promosyon uygulanmalı
model.predict_proba(x_new2)

