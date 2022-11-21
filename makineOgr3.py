#makineOgr3.py

import pandas as pd

data = pd.read_csv("Datasets/insurance.csv")

data.head()

# veri setinin satır + sütun sayısı
data.shape

data.info()

# eksik verileri gör
data.isnull()

# eksik veri var mı
data.isnull().sum()
data.dtypes

# veri tipini category tipe çevir
data["sex"] = data["sex"].astype("category")
data["smoker"] = data["smoker"].astype("category")
data["region"] = data["region"].astype("category")

data.dtypes

#sayısal değişkenlerin istatistiklerini gör
data.describe()

#daha ayrıntılı görmek için tranzpozunu al
data.describe().T

# sigara içen ve içmeyenlere göre ortalama ödeme miktarı
# ilk önce gruplama yap
# round(2) - virgülden sonra 2 yayı görmek için
smoke_data = data.groupby("smoker").mean().round(2)

# verileri ekrana yazdır.
# charges - > sigara içenler, içmeyenlere oranla daha fazla ücret ödüyor
smoke_data

# sayısal değişkenlerin ikili ilişkilerine bak
# pip install seaborn 
# seaborn kütüphanesini import et, istatiksel grafikler için kullanılır
import seaborn as sns

# grafik sitilini belirle
sns.set_style("whitegrid")

# ikili grafik için sütun seç, heu, palette- renklendirilecek alan adı
sns.pairplot(data[["age","bmi","charges","smoker"]],
            hue = "smoker",
            height = 3,
            palette = "Set1")

# sayısal değişkenler arasındaki ikili ilişkileri gör.
sns.heatmap(data.corr(), annot = True)

# category tipli değişkenleri OneHotEncoder kodlama yap
data.dtypes

# kolon isimlerini gör
data.columns

# kategorik alanlar için alt sütunlar oluştur
data = pd.get_dummies(data)

# kategorik alanlar için alt sütunlar oluştu
# sex, smoker, region 
data.columns

# makine öğrenmesi modelini kur
# girdi ve çıktı değişkenlerini belirle.
# sonuç değişkeni y dir
y = data["charges"] 

# Not: değişken adı uydurma, standart olanı kullan...
# X - girdi değişkenidir ve büyük harfle gösterilir.
# -axis = 1 - sütun olarak çıktı alınacak
X = data.drop("charges", axis = 1)

# veriseti eğitim ve test şeklinde ayrılır.
# eğitim verileri ile model kurulur;
# test verileri ile model değerlendirilir.
# verisetini parçalamak için train_test_split metodunu import yap
from sklearn.model_selection import train_test_split

# verisetini %80 eğitim; %20 test şeklinde parçala
# train_test_split metodu verisetini karıştırarak parçalar
# train_size=80 oranında parçala
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=1)

# modeli import et
from sklearn.linear_model import LinearRegression

# LinearRegression sınıfından bir örnekleme yap
lr = LinearRegression()

# eğitim verilerini kullanarak modeli kur
lr.fit(X_train, y_train)

# modelin performansına bak
# bunun için belirlme katsayısını bul
# bu değer 1 e yakınsa model o kadar iyidir
# round(3) virgülden sonra 3 basamak göster
lr.score(X_test, y_test).round(3)

# modelin eğitim verileri üzerindeki doğruluğunu göster
# modelin eğitim verileri üzerindeki doğruluğu test verilerine yakın çıktı.
# eğer modelin eğitim verileri üzerindeki doğruluğu yüksek çıksaydı modelde ezberleme problemi (overfitting) var demektir.
# modelde ezberleme problemi varsa model Regularization edilir. 
lr.score(X_train, y_train).round(3)

# modeli değerlendirmek içi hatakareler ortalaması
# test verilerine göre tahmin yapıldı
y_pred = lr.predict(X_test)

# ortalama 
from sklearn.metrics import mean_squared_error
import math

# veriyi modele tahmin ettir
math.sqrt(mean_squared_error(y_test, y_pred))

# ilk satırı tahmin et
# eğitim verisinin ilk satırını seç
data_new = X_train[:1]

# seçilen datayı göster
data_new

# verileri tahmin için predict metodu kullanılır
lr.predict(data_new)
# gerçek değeri gör, ilk satırın ödeme değerini yaz
# modelimiz girilen değerler için 10508 değerini tahmin ederken.
## gerçekte bu değer 10355 miş.
# model gerçek değere yakın bir tahmin yaptı
y_train[:1]
