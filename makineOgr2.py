#makineOgr2.py  

import pandas as pd

df_train = pd.read_csv("Datasets/train.csv") 
df_test = pd.read_csv("Datasets/test.csv")

df_train.head()

# inplace=True değişiklikleri kaydet
df_train.set_index("Id", inplace=True)
df_test.set_index("Id", inplace=True)

df_train.head()

# Verisetinin satır sütun sayısını gör
print("Train shape: ", df_train.shape)
print("Test shape: ", df_test.shape)

df_train.columns

df_test.columns

# int64-> tam sayıları,
# float -> ondalıklı sayıları,
# object -> karakter olduğunu gösterir
df_train.dtypes

df_test.dtypes

# eksik verileri gör
df_train.isnull().sum()

# eksik verilere göre sütunları sırala, ilk 20 kayudı göster
cols_with_null = df_train.isnull().sum().sort_values(ascending=False)
cols_with_null.head(20)

# toplam eksik veri sayısını göster
df_train.isnull().sum().sum()

# veriseti hakkındaki temel bilgileri gör
df_train.info()

# hedef sütunda eksik veri varmı öğren
df_train["SalePrice"].isnull().sum()

# eksik veri içeren sütunları kaldır;
# bunun için değişkene atama yap
# ilk 6 sütunu al, indexle ve listeye çevir
cols_to_drop  = (cols_with_null.head(6).index).tolist()

# en fazla eksikveri içeren 6 sütun;
# cols_to_drop içeriğini göster
cols_to_drop

# en fazla eksikveri içeren sütunlari hem veri, hemde etest setinden kaldır
df_train.drop(cols_to_drop, axis=1, inplace=True)
df_test.drop(cols_to_drop, axis=1, inplace=True)

# eğitim veri setinde kalan sütun sayısı
df_train.shape

# Verisetindeki sayısal sütunların istatistiklerini gör
df_train.describe().T

# Veri ön işleme
# Hedef ve öz nitelik değişkeni oluştur
y = df_train.SalePrice
X = df_train.drop(["SalePrice"], axis=1)

# Test verisinin bir kısmını validasyon verisi olark al
# Bunun için veri parçalama işlemi yap
# train_test_split fonksiyonu verileri karıştır ve rastgele böler
from sklearn.model_selection import train_test_split

# train_test_split fonksiyonunu çağırarak eğitim verisetini böl
X_train, X_val, y_train, y_val = train_test_split(X,y,train_size=0.8, random_state=0)

# Analize hazır hale getir
# veritipi object olup, 10 dan küçük kategorisi olan sütunları seç
categorical_cols = [cname for cname in X_train.columns
                   if X_train[cname].nunique()<10 and 
                       X_train[cname].dtype == "object"]

# seçilen sütunların sayısını gör; 35 sütun seçildi
len(categorical_cols)

# sayısal sürunları seç
numerical_cols = [cname for cname in X_train.columns
                 if X_train[cname].dtype in ["int64", "float64"]]

# seçilen sütun sayılarını gör; 35 sütun seçildi
len(numerical_cols)

# seçilmeyen sütunları analizden çıkart
# bunun için kategorik ve sayısal sütun değişkenlerini kullan
my_cols = categorical_cols + numerical_cols

# bu değişkenlere göre verisetindeki sütunları seç
# my_cols içindeki sütunlar veri setinden seçildi
# aynı işlemi validasyon verisi ve test verisi için yap
# çalıştıdığına sütunlar seçilmiş oldu
X_train = X_train[my_cols]
X_val = X_val[my_cols]
X_test = df_test[my_cols]

# sayısal ve kategorik sütunları ayrı ayrı ele al ve bunları analize uygun hale getir.
# bu işlen için Pipeline sınıfını import et
# SimpleImputer - Eksik Verilerin Tamamlanması için import et
# StandardScaler - verileri ölçekleme işlemi için
# OneHotEncoder - kategorik her veriye sayısal bir değer ata
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder 

# Pipeline sınıfından sayısal alanlar için bir örnek al
# StandardScaler sınıfından sayısal alanlar için bir örnek al
# sayısal alanlar için veri önişleme adımı tamamlandı.
numerical_transformer = Pipeline(steps=[
    ("imputer_num", SimpleImputer(strategy="median")), # eksik verileri nasıl ele alacağımızı belirttik
    ("scaller", StandardScaler())
    ])

# Kategorik veriler üzerine işlem yap
# Kategorik verilerde eksik veri yerine sütunun mod değerini kullan
categorical_transformer = Pipeline(steps=[
    ("imputer_cal", SimpleImputer(strategy="most_frequent")), # eksik verileri nasıl ele alacağımızı belirttik
    ("onehot", OneHotEncoder(handle_unknown="ignore")) #bilinmeyen bir kategori ile karşılaşıldığında 
    ])

# sütunları analize hazır hale getirmek için 2 adet  Pipeline oluşturuldu
# şimdi hangi sütunlara, handi Pipeline uygulanacak onu ayarla; bunun için 
# ColumnTransformer sınıfını import et 
from sklearn.compose import ColumnTransformer

# sütunların nasıl dönüşeceğini belirt.
# hangi dönüşümü hangi değişkene uygulayacağını belirt
# "num" - numerik sütun için
# "cat" - kategorik sütun için
# böylece veri önişleme şaması tamamlanmış oldu.
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# model kurma aşamasına geçiş
# RandomForest olgoritması kullanıldı. 
# bunun için RandomForestRegressor sınıfını import et.
from sklearn.ensemble import RandomForestRegressor

# model oluştur.
rf = RandomForestRegressor(n_estimators=100, random_state=0)

# Pipeline örneği oluştur
# burada model kuruldu
my_pipelie = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ("model", rf)
])

# şimdi modeli eğitelim
# burada model eğitilir ve model adımları ekrana yazılır.
my_pipelie.fit(X_train, y_train)

# model eğitildiğine göre
# validasyon verileri tahmin edilebilir
val_preds = my_pipelie.predict(X_val)

# modelin validasyon uzerindeki performansını gör
# bunu için mean_absolute_error fonksiyonunu import et
from sklearn.metrics import mean_absolute_error

# y_val - Gerçek etiketleri içerir 
# val_preds - Tahmişn ettgiğimiz değerleri gösterir
# Val MAE:  17721 -tahmin hatasını  gösterir.
# tüm verinin %20 si kullanıldı
print("Val MAE: ", mean_absolute_error(y_val, val_preds))

# Cross-validation tekniği
# tüm verinin %20 lik parçasının ayrı ayrı validasyon verisi olması için
# Cross-validation tekniği tekniği kullanılır. 
# tekniği import et ve modeli yeniden kur. 
from sklearn.model_selection import cross_val_score

# veriseti 5 parçaya bölünecek,
# -1 ile çarpılır; çıkan sonucun pozitif bir değer olması için 
scores = -1 * cross_val_score(my_pipelie, X,y, cv =5,
                             scoring = "neg_mean_absolute_error")

# 5 parçaya göre model eğitildi ve 5 skor bulundu.
# 5 skorların ortalaması modelin doğruluğu hakkında daha sağlıklı bilgi verir.
scores.mean()

