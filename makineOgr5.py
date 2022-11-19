#makineOgr5.py

import pandas as pd
df = pd.read_csv("Datasets/googleplaystore.csv")
df.head()

# sütun isimlerini gör
df.columns

# veriyi analize uygun hale getir
# sütun isimlerindeki boşlukları _ ile doldur
df.columns = df.columns.str.replace(" ", "_")
df.columns

# veri setindeki satır ve sütun sayıları
df.shape

# veri analizinde,
# satırlara örneklem, sütunlara da öznitelik denir.
# sadece Rating sütunu ondalık,
# Reviews, Size, Installs sütunları sayısal tipte olmalı
df.dtypes

# Önce eksik veri analizi yap, verileri gör
df.isnull().sum()

# eksik verileri görsel (grafik) hale getir
# seaborn kütüphanesini import et
import seaborn as sns

# grafiklerin daha iyi görünmesi için set_theme metodunu kullan
# grafik kalitesini ve boyutunu ayarla
sns.set_theme()
sns.set(rc={"figure.dpi":300, "figure.figsize":(12,9)})

# eksik verileri görselleştir
sns.heatmap(df.isnull(), cbar=False)

# eksik veri yerine sütunun medyanı (ortanca) alınır
# medyan: bir sayı dizisi küçükten büyüğe sıralayarak ortada kalan elemanı medyan değeri olarak belirleme işlemidir.
# Rating sütununun medyan değerini bul 
rating_median = df["Rating"].median()
rating_median

# Rating sütununun medyan değerini eksik verilere ata
df["Rating"].fillna(rating_median, inplace=True)

# diğer sütunların eksik verileri az olduğu için 
# dropna metodu ile veri setinden kaldır
df.dropna(inplace=True)

# veri setinde eksik veri yok.
df.isnull().sum()

# veri seti hakkındaki özet bilgileri gör
df.info()

# Reviews, Size, Installs sütunlarını sayısal tipte çevir
# describe metodu ile istatistiğini gör. bu sütunda;
# toplam (count) 10829 değer var
# tek değer (unique) sayısı 5999
# en çok değer (top) 0 dan oluşuyor
# 0 tekrar (freq) sayısı 594
df["Reviews"].describe()

# sütunu Int64-tamsayı tipine çevir
df["Reviews"] = df["Reviews"].astype("int64")

# Reviews sütunu özet bilgisini gör
# round() metodu ile değerleri 1 ler basamağına göre yuvarla
# verinin ortalaması: 444602
# ortanca değer: 50% -> 2100 verinin ortalaması medyan dan büyük
# standart sapma değeri yüksek: std -> 2929213
# maksimum değer: max -> 78158306 ortalamadan yüksek
# istatistikler bu sütunda aykırı değerler olduğunu gösteriyor. 
df["Reviews"].describe().round()

# Size sütunu uydulamanın boyutunu gösterir
# Sütundaki unique (tek) değerleri gör
# sütunda 457 adet tek değer var
# M -> megabyte, k -> KiloByte gösterir
print(len(df["Size"].unique()))
df["Size"].unique()

# Size sütunundaki M ve k harflerini kaldır.
# regex=True -> Regular Expressions, genellikle harflerden olusan karakterler dizisinin kısa yoldan ve esnek bir biçimde belirlenmesini sağlayan bir yapı
# inplace=True -> değişikliği kaydet
df["Size"].replace("M","", regex=True, inplace=True)
df["Size"].replace("k","", regex=True, inplace=True)

# veri setinin son halini gör
# sütundaki'Varies with device' metin ifadeini kaldır.
df["Size"].unique()

# Size sütunundaki'Varies with device' metin ifadesi yerine,
# sütunun medyan değerini yaz 
# Varies with device metin ifadesi bulunmayan değerleri seç ve sayıala çevir, 
# size_median değişkenine at.
# Size sütununun medyan değeri 15
size_median = df[df["Size"]!="Varies with device"]["Size"].astype(float).median()
size_median

# "Varies with device" metni yerine medyan değerini ata
df["Size"].replace("Varies with device", size_median, inplace=True)

# Size sütunun sayısala çavirebiliriz.
df.Size= pd.to_numeric(df.Size)

# sütunun ilk satırlarını gör
df.Size.head()

# sütunun özet istatistiğini gör
# sütunun tüm değerleri, ortalama, s.sapma, medyanı, çeyreklik değeri
df.Size.describe().round()

# Installs sütunundaki tek değerleri gör,
# sütundaki yükleme sayıları yanında + sembolü ve , var
df["Installs"].unique()

# + ve , sembollerini kaldır.
# değerleri tamsayıya çevirmek için int metodunu kullan
df.Installs = df.Installs.apply(lambda x:x.replace("+",""))
df.Installs = df.Installs.apply(lambda x:x.replace(",",""))
df.Installs = df.Installs.apply(lambda x:int(x))

# değerleri kontrol et
df["Installs"].unique()

# Price sütunundaki tek değerleri gör,
df["Price"].unique()

# sütundaki $ sembolünü kaldır
# değerleri sayısala çavir
df.Price = df.Price.apply(lambda x:x.replace("$",""))
df.Price = df.Price.apply(lambda x:float(x))

# unique (tek) değerleri yeniden gör
df["Price"].unique()

# Genres sütunu işlemlerini yap
# sütundaki tek değerleri gör
len(df["Genres"].unique())

# ilk 10 satırı gör
# bazı değerlerde ; sembolü var
# İlk değer Tür'ü, ikinci değer alt kategoriyi gösterir
df["Genres"].head(10)

# Önce iki ifadeyi ayır ve Genres değerini al
# str[0] -> ilk değeri seç
df["Genres"] = df["Genres"].str.split(";").str[0]

# sütundaki unique değer genel toplamını gör
len(df["Genres"].unique())

# sütundaki unique değer isimlerini gör
df["Genres"].unique()

# sütundaki grupların sayısını gör
df["Genres"].value_counts()

# list sonundaki Music & Audio yerine Music yaz 
# Music gurup adedi 25 iken 26 oldu
df["Genres"].replace("Music & Audio", "Music", inplace=True)

# Last_Updated sütunu işlemleri
#sütunun ilk satırlarını gör
df["Last_Updated"].head()

# Last_Updated sütununda tarihler var ve object tipinde
# bu sütunu datetime (tarih) tipine çevir 
df["Last_Updated"] = pd.to_datetime(df["Last_Updated"])

# veri temizleme ön işlemleri bitti
# veri setinin son halini gör
df.head()

# veri setindeki sütun tiperini gör
df.dtypes

# veri görselleştirme (grafik) işlemi
# ücretli ve ücretsiz uygulamaların sayısını göster
df["Type"].value_counts().plot(kind="bar", color="red")

# ücretli ve ücretsiz uygulamaların reytingleri, 
# kutu grafiğinde kutular verinin %25 ile %75 i gösterilir 
# ortadaki çizgi medyanı ifade eder
# üst ve alt çizgilere whiskers (bıyık) denir. 
# whiskers (bıyık) verinin alt ve üst çeyreğini gösterir
# 1 nci kutu ücretsiz, 2 nci kutu ücretli uygulamaları gösterir
# ücretli uygulamaların reyting ortalaması daha fazla
# noktalar ykırı değerleri gösterir. 
sns.boxplot(x = "Type", y = "Rating", data = df)

# grafik çizmek için çok sık kullanılan kütüphaneyi import et
import matplotlib.pyplot as plt

# Content_Rating sütunun incele
# sütun kategori sayılarını gör
# grafik başlığını değiştir
# uygulamanın en fazla hitap ettiği kesi Everyone(herkes)
# sora Teen-> gençler geliyor
sns.countplot(y= "Content_Rating", data = df)
plt.title("Content Rating with their counts")

# uygulamaların bu gruplara göre reytingini gör
sns.boxplot(x = "Content_Rating", y = "Rating", data = df)

# uygulamaların kategori sayısnı bul
cat_num = df["Category"].value_counts()
cat_num

# kategorilerin sayısnı gösteren grafik
# en fazla uygulama aile, sonra oyun, sonra tools ...
sns.barplot(x = cat_num, y = cat_num.index, data = df)
plt.title("The number of categories", size=20)

# en fazla para verilen kategorilerin sayısı
sns.scatterplot(x = "Price", y = "Category", data = df)
plt.title("en fazla para verilen kategoriler", size=20)

# sayısal değişken arasındaki fark
# corr() metodu-> korelasyon matris fonksiyonu
# korelasyon grafiğinde mavi renkler negatif korelasyonu, kırmızı renkler pozitif korelasyonu temsil eder.
# annot -> değişkenlerin isimnlerinin yazılması için
# linewidths -> hücreler arasındaki boşluk için
# fmt -> değerlerin yuvarlanması için (%1 ler basamağına göre) 
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt=".2f")

# reyting dağılımını bul
# kde-> dağılımı düzleştirmek için, olasılık tahmin eğrisi
# kde-> Kernel Density Estimation (Çekirdek Yoğunluğu Tahmini)
sns.histplot(df["Rating"], kde = True)

