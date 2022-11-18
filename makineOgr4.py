#makineOgr4.py
# PYTHON ile VERİ ANALİZİ

import pandas as pd
df = pd.read_csv("Datasets/forbes_2022_billionaires.csv")

df.head()

#Satır, sütun sayısını göster
df.shape

# sütun isimlerini gör
df.columns

# veri setindeki bazı sütunları seç
df = df.loc[:,["rank", "personName", "age", "finalWorth","category", "country", "gender"]]

# seçilen sütunları veri seti içerisinde gör
df.head()

# seçilen sütunları yeniden isimlendir.
df = df.rename(columns={"rank":"Sıra", 
                        "personName":"İsim", 
                        "age": "Yaş", 
                        "finalWorth":"Servet",
                        "category":"Kategori", 
                        "country":"Ülke", 
                        "gender":"Cinsiyet"})
df.head()

# Sıra sütununu index yap
df = df.set_index("Sıra")
df.head()

# versisetindeki sütun tipine bak
df.dtypes

#verisetindeki eksik verileri kontrol et
# df.isnull().sum().sum() - genel toplam sayıyı verir
df.isnull().sum()

# eksik veri içeren satırları verisetinden çıkart
# inplace=True - yapılan değişikliklerin kaydedilmesi için
df.dropna(inplace=True)

# veri setinin yapısını gör
df.shape

# veri setindeki erkek ve kadın sayısını gör
df["Cinsiyet"].value_counts()

# erkek ve kadın sayısını % olarak göster
# % 88 - erkek; % 11 kadın
df["Cinsiyet"].value_counts(normalize=True)

# Türkiyedeki durumu göster
df[df["Ülke"]=="Turkey"].Cinsiyet.value_counts(normalize=True)

# veri setindeki Ülke isimlerini gör
df["Ülke"].unique()

# Kanada ülkesine bak; zenginler için
# % 95 erkek, % 4 kadın
df[df["Ülke"]=="Canada"].Cinsiyet.value_counts(normalize=True)

# ZENGİNLERİN CİNSİYETE GÖRE YAŞ ORTALAMALARINI GÖR
df_cinsiyet = df.groupby(["Cinsiyet"])

# yaş sütununu seç , fonk. kullan
df_cinsiyet["Yaş"].mean()

# veri setindeki erkek ve kadın sayılarını grafik olarak göster
# temayı seç
# grafik kalitesini ayarla
import seaborn as sns
sns.set_theme()
sns.set(rc = {"figure.dpi":300})

# uyari mesajı almamak için kütüphaneleri import et.
# import warnings de hata veriyoe bu şekilde kullan.
import sys
if not sys.warnoptions:
    import warnings
    warnings.filterwarnings("ignore")

# grafik çizdir
df_cinsiyet.size().plot(kind = "bar")

# dünyanın en zenginlerinin grafiğini gör
sns.barplot(y = df["İsim"][:10], x = df["Servet"][:10])

# en fazla milyarderlerin olduğu ülkeler bul
# Listede Kaç ülke var sayılarını gör
len(df["Ülke"].unique())

# Ülkeye göre gruplama yap
df_ulke = df.groupby("Ülke")

# zenginlerin bulunduğu ülkeleri sırala
# veriyi DataFrame çevir
df_ulke_sayi = pd.DataFrame(df_ulke.size().sort_values(ascending=False), columns = ["Sayı"])

# ilk satırları gör
df_ulke_sayi.head()

# Ülkelerin grafiğini çizdir
sns.barplot(x = df_ulke_sayi["Sayı"][:10], y = df_ulke_sayi.index[:10])

# Türkiyedeki en zenginleri göster. Bunun için
# veri setinden Türkiye yi filtrele
df_turkiye = df[df["Ülke"]=="Turkey"]

# listedeki zengin sayısını göster
df_turkiye["İsim"].count()

# zengin isimlerini göster
df_turkiye.head(10)

# Grafiğini çizdir
sns.barplot(y = df_turkiye["İsim"][:10], x = df_turkiye["Servet"][:10])

# en fazla zengin hangi alanda/iş grubunda çalışıyor
df["Kategori"].unique()

# Katagori isimleri aradaki boşlukları kaldır,sembol yerine _ yazdır.
df["Kategori"] = df["Kategori"].apply(lambda x:x.replace(" ", "")).apply(lambda x:x.replace("&", "_"))

# yapılan değişikliği gör
df["Kategori"].unique()

# il 10 kategorinin grafiğini çiz
# kategori sayısını değişkene ata
df_kategori = df.groupby("Kategori").size()
df_kategori.head()

# veriyi DataFrame yapısına çevir
df_kategori = df_kategori.to_frame()
df_kategori.head()

# 0 sütununun ismini değiştir
# sayıları büyükten küçüğe sırala
df_kategori = df_kategori.rename(columns={0:"Sayi"}).sort_values(by="Sayi", ascending=False)

# veri setini göster
df_kategori.head()

# Grafik çizdir
sns.barplot(x = df_kategori["Sayi"][:10], y = df_kategori.index[:10])

# yaş ile servet arasıda bir ilişki var mı bak
#-- burada hata veriyor
sns.scatterplot(df["Yaş"], df["Servet"])

# zenginlerin yaş dağılımına bak
sns.histplot(df["Yaş"])

#en genç zenginleri göster
df_yas = df.sort_values("Yaş")
df_yas

# grafik çizdir
sns.barplot(y=df_yas["İsim"][:10], x=df_yas["Yaş"][:10])
