#makineOgr6.py   

# pip install pycaret -> hata verirse 
# pycaret-2.3.10.tar.gz + paketini aç, 
#			(cmd ortamında)  klasör içinde yaz çalıştır -> python setup.py install
# pip install -U --pre pycaret -> yaz çalıştır

import pandas as pd

df = pd.read_csv("Datasets/heart.csv")
df.head()
df.shape

# veri setini parçala
data = df.sample(frac = 0.95, random_state=0)
data.head()

# veri setinin geri kalanını değişkene ata
data_unseen = df.drop(data.index)
data_unseen

# veri seti index i resetle
data.reset_index(inplace=True, drop=True)
data.head()

# data_unseen index i resetle
data_unseen.reset_index(inplace=True, drop=True)
data_unseen

# veri önişleme işlemi
# DEATH içindeki sınıfları GÖR
# 0  etiketine sahip  131 
# 1  etiketine sahip   59 örneklem var
data["DEATH"].value_counts()

# sınıflandırma modülünü import et
# import pycaret.classification
from pycaret.classification import *

# balans ayarını yapmak için
from imblearn.over_sampling import RandomOverSampler

# pycaret de veri önişleme işlemi için setup fonksiyonu kullanılır
# target = "DEATH", # hedef sütun
# normalize = True, # ölçeklendirme için
# normalize_method = "minmax", # v.ölçekleme metodu
# train_size = 0.8, # veri setini parçalama oranı
# fix_imbalance = True, # sonuç değişkenini dengelemek için 
# fix_imbalance_method = RandomOverSampler(), # dengeleme metodu
# session_id = 0) # veri setini sabitlemek için
model = setup(data = data,
             target = "DEATH",
             normalize = True,
             normalize_method = "minmax",
             train_size = 0.8,
             fix_imbalance = True, 
             fix_imbalance_method = RandomOverSampler(), 
             session_id = 0) 

# moel seçme ve parametre ayarlama adımı
# 1. yol model seç
# 2. yol otomatik model seç
# verilere göre bir hastanın ölüp ölmeyeceğini tahmin et
knn = create_model("knn")

# modelin parametre ayarını yap
tuned_knn = tune_model(knn)

# matrix in grafiğini gör
# bu grafik true ve false Pozitif oranını gösterir 
plot_model(tuned_knn, plot = "auc")

# precision (kesinlik), recall (geri çağırma) grafiğini gör
plot_model(tuned_knn, plot = "pr")

# veri setini tahmin et
predict_model(tuned_knn)

# bütün modellerin performansını gör
# bu komut ile kütüphanedeki bütün modeller eğitilir,
# modelin performansı değerlendirilir
# tahminciler eğitildi, ölçüm değerleri döndürüldü
# sarı alanlar algoritmanın en iyi ölçüm değerini gösterir
best = compare_models()

# en iyi modelin parametrelerini ayarla
tuned_best = tune_model(best)

# en iyi modele göre oluşturulan grafiği gör
plot_model(tuned_best, plot ="pr")

# bütün veri setini kullanarak model eğit
# final modeli kur
final_best = finalize_model(tuned_best)

# final modelin performansını kontrol et
# fiinal  odelin ölçümleri ekrana yazıldı
predict_model(final_best)

# en başta oluşturulan data_unseen verilerini final modele göre tahmin et
# model daha önce görmediği verileri tahmin etti,
# veriler üzerinde doğruluk değeri %60 çıktı.
predict_model(final_best, data = data_unseen)


