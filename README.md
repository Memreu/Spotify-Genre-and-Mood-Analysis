# 🎧 Spotify Şarkı Analizi: Türler, Duygular ve Makine Öğrenmesi

~32 bin Spotify şarkısının ses özellikleriyle uçtan uca bir veri bilimi projesi: keşifsel analiz, istatistiksel testler, boyut indirgeme, açıklanabilir makine öğrenmesi ve ölçülebilir bir öneri sistemi.

👉 Tüm analiz ve grafikler: [`spotify_analizi.ipynb`](spotify_analizi.ipynb)

## Öne çıkan bulgular

- **Müzik hüzünleniyor:** 2010'larda hüzünlü + gergin şarkıların payı %32'den **%55'e** çıktı; şarkılar kısaldı (4,4 → 3,5 dk) ve yükseldi (−9,3 → −5,9 dB).
- **Popülerliği ses değil sanatçı belirliyor:** Ses özellikleri popülerliğin yalnızca %7'sini açıklıyor; sanatçı bilgisiyle %26.
- **Tür tahmini:** Optuna ile ayarlı LightGBM + şarkı adından TF-IDF metin modeli (stacking) → **%65,8 doğruluk, %83,9 ilk 2 doğruluk** (rastgele: %16,7).
- **Optimizasyon dersi:** Hiperparametre ayarı yalnızca ses özellikleriyle ~%62 tavanını aşamadı; kazanç modele yeni bilgi vermekten geldi.
- **Genelleme:** Sanatçıya göre bölmede (GroupKFold) doğruluk yalnızca 1,5 puan düşüyor; model sanatçıları ezberlemiyor.
- **Öneri sistemi:** Öneriler rastgeleye göre **7,7 kat** daha sık aynı alt türden geliyor.

## Kullanılan yöntemler

Kruskal–Wallis + etki büyüklüğü (ε²) · Russell duygu modeli · Spearman trend testleri · t-SNE · katlama dışı hedef kodlama · TF-IDF + lojistik regresyon · LightGBM · Optuna (TPE) · soft-voting ensemble · GroupKFold · SHAP · kosinüs benzerliği ile öneri sistemi

## Çalıştırma

```bash
pip install -r requirements.txt
jupyter lab
```

`spotify_analizi.ipynb` dosyasını açıp **Run ▸ Run All Cells** deyin. Veri ilk çalıştırmada otomatik indirilir; tüm notebook ~12 dakika sürer.

## Veri

[TidyTuesday — Spotify Songs (2020-01-21)](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21)
