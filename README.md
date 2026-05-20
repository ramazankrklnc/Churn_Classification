# Müşteri Kaybı (Churn) Tahmini — Kaggle Yarışması

Telekom müşterilerinin şirketten ayrılıp ayrılmayacağını tahmin eden bir **ikili sınıflandırma** projesi. Veri keşfi, özellik mühendisliği, klasik makine öğrenmesi, derin öğrenme ve ağırlıklı ensemble harmanlama adımlarını içerir.

Ana çalışma defteri: `ChurnClassification.ipynb`

---

## Problem tanımı

Bir telekom operatörünün müşteri verileri kullanılarak, müşterinin **churn** (hizmeti bırakma) olasılığı tahmin edilir.


| Hedef       | Açıklama                              |
| ----------- | ------------------------------------- |
| `churn = 0` | Müşteri hizmette kalmaya devam ediyor |
| `churn = 1` | Müşteri son dönemde ayrıldı           |


Veri seti **dengesizdir**: eğitim kümesinde yaklaşık **4.139** kalan ve **1.495** ayrılan müşteri bulunur (~%26.5 churn oranı).

---

## Veri seti


| Dosya       | Satır | Açıklama                               |
| ----------- | ----- | -------------------------------------- |
| `train.csv` | 5.634 | Etiketli eğitim verisi (`churn` dahil) |
| `test.csv`  | 1.409 | Etiketsiz test verisi (tahmin için)    |


**Özellik grupları (21 sütun):**

- **Demografi:** `customerid`, `gender`, `seniorcitizen`, `partner`, `dependents`
- **Hizmetler:** `tenure`, `phoneservice`, `multiplelines`, `internetservice`, ek paketler (`onlinesecurity`, `streamingtv`, vb.)
- **Sözleşme & ödeme:** `contract`, `paperlessbilling`, `paymentmethod`, `monthlycharges`, `totalcharges`
- **Hedef:** `churn` (yalnızca train)

---

## Proje yapısı

```
.
├── ChurnClassification.ipynb    # Ana notebook (EDA → model → submission)
├── train.csv                    # Eğitim Verisi
├── test.csv                     # Test Verisi
└── submission.csv               # Ağırlıklı ensemble tahminleri
```

---

## Metodoloji

### 1. Keşifsel veri analizi (EDA)

- Eksik değer ve veri tipi kontrolü (`totalcharges` vb.)
- Sayısal ve kategorik dağılımlar, churn oranları
- Korelasyon matrisi ve churn ile ilişkili içgörüler  
*(ör. `tenure` arttıkça churn azalır; `contract` ve `monthlycharges` güçlü belirleyiciler)*

### 2. Veri hazırlığı

- Kategorik değişkenler için **Label Encoding**
- Özellik mühendisliği (sözleşme ağırlığı, one-hot türevleri vb.)
- **StandardScaler** (yalnızca eğitim parçasında `fit`, sızıntı önlenir)
- Dengesiz sınıflar için **SMOTE** (yalnızca eğitim verisinde; validation seti orijinal kalır)

### 3. Modeller

**Klasik makine öğrenmesi (scikit-learn):**

- Logistic Regression
- SVM
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost

**Derin öğrenme (TensorFlow / Keras):**

- ANN (çok katmanlı yapay sinir ağı)
- 1D CNN
- Transfer Learning tabanlı mimari

### 4. Değerlendirme

Modeller **ROC-AUC**, accuracy, F1-score, Cohen's Kappa ve confusion matrix ile karşılaştırılır. Çapraz doğrulama için 5-fold CV kullanılır.

### 5. Final tahmin (ensemble)

En iyi genelleme için **ağırlıklı blending**:


| Model               | Ağırlık |
| ------------------- | ------- |
| XGBoost             | %40     |
| Gradient Boosting   | %30     |
| Logistic Regression | %15     |
| Random Forest       | %15     |


Çıktı: `submission.csv` — `customerid` ve ondalık **Churn olasılığı** sütunları.

---

## Yerel doğrulama sonuçları (özet)

Validation setinde örnek ROC-AUC skorları (`Deneme.ipynb`):


| Model               | ROC-AUC |
| ------------------- | ------- |
| Random Forest       | 0.847   |
| XGBoost             | 0.846   |
| Gradient Boosting   | 0.843   |
| Logistic Regression | 0.841   |
| ANN                 | 0.838   |
| CNN                 | 0.838   |
| AdaBoost            | 0.837   |
| Transfer Learning   | 0.837   |
| SVM                 | 0.797   |
| Decision Tree       | 0.748   |


---

## Kurulum

Python **3.10+** önerilir (notebook: 3.12.4).

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn tensorflow jupyter
```

Gerekli paketler:

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`, `imbalanced-learn`
- `tensorflow` (Keras modelleri için)
- `jupyter` veya VS Code Jupyter eklentisi

---

## Çalıştırma

1. Depoyu klonlayın ve proje klasörüne gidin.
2. `train.csv` ve `test.csv` dosyalarının kök dizinde olduğundan emin olun.
3. Jupyter ile ana defteri açın:

```bash
jupyter notebook ChurnClassification.ipynb
```

1. Hücreleri **yukarıdan aşağıya** sırayla çalıştırın (model nesneleri bellekte kalmalıdır).
2. Ensemble hücresi tamamlandığında `submission.csv` oluşur; bu dosyayı Kaggle’a yükleyebilirsiniz.

---

## Lisans

Bu repo eğitim amaçlıdır. Veri seti kullanım koşulları ilgili Kaggle sayfasına tabidir.
