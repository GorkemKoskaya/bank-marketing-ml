# 🏦 Bank Marketing Campaign Target Prediction with Machine Learning (Python)

This project involves building and evaluating machine learning models to predict whether a client will subscribe to a term deposit based on the Bank Marketing Dataset. The focus is on model comparison, class imbalance handling, feature selection, and performance optimization.

## 📁 Dataset

- **Source**: [Kaggle - Bank Marketing Dataset](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)
- **Records**: 45,211 samples (train-test split applied)
- **Target variable**: `y` (yes = subscribed, no = not subscribed)
- **Features**: 16 attributes related to client and campaign details

## 🧠 Models Implemented

We trained and evaluated the following models:

- `model_lg`: Logistic Regression  
- `model_rf`: Random Forest  
- `model_xgb`: XGBoost  
- `model_nb`: Naive Bayes  
- `model_svm`: Support Vector Machine  

Each model was evaluated using:

- Accuracy  
- Precision, Recall, F1-Score  
- ROC-AUC  
- Confusion Matrix  

## ⚖️ Handling Class Imbalance

The target class is highly imbalanced (approx. 88% "no", 12% "yes"). To address this, we applied:

- SMOTE (Synthetic Minority Oversampling Technique)  
- Threshold tuning on probability scores  

All experiments were repeated after applying SMOTE and adjusting thresholds for better minority class recall.

## 📊 Feature Selection

- Feature Importance from XGBoost  
- SHAP (SHapley Additive exPlanations) values  
- Final model trained using selected features based on contribution

## 🧪 Optimization

- Hyperparameter tuning with `GridSearchCV` for XGBoost  
- Optimized thresholds for best F1/Recall trade-off  
- ROC-AUC and classification metrics used to select final model  

## 📈 Visualizations

- SHAP summary plots  
- ROC curves  
- Feature importance bar charts  
- Performance comparison charts across models  

## 🧰 Libraries Used

```python
pandas  
numpy  
scikit-learn  
xgboost  
imbalanced-learn  
shap  
matplotlib  
seaborn  
```
### 📌 Conclusion

Through this project:

- The classification performance of multiple models was tested.
- The impact of SMOTE and threshold optimization was demonstrated.
- SHAP analysis was used to enhance the explainability of the model's decisions.

💡 **The best results** were achieved using a combination of class balancing, SHAP-based feature selection, and an optimized `XGBoost` model via GridSearch.


# 🇹🇷 Bankacılık Verisi ile Vadeli Mevduat Kampanyasına Katılım Tahmini

Bu proje, **Bank Marketing** veri seti kullanılarak müşterilerin vadeli mevduat teklifine **"evet" veya "hayır"** deme olasılığını sınıflandırmayı amaçlamaktadır. Python dili ile geliştirilen bu çalışma; veri ön işleme, model karşılaştırmaları, dengesiz sınıf yapısı ile baş etme yöntemleri, hiperparametre optimizasyonu ve model açıklanabilirliği gibi adımları kapsamaktadır.

---

### 📁 Veri Seti
- **Kaynak:** [Kaggle - Bank Marketing Dataset](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)
- **Toplam Gözlem:** 45.211
- **Hedef Değişken:** `y` (Müşteri abone oldu mu? - yes / no)
- **Özellikler:** Yaş, meslek, eğitim durumu, kredi geçmişi, kampanya iletişim bilgileri, vb. 16 değişken.

---

### 🧠 Uygulanan Modeller

Temel sınıflandırma algoritmaları uygulanmıştır:

- `model_lg` – Lojistik Regresyon  
- `model_rf` – Rastgele Orman  
- `model_xgb` – XGBoost  
- `model_svm` – Destek Vektör Makineleri  
- `model_nb` – Naive Bayes  

Her model, aşağıdaki metriklerle değerlendirilmiştir:

- Doğruluk (Accuracy)  
- ROC AUC  
- F1-Score  
- Precision / Recall

---

### ⚖️ Sınıf Dengesizliği

Veri setindeki dengesiz sınıf dağılımı (%88 Hayır, %12 Evet) nedeniyle aşağıdaki yöntemler uygulanmıştır:

- `SMOTE` (Synthetic Minority Over-sampling)
- Ağırlıklı sınıf yaklaşımları
- Eşik (Threshold) optimizasyonu
- SHAP değerlerine göre önemli değişken seçimi

---

### 🔧 Hiperparametre Ayarlaması

`GridSearchCV` ile optimize edilmiş parametreler:

```python
param_grid = {
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5, 7],
    "n_estimators": [100, 200],
    "subsample": [0.8, 1],
    "colsample_bytree": [0.8, 1]
}
```

---

### 📊 Görselleştirmeler

- Sınıf dağılımı grafiği  
- ROC eğrileri  
- Precision / Recall karşılaştırmaları  
- Feature importance (XGBoost & SHAP)  
- SHAP summary & bar plot  
- SMOTE sonrası model performansları  
- Threshold optimizasyonu sonrası değerlendirme

---

### 🧮 Kullanılan Kütüphaneler

```python
pandas, numpy, matplotlib, seaborn  
sklearn, xgboost, imblearn  
shap, joblib
```

---

### 📌 Sonuç

Bu proje sayesinde:

- Farklı modellerin sınıflandırma performansı test edilmiştir.  
- SMOTE ve eşik optimizasyonunun etkisi gösterilmiştir.  
- SHAP analizi ile modelin kararlarına açıklık getirilmiştir.  

💡 **En iyi sonuç**, sınıf dengesizliği + SHAP temelli değişken seçimi + `XGBoost` modelinin grid search sonrası optimize edilmesi ile elde edilmiştir.
