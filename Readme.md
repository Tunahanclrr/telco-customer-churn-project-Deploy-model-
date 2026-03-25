# Telco Müşteri Churn Tahmin Projesi

## 📋 Proje Açıklaması

Bu proje, telekomünikasyon sektöründe müşteri churn (müşteri kaybı) tahminini amaçlayan kapsamlı bir makine öğrenmesi uygulamasıdır. IBM Sample Dataset kullanılarak geliştirilen sistem, müşterilerin aboneliklerini iptal etme olasılığını tahmin eder ve şirketlerin proaktif müşteri tutma stratejileri geliştirmesine yardımcı olur.

Müşteri churn tahmini, yeni müşteri kazanmanın maliyetinin mevcut müşterileri tutmaktan çok daha yüksek olduğu telekom sektöründe kritik bir öneme sahiptir.

## 🎯 Özellikler

- **Veri Analizi**: Kapsamlı EDA (Exploratory Data Analysis) ile müşteri davranışlarının analizi
- **Makine Öğrenmesi**: LightGBM algoritması ile optimize edilmiş churn tahmin modeli
- **Web API**: FastAPI ile geliştirilmiş REST API servisi
- **Web Arayüzü**: Kullanıcı dostu HTML arayüzü ile model tahminleri
- **Model Değerlendirmesi**: Confusion Matrix, ROC AUC, Precision, Recall gibi metriklerle performans analizi

## 📊 Veri Kümesi

- **Kaynak**: IBM Sample Dataset (Telco Customer Churn)
- **Boyut**: 7,043 müşteri kaydı, 21 özellik
- **Hedef Değişken**: Churn (Evet/Hayır) - Son ayda abonelik iptali
- **Önemli Özellikler**:
  - Demografik: Cinsiyet, Yaş, Partner durumu, Bağımlılar
  - Servis: İnternet tipi, Ek servisler (Güvenlik, Yedekleme, vb.)
  - Hesap: Sözleşme tipi, Aylık ücretler, Ödeme yöntemi

## 🛠️ Kullanılan Teknolojiler ve Araçlar

### Makine Öğrenmesi & Veri İşleme
- **Python**: Ana programlama dili
- **pandas & numpy**: Veri manipülasyonu ve analizi
- **scikit-learn**: Makine öğrenmesi algoritmaları ve pipeline
- **LightGBM**: Gradient boosting algoritması (optimize edilmiş model)
- **XGBoost**: Alternatif boosting algoritması

### Web Geliştirme
- **FastAPI**: Yüksek performanslı REST API framework
- **uvicorn**: ASGI sunucusu
- **HTML/CSS**: Web arayüzü

### Görselleştirme ve Analiz
- **matplotlib & seaborn**: Veri görselleştirme
- **Jupyter Notebook**: İnteraktif veri analizi

### Diğer Araçlar
- **joblib**: Model serileştirme ve kaydetme
- **requests**: HTTP istekleri

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.8+
- pip paket yöneticisi

### Adımlar

1. **Depoyu klonlayın veya indirin**
   ```bash
   git clone <repo-url>
   cd telco-churn-project
   ```

2. **Gereksinimleri yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

3. **Modeli eğitin (opsiyonel - önceden eğitilmiş model mevcut)**
   ```bash
   python main.py
   ```

4. **API sunucusunu başlatın**
   ```bash
   cd api
   uvicorn app:app --reload
   ```

5. **Web arayüzüne erişin**
   - Tarayıcınızda `http://localhost:8000` adresine gidin

## 📁 Proje Yapısı

```
telco-churn-project/
│
├── main.py                    # Ana eğitim scripti
├── requirements.txt           # Python bağımlılıkları
├── Readme.md                  # Bu dosya
│
├── api/                       # Web API servisi
│   ├── app.py                 # FastAPI uygulaması
│   └── templates/
│       └── index.html         # Web arayüzü
│
├── data/                      # Veri dosyaları
│   └── telco.csv              # IBM telco dataset
│
├── models/                    # Eğitilmiş modeller
│   └── lgbm_churn_model.pkl   # LightGBM modeli
│
├── notebooks/                 # Jupyter notebook'lar
│   └── eda.ipynb              # Keşifsel veri analizi
│
└── src/                       # Kaynak kod modülleri
    ├── data_loader.py         # Veri yükleme
    ├── preprocess.py          # Veri ön işleme
    ├── train.py               # Model eğitimi
    └── evaluate.py            # Model değerlendirmesi
```

## 📈 Model Performansı

- **Algoritma**: LightGBM Classifier
- **Konfigürasyon**: Sınıf dengesizliği için optimize edilmiş (pos_weight=2.77)
- **Metrikler**:
  - ROC AUC: ~0.85
  - Accuracy: ~80%
  - Precision/Recall dengesi

## 🔧 API Kullanımı

### Tahmin İsteği
```bash
POST /predict
Content-Type: application/json

{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 79.85
}
```

### Yanıt
```json
{
  "churn_probability": 0.78,
  "prediction": "Yes",
  "confidence": "High"
}
```

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📞 İletişim

Proje ile ilgili sorularınız için issue açabilir veya katkıda bulunabilirsiniz.
