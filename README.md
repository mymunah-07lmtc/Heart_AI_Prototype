# 🫀 AI-Powered Heart Sound Screener

**Research Prototype | By Maimouna Tougoutcho Coulibaly**

An end-to-end machine learning prototype for detecting heart murmurs in pediatric patients. This project extends the lung sound screener framework to cardiac auscultation, aiming to screen for rheumatic heart disease in low-resource settings.

- **Trained on 1,604 heart sounds** from the CirCor DigiScope dataset
- **36-feature audio pipeline** (MFCCs, spectral features, RMS, ZCR)
- **F1-Macro: 0.595** (5-fold cross-validation)
- **Handles class imbalance** using SMOTE (5% Normal, 95% Abnormal)
- **Full Streamlit web interface** with waveform visualization and PDF reports
- **Raspberry Pi deployment ready** ($66 hardware stack)

---

## 📦 Project Structure

```
Heart_AI_Prototype/
├── app_heart.py              # Main Streamlit web application
├── build_heart_model.py      # Training script (with SMOTE)
├── extract_features.py       # Feature extraction from audio
├── requirements.txt          # Python dependencies
├── heart_model_best.pkl      # Pre-trained model
├── scaler.pkl                # Feature normalizer
└── label_map.pkl             # Label encoder (0=Normal, 1=Abnormal)
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Heart_AI_Prototype.git
cd Heart_AI_Prototype
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app_heart.py
```
Your browser will open at `http://localhost:8501`.

---

## 📊 Dataset

- **CirCor DigiScope Phonocardiogram Dataset**
- **1,604 recordings** from 1,568 pediatric patients
- **4 auscultation locations** (AV, PV, TV, MV)
- **Labels:** Normal, Innocent Murmur, Pathological Murmur
- **Class distribution:** 5% Normal, 95% Abnormal (imbalanced)

---

## 🧠 Model Performance

| Metric | Value |
| :--- | :--- |
| **Algorithm** | RandomForest (with SMOTE) |
| **Cross-Validation** | 5-Fold Stratified |
| **F1-Macro** | 0.595 |
| **Recall** | 0.589 |
| **Precision** | 0.605 |
| **Accuracy** | 92.9% |

> ⚠️ *Note: Accuracy is high due to class imbalance. F1-Macro is the key metric.*

---

## 🔬 Future Work

- Collect more **Normal** heart sounds to improve F1-Macro
- Multi-class classification (Normal vs. Innocent vs. Pathological)
- Clinical validation in Malian pediatric clinics
- Integration with the existing lung sound screener hardware

---

## 📄 Disclaimer

⚠️ **For Research Purposes Only.** This tool is a proof-of-concept prototype. It is **NOT** a certified medical device. Do not make clinical decisions based solely on this output.

---

## 📬 Contact

**Author:** Maimouna Tougoutcho Coulibaly  
**Email:** maimounatcoul@gmail.com  
**GitHub:** [github.com/mymunah-07lmtc](https://github.com/mymunah-07lmtc)  
**LinkedIn:** [linkedin.com/in/maimouna-tougoutcho-coulibaly](https://linkedin.com/in/maimouna-tougoutcho-coulibaly)

---

**Built with ❤️ in Bamako, Mali | CirCor DigiScope Dataset**
```
