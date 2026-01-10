# Patient Appointment No-Show Prediction

## 🚀 Project Overview

**Goal:** Predict whether a patient will miss a medical appointment.

No-shows in healthcare lead to wasted resources, inefficiency, and reduced access for other patients. This project uses machine learning to **predict no-shows**, allowing clinics to proactively manage scheduling and reminders.

**Type:** Binary Classification (No-show: Yes / No)  
**Dataset:** [Medical Appointment No Shows – Kaggle](https://www.kaggle.com/datasets/joniarroba/noshowappointments)  
- **Size:** ~110,000 medical appointments  
- **Source:** Brazilian public healthcare system (2016)  
- **File:** `KaggleV2-May-2016.csv` (raw data)

---

## 🎯 Key Findings

- **Top Predictor:** Days between scheduling and appointment (`days_between`) - longer gaps = higher risk
- **Age Matters:** Younger patients (20-30) have 25% higher no-show rates than older patients
- **SMS Works:** Patients receiving SMS reminders are 10% less likely to miss appointments
- **Chronic Conditions:** Patients with hypertension or diabetes are more reliable attendees
- **Gender Effect:** Females have slightly higher attendance rates (3-5% difference)

---

## 📊 Dataset Features

- **Patient Information:** Age, Gender, Scholarship (welfare program)
- **Appointment Info:** Scheduled day, Appointment day, SMS received
- **Health Conditions:** Hypertension, Diabetes, Alcoholism, Handicap
- **Location:** Neighborhood (clinic location)

---

## 🔬 Exploratory Data Analysis (EDA)

**No-show rate:** ~20% of appointments missed  
**Data Cleaning:**
- Removed negative `days_between` values (appointment date before scheduled date)
- Kept same-day appointments (`days_between = 0`) as valid data
- Converted date columns to proper datetime format

**Insights:**
- **Scheduling gap:** Risk increases exponentially after 15+ days
- **Age distribution:** Bimodal with peaks at 0-10 (children) and 50-60 (adults)
- **Weekday effect:** Tuesday/Wednesday have highest attendance rates
- **Neighborhood:** Some areas show 30%+ no-show rates

---

## 🧱 Feature Engineering

- `days_between` → Days between scheduling and appointment  
- `appointment_weekday` → Day of the week (0=Monday, 6=Sunday)
- `is_weekend` → Boolean flag for Saturday/Sunday appointments  
- Binary chronic disease flags (Hypertension, Diabetes, Alcoholism, Handicap)  
- `SMS_received` → Boolean flag for reminder receipt
- `hour_of_day` → Extracted from ScheduledDay (for possible time-of-day patterns)

---

## ⚡ Modeling Approach

### **Why Random Forest?**
Random Forest was chosen as the primary model because:
- **Handles mixed data types** naturally (numerical + categorical)
- **Captures non-linear relationships** (age effects, diminishing returns of SMS)
- **Provides feature importance** for model interpretability
- **Robust to outliers** and requires minimal data preprocessing
- **Reduces overfitting** through ensemble averaging

### **Model Comparison:**
- **Logistic Regression:** Baseline model (linear relationships only)
- **Random Forest:** Best balance of performance and interpretability
- **Evaluation showed** Random Forest outperformed Logistic Regression by 15% in recall

### **Evaluation Strategy:**
- **Primary metric:** **Recall** (focused on catching no-shows, minimizing false negatives)
- **Secondary metrics:** F1-score, ROC-AUC, Precision
- **Validation:** Stratified 80/20 split with `random_state=42` for reproducibility
- **Scaling:** StandardScaler applied to all numerical features

---

## 🧠 Explainability

**SHAP (SHapley Additive exPlanations)** used to interpret model decisions:

### **Global Feature Importance:**
1. `days_between` → Strongest predictor (longer wait = higher risk)
2. `Age` → Older patients more reliable
3. `SMS_received` → Reduces no-show probability
4. Chronic conditions → Associated with better attendance

### **Local Explanations:**
- Individual patient predictions explained with **waterfall plots**
- **Example:** A 25-year-old patient with 30-day wait and no SMS → High risk flag
- **Counter-example:** 60-year-old with hypertension and SMS → Low risk

> Note: Interactive SHAP force plots may not render in VS Code; waterfall plots were used for compatibility.

---

## 📈 Results (Random Forest)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Recall** | 0.85 | Captures 85% of actual no-shows |
| **F1-score** | 0.65 | Balanced measure of precision/recall |
| **ROC-AUC** | 0.78 | Good discriminatory power |
| **Precision** | 0.52 | 52% of predicted no-shows are correct |

**Confusion Matrix:**
- True Positives: 1,704 (correctly predicted no-shows)
- False Positives: 1,571 (predicted no-show but attended)
- False Negatives: 298 (missed no-shows)
- True Negatives: 17,535 (correctly predicted attendance)

**Interpretation:** The model effectively identifies high-risk patients while maintaining reasonable precision. Feature importance aligns with clinical intuition and domain knowledge.

---

## 💡 Business Impact

### **Immediate Applications:**
1. **Targeted Reminders:** Focus SMS/calls on high-risk patients (top 20% risk score)
2. **Dynamic Scheduling:** Schedule high-risk patients closer to appointment date
3. **Resource Allocation:** Optimize staff scheduling based on predicted attendance
4. **Patient Segmentation:** Identify chronically late patients for intervention

### **Expected Benefits:**
- Reduce wasted appointment slots by 15-20%
- Improve clinic efficiency and staff utilization
- Increase patient access by reallocating no-show slots
- Enhance patient satisfaction through better scheduling

---

## 🔮 Future Improvements

### **Short-term (1-3 months):**
- **Model Enhancements:**
  - Compare XGBoost, LightGBM, and Neural Networks
  - Implement hyperparameter tuning with Optuna/Bayesian optimization
  - Add ensemble methods (voting/stacking classifiers)
- **Feature Engineering:**
  - Incorporate weather data (rain = higher no-show risk)
  - Add public holiday indicators
  - Include transportation accessibility metrics

### **Medium-term (3-6 months):**
- **Real-time System:**
  - Deploy as REST API for clinic scheduling systems
  - Create Streamlit dashboard for clinic staff
  - Implement batch prediction for weekly scheduling
- **Advanced Analytics:**
  - Time-series analysis of patient attendance patterns
  - Clustering of patient types for personalized interventions
  - Cost-benefit analysis of different intervention strategies

### **Long-term (6-12 months):**
- **Integration & Scaling:**
  - EHR (Electronic Health Record) system integration
  - Multi-clinic deployment with federated learning
  - Mobile app for patient self-scheduling with risk feedback
- **Ethical Considerations:**
  - Fairness analysis across demographic groups
  - Privacy-preserving techniques for sensitive health data
  - Transparent reporting of model limitations

---

## 📁 Project Structure
```
patient-no-show-prediction/
├── data/
│   ├── raw/
│   │   └── KaggleV2-May-2016.csv          # Original dataset
├── notebooks/
│   ├── 01_initial_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb       # Feature creation & selection
│   └── 03_modeling.ipynb                  # Modeling & interpretation
│   └── 04_explainability.ipynb            # Explainbility and evaluations
├── src/
│   ├── data_preprocessing.py              # Data cleaning pipeline
│   ├── model.py                           # Model training
│   └── train_evaluate.py                  # Run Model training & evaluation
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
└── .gitignore                             # Git exclusion rules
```

## 🛠️ Installation & Usage

### **Requirements:**
- Python 3.8+
- Dependencies: pandas, scikit-learn, shap, matplotlib, seaborn, numpy

### **Setup:**
```bash
# Clone repository
git clone https://github.com/basseljouda/patient-no-show-prediction.git
cd patient-no-show-prediction

# Install dependencies
pip install -r requirements.txt

# Alternative: Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **Run Analysis:**
```bash
# Option 1: Jupyter Notebooks (interactive exploration)
jupyter notebook notebooks/01_initial_eda.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_modeling.ipynb
jupyter notebook notebooks/04_explainability.ipynb

# Option 2: Run pipeline programmatically
python src/train_evaluate.py
```

---

## 📚 References & Resources

### **Academic & Industry:**
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [SHAP GitHub Repository](https://github.com/slundberg/shap)
- [Kaggle Dataset Source](https://www.kaggle.com/datasets/joniarroba/noshowappointments)

### **Healthcare Context:**
- "Reducing No-Show Rates" - Journal of Medical Practice Management
- "Predictive Analytics in Healthcare" - Healthcare Informatics
- Brazilian Public Health System (SUS) Guidelines

---

## 📝 License
This project is open source under the MIT License. See LICENSE file for details.

## 👤 Author

**Basel Jouda**  
Data Scientist | Healthcare Analytics  
[basseljouda@gmail.com](mailto:basseljouda@gmail.com)  
[LinkedIn Profile](https://linkedin.com/in/yourprofile) | [GitHub Portfolio](https://github.com/yourusername)

---

## 🙏 Acknowledgments
- Kaggle community for the dataset
- Brazilian public health system for data transparency
- Open-source contributors to scikit-learn, pandas, and SHAP libraries

---
