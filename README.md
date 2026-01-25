# ✈️ BA Customer Booking Prediction

## 📌 Project Overview
This project aims to predict whether a customer will complete a flight booking based on historical booking behavior and travel-related features.  
A **Random Forest Classifier** is used to model booking completion outcomes.

---

## 📊 Dataset Description
- Records: ~50,000 customer booking attempts
- Target Variable: `booking_complete`  
  - 1 → Booking completed  
  - 0 → Booking not completed
- Features include:
  - Purchase lead time
  - Length of stay
  - Flight day and hour
  - Number of passengers
  - Add-on services (extra baggage, seat selection, meals)

---

## ⚙️ Methodology
1. Data Cleaning & Exploration
2. Categorical Encoding (One-Hot Encoding)
3. Train-Test Split (80/20 with stratification)
4. Random Forest Classification
5. Model Evaluation:
   - Accuracy
   - ROC-AUC Score
   - Classification Report
6. Cross-Validation
7. Feature Importance Analysis

---

## 📈 Results
- Accuracy: ~85%
- ROC-AUC: ~0.79
- The model performs well for non-bookings but struggles with completed bookings due to class imbalance.
- **Purchase lead time** is the most influential predictor.

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn
