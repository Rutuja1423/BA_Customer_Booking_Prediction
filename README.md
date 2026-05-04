# BA Customer Booking Prediction

## 1. Business Problem
In the airline industry, a significant proportion of customers who initiate the booking process do not complete their purchase. Understanding the factors that influence booking completion is critical for airlines to optimize their conversion strategies, reduce revenue leakage, and improve customer experience. This project identifies key drivers of booking completion to provide actionable insights for targeted marketing and customer retention.

## 2. Objective
We are predicting whether a customer will complete a flight booking based on historical customer booking data, travel characteristics, and service preferences.

## 3. Dataset
*   **Source**: British Airways Customer Booking Data
*   **Rows**: ~50,000 customer booking attempts
*   **Columns**: 14 (numerical and categorical features)
*   **Target variable**: `booking_complete` (1 = completed, 0 = not completed)
*   **Key features**: `purchase_lead`, `length_of_stay`, `flight_hour`, `flight_day`, `route`, `booking_origin`

## 4. Tools Used
*   Python
*   Pandas (Data manipulation)
*   NumPy (Numerical computation)
*   Scikit-learn (Modeling and evaluation)
*   Matplotlib (Data visualization)

## 5. Methodology
*   **Data cleaning**: Mapped categorical day names (`flight_day`) to numerical values. No missing values were found.
*   **Feature engineering**: Applied one-hot encoding to categorical variables (`sales_channel`, `trip_type`, `booking_origin`, `route`).
*   **Modeling**: Trained a **Random Forest Classifier** (200 estimators). Addressed class imbalance using `class_weight='balanced'`.
*   **Validation**: 80/20 train-test split with stratification. Evaluated using accuracy, ROC-AUC, classification report, and 5-fold cross-validation.

## 6. Results
| Metric | Value |
| :--- | :--- |
| **Accuracy** | 85% |
| **ROC-AUC** | 0.79 |
| **Precision (Completed Bookings)** | 0.53 |
| **Recall (Completed Bookings)** | 0.12 |
| **F1-Score (Completed Bookings)** | 0.20 |

*   **Class Imbalance Impact**: The dataset is highly imbalanced (~85% non-completed vs ~15% completed). While overall accuracy is high (driven by predicting the majority class correctly), the recall for actual completed bookings is very low (12%). The model struggles to identify the minority of customers who actually book.
*   **Main insight**: `purchase_lead` (how far in advance the booking is initiated) is the single most important predictor. Travel timing features (length of stay, flight day) are also significant, whereas optional add-ons have a smaller impact.
*   **Business meaning**: Early planners behave differently than last-minute searchers. Strategies to encourage booking completion might be most effective if tailored to the lead time and duration of the planned trip.

## 7. Visuals
### Confusion Matrix
The confusion matrix highlights the class imbalance issue. The model correctly identifies most non-completed bookings but misses many actual completed bookings.
![Confusion Matrix](/r:/Projects/BA_Customer_Booking_Prediction/Image/Confusion_Matrix.png)

### ROC Curve
The ROC curve demonstrates the model's overall discrimination ability (AUC = 0.79), performing better than random guessing.
![ROC Curve](/r:/Projects/BA_Customer_Booking_Prediction/Image/ROC_Curve.png)

### Model Tuning Comparison (Future/Current State)
*   **Current Baseline (Random Forest with class weights)**: High accuracy (85%), moderate AUC (0.79), poor minority class recall (12%).
*   **Future Tuning Goal**: Implement techniques like SMOTE (Synthetic Minority Over-sampling Technique) or gradient boosting models (XGBoost/LightGBM) optimized for recall or F1-score to better capture the actual buyers, even at the cost of some overall accuracy.

## 8. How to Run
*   **Clone repo**: `git clone https://github.com/Rutuja1423/BA_Customer_Booking_Prediction.git`
*   **Install requirements**: `pip install -r Requirements.txt`
*   **Run notebook**: `jupyter notebook BA_Customer_Booking_Prediction.ipynb`

## 9. Limitations
*   The model assumes past booking behavior is indicative of future behavior, without accounting for external factors like economic changes, seasonal promotions, or competitor pricing.
*   The current approach does not adequately address the severe class imbalance, resulting in poor identification of actual buyers.

## 10. Future Improvements
*   **Advanced Sampling**: Apply SMOTE or ADASYN to balance the training data.
*   **Alternative Models**: Explore XGBoost, LightGBM, or CatBoost, which often handle imbalanced data better.
*   **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV to optimize model parameters specifically for F1-score or recall.
*   **Threshold Tuning**: Adjust the classification decision threshold to increase recall for the positive class.

## 11. Author
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rutuja%20Shinde-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/rutuja-shinde-bb83b0215/)
[![GitHub](https://img.shields.io/badge/GitHub-Rutuja1423-black?style=flat&logo=github)](https://github.com/Rutuja1423)

