"""
==============================================================================
British Airways Customer Booking Prediction
==============================================================================

Problem Statement:
    In the airline industry, a significant proportion of customers who initiate
    the booking process do not complete their purchase. Understanding the factors
    that influence booking completion is critical for airlines to optimize their
    conversion strategies, reduce revenue leakage, and improve customer
    experience. This project aims to build a predictive classification model
    using historical customer booking data to determine whether a customer will
    complete a flight booking. The model leverages customer demographics, travel
    characteristics, and service preferences to identify key drivers of booking
    completion and to provide actionable insights for business decision-making.

Author : Rutuja1423
Date   : April 2026
==============================================================================
"""

# STEP 1: Import required libraries
# pandas and numpy are used for data handling and manipulation
# matplotlib is used for visualisation
# sklearn provides tools for preprocessing, modelling, and evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# STEP 2: Load the dataset
# The dataset contains customer-level booking behaviour
# Target variable: 'booking_complete' (1 = booking made, 0 = not made)
# Display the first few rows to understand the structure
df = pd.read_csv("C:\Projects\BA_Customer_Booking_Prediction\Data\customer_booking.csv", encoding='latin1')
df.head()

"""The sample data represents customer flight booking attempts for the AKL-DEL route, mostly round-trip journeys booked through the internet. Customers vary in the number of passengers, booking lead time, and length of stay. Some customers opted for additional services such as extra baggage, preferred seats, or in-flight meals, while others did not. Despite early booking in many cases, these sample records show that the bookings were not completed (booking_complete = 0), indicating that factors beyond early planning -- such as preferences, timing, or customer origin -- may influence final booking decisions."""

#STEP 3: Explore the dataset
# Checking data types, missing values, and basic structure

df.info()

"""The dataset has 50,000 records and 14 columns with no missing values. It contains numerical and categorical features related to customer bookings, and the target variable booking_complete is a binary column, making the data clean and ready for modeling after encoding categorical variables."""

df["flight_day"].unique()

"""The flight_day column contains categorical values representing the days of the week (Monday to Sunday). Since these are non-numeric, they need to be converted into numerical form before being used in a machine learning model."""

# Apply the mapping to transform flight_day into numeric form
mapping = {"Mon": 1,"Tue": 2,"Wed": 3,"Thu": 4,"Fri": 5,"Sat": 6,"Sun": 7,}

df["flight_day"] = df["flight_day"].map(mapping)

# To check successfully converted to numerical values.
df["flight_day"].unique()

"""The flight_day column has been successfully converted from categorical day names to numerical values (1-7), representing days from Monday to Sunday."""

# Summary statistics for numerical variables
df.describe()

"""The summary statistics show that most bookings involve 1-2 passengers, with an average purchase lead time of about 85 days, indicating many customers book well in advance. The typical length of stay is around 23 days, and flights are most commonly scheduled around mid-day. Service add-ons such as extra baggage, preferred seats, and in-flight meals are chosen by a significant portion of customers. The target variable booking_complete has a low mean value, confirming that only a small percentage of bookings are completed, which indicates class imbalance in the dataset."""

# Check the distribution of the target variable
# This helps identify whether the data is imbalanced
df['booking_complete'].value_counts(normalize=True)

"""The target variable booking_complete is highly imbalanced, with about 85% of records representing incomplete bookings and only around 15% representing completed bookings."""

#STEP 4: Prepare data for modelling
# Separate input features (X) and target variable (y)
X = df.drop('booking_complete', axis=1)
y = df['booking_complete']

# One-hot encode categorical variables
#Convert categorical variables into numerical format
X_encoded = pd.get_dummies(X, drop_first=True)

# STEP 5: Split data into training and testing sets
# 80% of data is used for training the model
# 20% of data is used for evaluating performance
# Stratification ensures class balance is maintained

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y,test_size=0.2,random_state=42,stratify=y)

# STEP 6: Train the Random Forest model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200,random_state=42,class_weight='balanced')
rf.fit(X_train, y_train)

"""A Random Forest classifier with 200 decision trees is used to predict booking completion. The class_weight='balanced' parameter is applied to handle the class imbalance in the target variable, while random_state=42 ensures reproducible results."""

# STEP 7: Evaluate model performance on test data
# Predictions are generated on test data
# Multiple metrics are used for robust evaluation

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))

"""The model achieves an accuracy of about 85%, but this is mainly driven by correctly predicting non-completed bookings. The ROC-AUC score of ~0.79 indicates good overall discrimination ability. However, the recall for completed bookings (booking_complete = 1) is low, showing that while the model is strong at identifying non-bookings, it struggles to correctly capture customers who actually complete a booking a common issue with imbalanced datasets."""

# STEP 8: Cross-validation
# Cross-validation evaluates how stable the model is
# across different subsets of the data
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf, X_encoded, y,cv=5,scoring='roc_auc')

print("Cross-validation AUC scores:", cv_scores)
print("Mean CV AUC:", cv_scores.mean())

"""The cross-validation results show high variation in ROC-AUC scores across different folds, with a low mean AUC of approximately 0.48. This indicates that the model's performance is unstable across subsets of the data and may not generalize well. The inconsistency suggests the presence of class imbalance and potential data complexity, highlighting the need for further model tuning or alternative modeling approaches."""

# STEP 9: Feature importance analysis
# Feature importance helps identify which variables
# contribute most to predicting booking completion
import matplotlib.pyplot as plt
import numpy as np

importances = rf.feature_importances_
# Select top 15 most important features
indices = np.argsort(importances)[-15:]

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X_encoded.columns[indices])
plt.xlabel("Feature Importance")
plt.title("Top 15 Features Influencing Booking Completion")
plt.show()

"""The feature importance analysis shows that purchase_lead is the most influential factor in predicting booking completion, indicating that customers who plan earlier behave differently from last-minute bookers. Other important features include length of stay, flight hour, and flight day, suggesting that travel timing and trip duration strongly affect booking decisions. Customer origin, flight duration, and number of passengers also contribute, while add-on services such as extra baggage, preferred seats, and in-flight meals have a smaller but noticeable impact. Overall, booking completion is driven mainly by planning behavior and travel characteristics rather than optional services alone."""