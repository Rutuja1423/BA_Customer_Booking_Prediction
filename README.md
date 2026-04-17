# British Airways Customer Booking Prediction

| Attribute | Details |
| :--- | :--- |
| **Project** | British Airways Customer Booking Prediction |
| **Domain** | Aviation / Airline |
| **Libraries** | Pandas, NumPy, Matplotlib, Scikit-learn |
| **Model** | Random Forest Classifier |
| **Author** | Rutuja1423 |
| **Date** | April 2026 |

## Problem Statement

In the airline industry, a significant proportion of customers who initiate the booking process do not complete their purchase. Understanding the factors that influence booking completion is critical for airlines to optimize their conversion strategies, reduce revenue leakage, and improve customer experience. This project aims to build a predictive classification model using historical customer booking data to determine whether a customer will complete a flight booking. The model leverages customer demographics, travel characteristics, and service preferences to identify key drivers of booking completion and to provide actionable insights for business decision-making.

---

## Project Overview

This project predicts whether a customer will complete a flight booking based on historical booking behaviour and travel-related features. A **Random Forest Classifier** is trained on approximately 50,000 customer booking records to model booking completion outcomes.

---

## Dataset Description

| Property         | Detail                                              |
|------------------|-----------------------------------------------------|
| Records          | ~50,000 customer booking attempts                   |
| Features         | 14 columns (numerical and categorical)              |
| Target Variable  | `booking_complete` (1 = completed, 0 = not completed) |
| Missing Values   | None                                                |

### Key Features

- **purchase_lead** -- Number of days between purchase date and departure date
- **length_of_stay** -- Duration of the trip in days
- **flight_hour** -- Scheduled hour of the flight
- **flight_day** -- Day of the week of the flight
- **num_passengers** -- Number of passengers in the booking
- **sales_channel** -- Channel through which the booking was initiated
- **trip_type** -- Type of trip (round trip, one way, circle trip)
- **wants_extra_baggage** -- Whether the customer requested extra baggage
- **wants_preferred_seat** -- Whether the customer requested a preferred seat
- **wants_in_flight_meals** -- Whether the customer requested in-flight meals
- **flight_duration** -- Duration of the flight in hours
- **booking_origin** -- Country of origin for the booking
- **route** -- Flight route

---

## Methodology

1. **Data Loading and Exploration** -- Loaded the dataset and examined its structure, data types, and distributions.
2. **Data Preprocessing** -- Mapped categorical day names to numerical values and applied one-hot encoding to remaining categorical variables.
3. **Train-Test Split** -- Split data into 80% training and 20% testing sets with stratification to preserve class distribution.
4. **Model Training** -- Trained a Random Forest Classifier with 200 estimators and balanced class weights to address class imbalance.
5. **Model Evaluation** -- Assessed performance using accuracy, ROC-AUC score, and a detailed classification report.
6. **Cross-Validation** -- Performed 5-fold cross-validation to evaluate model stability and generalization.
7. **Feature Importance Analysis** -- Identified the top 15 features contributing to booking completion predictions.

---

## Results

| Metric       | Value  |
|--------------|--------|
| Accuracy     | ~85%   |
| ROC-AUC      | ~0.79  |

### Key Findings

- The model performs well for predicting non-completed bookings but exhibits lower recall for completed bookings due to class imbalance.
- **Purchase lead time** is the most influential predictor of booking completion.
- Travel timing features (flight hour, flight day, length of stay) are significant predictors.
- Add-on services (extra baggage, preferred seats, in-flight meals) have a smaller but noticeable impact.
- Cross-validation indicates variability in model performance, suggesting opportunities for further tuning or alternative approaches.

---

## Project Structure

```
BA_Customer_Booking_Prediction/
|-- Data/
|   |-- customer_booking.csv
|-- Image/
|-- ba_customer_booking_prediction_ipynb.py
|-- BA_Customer_Booking_Prediction.ipynb
|-- README.md
|-- Requirements.txt
```

---

## Technologies Used

| Library       | Purpose                                  |
|---------------|------------------------------------------|
| Python        | Programming language                     |
| Pandas        | Data manipulation and analysis           |
| NumPy         | Numerical computation                    |
| Matplotlib    | Data visualisation                       |
| Scikit-learn  | Machine learning and model evaluation    |

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Rutuja1423/BA_Customer_Booking_Prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install -r Requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook BA_Customer_Booking_Prediction.ipynb
   ```

---

## Author

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rutuja%20Shinde-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/rutuja-shinde-bb83b0215/)
[![GitHub](https://img.shields.io/badge/GitHub-Rutuja1423-black?style=flat&logo=github)](https://github.com/Rutuja1423)

---

## License

This project is intended for educational and analytical purposes.
