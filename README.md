# 🚗 Used Car Price Prediction

A machine learning web application that predicts the estimated selling price of a used car based on its specifications, including brand, model, model year, mileage, engine, transmission, fuel type, and vehicle condition.

The application is built using Python, Scikit-learn, LightGBM, and Streamlit. It uses a tuned LightGBM model with a log-transformed target to generate price estimates and provides an option to compare predictions from multiple machine learning models.

## ✨ Features

- **Used Car Price Prediction:** Estimate a car's price from user-provided specifications.
- **Multiple Model Comparison:** Compare predictions from available regression models.
- **Data Preprocessing:** Handle missing values and encode categorical features using a machine learning pipeline.
- **Model Evaluation:** Evaluate models using regression metrics such as MAE, RMSE, and R².
- **Interactive Web Interface:** Enter car details and view predicted prices through Streamlit.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- LightGBM
- XGBoost
- CatBoost
- Joblib
- Plotly
- Streamlit

## 📂 Project Structure

```text
used-car-price-prediction/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── catboost.pkl
    ├── lightgbm.pkl
    ├── lightgbm_log.pkl
    ├── linear_regression.pkl
    ├── random_forest.pkl
    ├── tuned_lightgbm_log.pkl
    └── xgboost.pkl
```

## 🤖 Machine Learning Models

The project explores the following regression models:

1. Linear Regression
2. Random Forest Regressor
3. LightGBM Regressor
4. LightGBM with Log-Transformed Target
5. Tuned LightGBM with Log-Transformed Target
6. CatBoost Regressor
7. XGBoost Regressor

The tuned LightGBM model with a log-transformed target is the primary model used for price prediction.

## 📊 Model Performance

The tuned LightGBM model with a log-transformed target achieved the following results on the project's test set:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | $17,162.88 |
| Root Mean Squared Error (RMSE) | $68,680.75 |
| R² Score | 0.1518 |
| Mean Absolute Percentage Error (MAPE) | 37.50% |
| Median Absolute Error | $6,598.16 |

**Note:** These are experimental results from the project's test split. The relatively low R² score and high RMSE indicate that predicting used-car prices remains challenging, particularly for expensive vehicles. Predictions should be treated as estimates, not guaranteed market values.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/mahatmemadhura4/used-car-price-prediction.git
cd used-car-price-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use compatible Python and library versions for the saved model files.

### 4. Run the application

```bash
python -m streamlit run app.py
```

The application should open in your browser. If it does not, navigate to the local URL displayed in your terminal.

## 🚀 Deployment

The application can be deployed using [Streamlit Community Cloud](https://share.streamlit.io/).

1. Push the project to a GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create a new app and select this repository.
4. Set the main file path to `app.py`.
5. Deploy the application.

Ensure that the required model files are included in the repository and that `requirements.txt` specifies compatible dependencies.

## 🔍 Input Features

The application uses the following vehicle attributes:

- Brand and model
- Model year
- Mileage
- Fuel type
- Engine description
- Transmission
- Exterior and interior colors
- Accident history
- Clean title status

## ⚠️ Limitations

- Predictions depend on the quality and coverage of the training dataset.
- Luxury and unusually expensive vehicles may have larger prediction errors.
- Prices can vary by location, vehicle condition, market demand, and other factors not captured by the model.
- Saved model files may require specific library versions to load correctly.
- Predicted prices are estimates and should not replace a professional vehicle valuation.

## 🔮 Future Improvements

- Improve prediction accuracy through additional feature engineering and model tuning.
- Add more comprehensive evaluation and model interpretability.
- Expand the dataset to include market location and vehicle condition information.
- Improve the user interface with additional visualizations.
- Add further validation using an independent dataset.

## 👩‍💻 Author

**Madhura**

GitHub: [@mahatmemadhura4](https://github.com/mahatmemadhura4)

---

*This project was developed as a machine learning application for used-car price estimation and model comparison.*
