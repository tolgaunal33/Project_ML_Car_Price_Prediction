# 🚗 Car Price Prediction — Machine Learning Project

A machine learning project that predicts used car market prices based on vehicle specifications, deployed as an interactive web application.

---

## 📌 Problem Statement

Buying or selling a used car often comes with uncertainty — is the asking price fair? Is the seller overcharging? Is the buyer getting a good deal?

This project addresses that question by building a predictive model trained on real market data, enabling users to estimate a vehicle's fair market value based on its key characteristics.

---

## 🎯 Objective

- Scrape real used car listings from a US automotive platform
- Clean, process, and explore the data
- Train and compare multiple ML models to find the best predictor
- Deploy the final model as an interactive web app for buyers and sellers

---

## 🔄 Project Workflow

### 1. Web Scraping (`web_scrapping.ipynb`)
- Scraped **29,000+ vehicle listings** from a US automotive website using Python
- Collected key features: make, model, year, mileage, engine size, horsepower, fuel type, transmission, and price

### 2. Data Cleaning & EDA (`ml_model_building.ipynb`)
- Handled missing values, outliers, and inconsistent formatting
- Explored distributions of price, mileage, and other features
- Identified key correlations between vehicle specs and market price

### 3. Model Building & Comparison (`ml_model_building.ipynb`)
Trained and evaluated three models:

| Model | Performance |
|---|---|
| Linear Regression | Baseline — high error on non-linear relationships |
| Random Forest | Improved accuracy, better at capturing complex patterns |
| **XGBoost** | **Best performer — selected for deployment** |

XGBoost was selected based on lowest MAE and highest R² score.

### 4. Deployment (`app.py`, `streamlit_app.ipynb`)
- Built an interactive **Streamlit web app**
- Users input vehicle specs (make, model, year, mileage, engine, HP) and receive an instant price prediction
- Designed for both sellers (pricing guidance) and buyers (overpaying detection)

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Collection | Python, BeautifulSoup / Selenium |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Environment | Jupyter Notebook, Git |

---

## 📁 Repository Structure

```
Project_ML_Car_Price_Prediction/
│
├── web_scrapping.ipynb          # Data collection via web scraping
├── ml_model_building.ipynb      # EDA, feature engineering, model training
├── streamlit_app.ipynb          # Streamlit app development notebook
├── app.py                       # Streamlit app — production file
│
├── cars_data.csv                # Cleaned dataset
├── all_cars_data.xlsx           # Raw scraped dataset
│
├── linear_regression_model.pkl  # Saved Linear Regression model
├── random_forest_model.pkl      # Saved Random Forest model
└── xgb_model.pkl                # Saved XGBoost model (deployed)
```

---

## 🚀 How to Run the App

```bash
# Clone the repository
git clone https://github.com/tolgaunal33/Project_ML_Car_Price_Prediction.git
cd Project_ML_Car_Price_Prediction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## 📊 Presentation

Full project presentation available here:
👉 [View Presentation](https://view.genially.com/67cb2be9d18c72af9c246ed5/guide-car-price-prediction-model)

---

## 👤 Author

**Tolga Unal**
[LinkedIn](https://www.linkedin.com/in/tolgaaunall) · [GitHub](https://github.com/tolgaunal33)
