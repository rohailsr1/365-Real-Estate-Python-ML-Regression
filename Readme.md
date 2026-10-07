# 🏢 Dubai Real Estate Valuation Engine | Machine Learning & Streamlit

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Regression-9ACD32)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)

## 📌 Project Overview

The **Dubai Real Estate Valuation Engine** is an end-to-end machine learning project developed to estimate residential property values across Dubai.

The project uses **172,704 raw Dubai property transactions** and applies data cleaning, feature engineering, exploratory analysis, categorical encoding, model comparison, hyperparameter tuning, and model deployment.

The final solution uses a **tuned LightGBM Regressor** to predict property values based on property characteristics, size, market community, nearby landmarks, nearby malls, registration status, and transaction timing.

The trained model is deployed through a **Streamlit valuation application** that allows users to enter property specifications and receive an estimated property value and price-per-square-foot range.

---

## 🎯 Business Problem

Real estate valuation can vary significantly depending on property size, location, layout, market conditions, and whether a property is ready or off-plan.

The objective of this project was to build a data-driven automated valuation model that can:

- Estimate residential property prices using historical transaction data
- Capture non-linear relationships between property characteristics and price
- Account for location and market differences across Dubai
- Provide an easy-to-use valuation interface for end users
- Produce a practical valuation range rather than a single estimate

---

## 📊 Dataset

The project began with **172,704 Dubai Land Department transaction records**.

### Data filtering

The analysis focused on:

- Sales transactions
- Residential properties
- Property types: **Unit** and **Villa**
- Properties with valid transaction values and areas
- Removal of non-residential subtypes such as offices, shops, and hotel apartments

After filtering and data-quality treatment, the modeling dataset contained approximately **82,178 transactions**.

### Key information used

| Feature | Description |
|---|---|
| `actual_worth` | Transaction property value |
| `procedure_area` | Property area in square meters |
| `rooms_en` | Property layout / bedroom classification |
| `property_type_en` | Unit or Villa |
| `reg_type_en` | Ready or Off-Plan |
| `area_name_en` | Original DLD area classification |
| `nearest_landmark_en` | Nearest landmark |
| `nearest_mall_en` | Nearest shopping mall |
| `instance_date` | Transaction date |
| Building / project fields | Property location context |

---

## 🧹 Data Preparation & Cleaning

Several data-quality and transformation steps were performed before modeling.

### Data quality checks

- Identified duplicate records
- Examined data types and missing values
- Removed columns with 100% missing values
- Handled missing categorical information
- Removed records with missing critical pricing or room information
- Standardized categorical values using trimming and title casing
- Investigated inconsistent community names

### Geographic standardization

Dubai Land Department area names were mapped into more recognizable market communities.

Examples include:

- `Marsa Dubai` → **Dubai Marina**
- `Burj Khalifa` → **Downtown Dubai**
- `Al Barsha South Fourth` → **Jumeirah Village Circle (JVC)**
- `Al Merkadh` → **Mohammed Bin Rashid City (MBR)**
- `Al Thanyah Fifth` → **Jumeirah Lakes Towers (JLT)**
- `Hadaeq Sheikh Mohammed Bin Rashid` → **Dubai Hills Estate**
- `Nadd Hessa` → **Dubai Silicon Oasis**
- `Al Hebiah Fourth` → **Jumeirah Village Triangle (JVT)**

This standardization reduced inconsistencies caused by variations in naming, spacing, and capitalization.

---

## 🔎 Exploratory Data Analysis

The project examined the major drivers and patterns behind Dubai residential property prices.

### Price distribution

The raw property-value distribution was highly right-skewed due to the presence of luxury transactions.

- Raw price skewness: **6.47**
- Log-transformed price skewness: **0.58**

A logarithmic transformation was therefore applied to the target variable to improve distribution balance and make the model less dominated by extreme luxury transactions.

### Community-level pricing

Community analysis revealed substantial variation in both total property prices and price per square foot.

Key observations included:

- **Palm Jumeirah** recorded the highest median property values and among the highest median price-per-square-foot levels.
- **Downtown Dubai** ranked among the strongest markets in price per square foot.
- Areas such as **International City, Dubai Production City, and Dubai Silicon Oasis** were among the lower-priced markets.
- Some communities showed relatively high total property values despite comparatively lower price-per-square-foot levels, highlighting the importance of property size.

### Market trend

The dataset covered transactions from **2020 to 2025**.

Median property prices increased from approximately **AED 1.1M in 2020** to over **AED 1.92M in 2025**, while median price per square foot increased from approximately **AED 922** to **AED 1,632**.

Transaction volume peaked in **2024 at approximately 16,816 transactions**.

### Property characteristics

The analysis also investigated:

- Property size vs. price
- Bedroom/layout vs. price
- Ready vs. off-plan properties
- Price-per-square-foot differences
- High-value and luxury-property outliers

Larger properties generally commanded higher total prices, while properties with similar sizes could still show substantial price differences depending on location and other property characteristics.

---

## ⚙️ Feature Engineering

The model was trained using engineered numerical and categorical variables.

### Engineered features

- `log_size_sqft`
- `rooms_ordinal`
- `is_off_plan`
- `transaction_year`
- `transaction_quarter`
- `transaction_month`
- `market_age_months`

### Target transformation

The target property value was transformed using:

```python
log_actual_worth = np.log(actual_worth)
```

Property size was also transformed:

```python
log_size_sqft = np.log(size_sqft)
```

This helped reduce skewness and allowed the model to better capture relative differences across low-, mid-, and high-value properties.

### Bedroom encoding

Property layouts were converted into an ordinal representation:

```text
Studio        → 0
Single Room   → 1
1 B/R         → 2
2 B/R         → 3
2 B/R+M       → 4
3 B/R         → 5
...
Penthouse     → 12
```

---

## 🤖 Machine Learning Approach

Multiple regression algorithms were evaluated to identify the strongest predictive approach.

### Models evaluated

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

Categorical variables with high cardinality, particularly location-related fields, were handled using **Target Encoding**.

The data was split into:

- **80% Training**
- **20% Testing**

with:

```python
random_state = 42
```

---

## 📈 Model Development

### Raw-price vs. log-price Linear Regression

The log transformation produced a major improvement in linear regression performance.

| Model | MAPE | R² |
|---|---:|---:|
| Linear Regression – Raw Price | 75.60% | — |
| Linear Regression – Log Price | 18.23% | — |

This demonstrated the importance of transforming the heavily skewed property-value target.

---

## 🏆 Final Model: Tuned LightGBM

LightGBM produced the strongest overall results among the evaluated models.

Hyperparameters were optimized using:

**RandomizedSearchCV**

with:

- 5-fold cross-validation
- 8 parameter combinations
- 40 total model fits
- Negative Mean Squared Error as the optimization metric

The tuned model used parameters including:

- `num_leaves`
- `learning_rate`
- `n_estimators`
- `min_child_samples`
- `subsample`
- `colsample_bytree`

---

## 📊 Final Model Performance

The tuned LightGBM model achieved:

| Metric | Result |
|---|---:|
| **R²** | **94.64%** |
| **MAPE** | **13.52%** |
| **MAE** | **≈ AED 268,400** |
| **MAE %** | **15.35%** |

The model reduced MAPE from **13.57% to 13.52%** after tuning and slightly improved R² from **94.59% to 94.64%**.

A lower MAPE indicates that the model's predicted property values were, on average, relatively close to the actual transaction values.

---

## 💻 Streamlit Valuation Application

The trained model was deployed through a Streamlit application designed as a practical automated valuation interface.

### User Inputs

Users can specify:

- Property Archetype
  - Unit
  - Villa
- Registration Status
  - Ready
  - Off-Plan
- Bedrooms / Layout
- Property Size
- Market Community
- Nearest Landmark
- Nearest Mall

### Application Output

The application provides:

**Estimated Property Value**

- Conservative Estimate
- Baseline Valuation
- Optimistic Estimate

**Price per Square Foot**

- Conservative estimate
- Baseline estimate
- Optimistic estimate

The valuation range is derived using the model's **13.52% MAPE** as a symmetric margin around the baseline prediction.

> **Note:** This range is a practical model-based valuation band and should not be interpreted as a formal statistical confidence interval.

---

## 🖥️ Application Design

The application uses a professional **Navy & Gold real estate theme** with:

- Branded header
- Company logo support
- Property specification cards
- Sidebar input controls
- Valuation metric cards
- Total-price breakdown
- Price-per-square-foot breakdown
- Responsive wide-screen layout

The application dynamically calculates transaction-year, quarter, month, and market-age features before generating a prediction.

---

## 🛠️ Tech Stack

### Programming & Analysis

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- LightGBM
- XGBoost
- Category Encoders
- Joblib

### Deployment

- Streamlit

### Data Processing

- Data cleaning
- Feature engineering
- Target encoding
- Log transformation
- Model comparison
- Hyperparameter tuning
- Cross-validation

---

## 📁 Project Structure

```text
365-Real-Estate/
│
├── dld_transactions.csv
├── df_clean.csv
├── uae_real_estate_avm_tuned_model.joblib
├── app.py
├── notebook.ipynb
├── logo.png
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd 365-Real-Estate
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost category_encoders streamlit joblib openpyxl
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📌 Key Business Insights

### 1. Location has a major influence on valuation

Dubai communities showed substantial differences in both median property price and price per square foot, making location-related features critical to the valuation model.

### 2. Price increases with property size, but size alone is not sufficient

Larger properties generally command higher total prices, but properties of similar sizes can have significantly different values depending on location and other characteristics.

### 3. Luxury properties are difficult to model

High-value villas, penthouses, and other luxury transactions create a much wider pricing distribution and are harder to predict accurately.

### 4. Market timing matters

The increase in median property values and price per square foot from 2020 through 2025 demonstrates the importance of transaction timing in the valuation process.

### 5. Log transformation substantially improved model behavior

Transforming the target from raw property price to log(price) reduced the impact of extreme luxury transactions and dramatically improved baseline linear-model performance.

---

## ⚠️ Model Limitations

Although the final model achieved strong predictive performance, several factors were not available in the dataset.

The model does not directly capture:

- View orientation
- Floor elevation
- Natural lighting
- Renovation or interior quality
- Individual building quality
- Developer-specific quality at building level
- Seller motivation or distressed transactions
- Off-plan payment-plan structures

Identical layouts within the same tower can also vary significantly in value because of these property-level characteristics.

### Deployment consideration

The historical modeling dataset covers **2020–2025**, while the Streamlit application dynamically uses the current year and month as prediction inputs. As a result, predictions made after 2025 involve extrapolation of the time-related features beyond the training period.

For production deployment, the model should be regularly retrained using the latest transaction data.

---

## 🔮 Future Improvements

Potential improvements include:

- Adding building-level features
- Adding floor number and view orientation
- Incorporating developer information
- Adding renovation and property-condition variables
- Incorporating geospatial distance features
- Adding recent market indicators
- Retraining the model periodically
- Developing confidence/uncertainty estimates
- Expanding the application with historical valuation comparisons

---

## ✅ Project Outcome

This project demonstrates a complete machine learning workflow from **raw real estate transaction data to a deployed valuation application**.

The final tuned LightGBM model achieved a **94.64% R²** with **13.52% MAPE** and an approximate **AED 268K MAE**, providing a practical foundation for automated residential property valuation in Dubai.

---

## 👨‍💻 Author

**Muhammad Rohail**

Data Analyst | Business Intelligence | Machine Learning