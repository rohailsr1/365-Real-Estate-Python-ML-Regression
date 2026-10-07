# 🏢 Dubai Real Estate Valuation Engine | Machine Learning & Streamlit

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Regression-9ACD32)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)

# 🎯 Business Problem

Real estate prices can vary significantly depending on:

- Property size
- Community
- Property type
- Bedroom/layout
- Ready vs. off-plan status
- Market conditions
- Nearby landmarks and malls
- Transaction period

The objective of this project was to develop a data-driven **Automated Valuation Model (AVM)** capable of learning these relationships and generating practical property-value estimates.

### Objectives

- Analyze historical Dubai real estate transactions
- Identify major pricing patterns
- Clean and standardize inconsistent transaction data
- Engineer machine learning features
- Compare multiple regression algorithms
- Optimize the strongest-performing model
- Deploy the final model through Streamlit

---

# 📊 Exploratory Data Analysis

## 🎯 Distribution of Target Variables

The original property-value distribution was highly right-skewed because of high-value and luxury transactions.

- Raw `actual_worth` skewness: **6.47**
- Log-transformed target skewness: **0.58**

A logarithmic transformation was therefore applied to the target variable.

<p align="center">
  <img src="Images/Distribution%20of%20Target%20Variables.png" width="850">
</p>

### Insight

The log transformation substantially reduced the influence of extreme luxury transactions and created a more balanced target distribution for machine learning.

---

## 🏠 Bedroom Count vs Price

The relationship between bedroom/layout categories and property value was analyzed to understand how property configuration affects price.

<p align="center">
  <img src="Images/Bedroom%20Count%20VS%20Price.png" width="850">
</p>

### Insight

Properties with more bedrooms generally commanded higher total prices. However, significant variation remained within individual bedroom categories, showing that **location, property size, property type and other characteristics also influence valuation**.

Luxury layouts such as penthouses also showed considerably wider price distributions.

---

## 📐 Size vs Price

Property size was compared against actual transaction value.

<p align="center">
  <img src="Images/Size%20VS%20Price.png" width="850">
</p>

### Insight

Larger properties generally had higher transaction values, but the relationship was not purely linear.

Properties with similar sizes could still have substantially different values, highlighting the importance of **location and market-specific characteristics**.

---

## 🗺️ Price & Price per Square Foot by Community

Community-level analysis was performed using median property price and median price per square foot.

<p align="center">
  <img src="Images/Price%20and%20price%20per%20square%20foot%20by%20community.png" width="900">
</p>

### Key Insights

- **Palm Jumeirah** recorded the highest median property values and was among the strongest markets in price per square foot.
- **Downtown Dubai** ranked among the highest communities by price per square foot.
- **Dubai Hills Estate** showed high median property values while ranking differently on price per square foot, demonstrating the effect of property size.
- **International City, Dubai Production City, and Dubai Silicon Oasis** were among the lower-priced communities.

### Business Takeaway

Location is one of the most important dimensions of Dubai property valuation, and community-level differences must be captured by an automated valuation model.

---

## 📈 Property Price Movement Across Six Years

Transaction trends were analyzed from **2020 to 2025**.

<p align="center">
  <img src="Images/Price%20Movement%20Across%20The%20Six%20Years.png" width="900">
</p>

### Key Insights

Median property price increased from approximately:

**AED 1.1M → AED 1.92M+**

Median price per square foot increased from approximately:

**AED 922 → AED 1,632**

Transaction volume peaked in **2024 at approximately 16,816 transactions**.

### Modeling Implication

These market movements reinforced the importance of including:

- Transaction Year
- Transaction Quarter
- Transaction Month
- Market Age

as predictive features.

---

## 🏗️ Off-Plan vs Ready Properties

Ready and off-plan properties were compared to understand differences in pricing.

<p align="center">
  <img src="Images/OFF%20Plan%20VS%20Ready.png" width="850">
</p>

### Insight

Ready properties showed slightly higher median prices and price-per-square-foot levels in the analyzed dataset, while both categories contained significant high-value outliers.

This supported the inclusion of **registration status** as a machine learning feature.

---

# 🧹 Data Preparation

The project started with:

### **172,704 Raw Transactions**

The dataset was filtered to focus on the target residential population.

### Filtering Criteria

1. Sales transactions only
2. Residential properties only
3. Unit and Villa property types
4. Positive transaction value
5. Positive property area
6. Removal of non-target property subtypes such as:
   - Office
   - Shop
   - Hotel Apartment

After filtering and data-quality treatment:

### **82,178 Transactions**

remained for analysis and modeling.

### Data Quality Treatment

- Duplicate checks
- Missing-value analysis
- Data-type validation
- Removal of completely empty columns
- Missing categorical-value handling
- Removal of records missing critical pricing/layout information
- Categorical standardization
- Community-name normalization

Columns such as `rent_value` and `meter_rent_price`, which were completely missing, were removed.

---

# 📍 Location Standardization

Dubai Land Department area names were mapped into more recognizable market communities.

| Original DLD Area | Market Community |
|---|---|
| Al Barsha South Fourth | Jumeirah Village Circle (JVC) |
| Marsa Dubai | Dubai Marina |
| Burj Khalifa | Downtown Dubai |
| Al Merkadh | Mohammed Bin Rashid City (MBR) |
| Al Thanyah Fifth | Jumeirah Lakes Towers (JLT) |
| Hadaeq Sheikh Mohammed Bin Rashid | Dubai Hills Estate |
| Nadd Hessa | Dubai Silicon Oasis |
| Me’Aisem First | Dubai Production City |
| Al Hebiah Fourth | Jumeirah Village Triangle (JVT) |
| Al Khairan First | Dubai Creek Harbour |

This standardization addressed inconsistencies caused by differences in naming, capitalization, and spacing.

---

# ⚙️ Feature Engineering

The model used engineered numerical and categorical features.

### Numerical Features

```text
log_size_sqft
rooms_ordinal
is_off_plan
transaction_year
transaction_quarter
transaction_month
market_age_months
```

### Categorical Features

```text
property_type_en
market_community_name
nearest_landmark_en
nearest_mall_en
```

### Log Transformation

```python
log_actual_worth = np.log(actual_worth)
log_size_sqft = np.log(size_sqft)
```

The target was modeled on the **log scale** rather than raw property value.

### Bedroom Encoding

Bedroom/layout categories were converted into an ordinal representation:

```text
Studio        → 0
Single Room   → 1
1 B/R         → 2
2 B/R         → 3
2 B/R+M       → 4
3 B/R         → 5
3 B/R+M       → 6
4 B/R         → 7
4 B/R+M       → 8
5 B/R         → 9
5 B/R+M       → 10
6 B/R         → 11
Penthouse     → 12
```

---

# 🤖 Machine Learning

Several regression algorithms were evaluated:

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

The dataset was divided into:

**80% Training / 20% Testing**

using:

```python
random_state = 42
```

High-cardinality categorical variables were handled using **Target Encoding**.

---

# 🧪 Model Development

## Raw Price vs Log Price

One of the major experiments compared Linear Regression using raw property value against a log-transformed target.

| Model | MAPE |
|---|---:|
| Linear Regression – Raw Price | 75.60% |
| Linear Regression – Log Price | **18.23%** |

### Key Finding

Applying a log transformation reduced MAPE by:

### **57.37 percentage points**

This demonstrated the importance of handling the heavily skewed real estate price distribution.

---

# 🏆 Final Model: Tuned LightGBM

Among the evaluated algorithms, **LightGBM** produced the strongest overall performance.

The model was optimized using:

### RandomizedSearchCV

with:

- 5-fold cross-validation
- 8 parameter combinations
- 40 total model fits
- Negative Mean Squared Error scoring

### Tuned Parameters Included

```text
num_leaves
learning_rate
n_estimators
min_child_samples
subsample
colsample_bytree
```

---

# 📊 Model Performance

The final tuned LightGBM model achieved:

| Metric | Result |
|---|---:|
| **R²** | **94.64%** |
| **MAPE** | **13.52%** |
| **MAE** | **≈ AED 268,400** |
| **MAE %** | **15.35%** |

### Baseline vs Tuned LightGBM

| Metric | Baseline | Tuned |
|---|---:|---:|
| R² | 94.59% | **94.64%** |
| MAPE | 13.57% | **13.52%** |
| MAE % | 15.51% | **15.35%** |

<p align="center">
  <img src="Images/Running%20Model%201.png" width="850">
</p>

<p align="center">
  <img src="Images/Running%20Model%202.png" width="850">
</p>

### Model Takeaway

Hyperparameter tuning produced a modest but measurable improvement in predictive performance.

> **Note:** MAPE is used as the primary relative-error metric. `100 - MAPE` should not be interpreted as a conventional classification accuracy score.

---

# 💻 Streamlit Automated Valuation Application

The trained model was exported as:

```text
uae_real_estate_avm_tuned_model.joblib
```

A Streamlit application was built to make the model accessible to non-technical users.

## 📝 User Inputs

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

The application dynamically generates transaction-related features such as:

- Transaction Year
- Transaction Quarter
- Transaction Month
- Market Age

before generating the prediction.

---

# 💰 Valuation Output

The application generates:

## Total Property Valuation

- Conservative Estimate
- Baseline Valuation
- Optimistic Estimate

## Price per Square Foot

- Conservative Estimate
- Baseline Price / Sq. Ft.
- Optimistic Estimate

The valuation range uses the model's **13.52% MAPE** as a symmetric margin around the baseline prediction.

> This is a model-based valuation range and should not be interpreted as a formal statistical confidence interval.

---

# 🎨 Application Design

The Streamlit application was designed using a professional **Navy & Gold** real-estate theme.

### Interface Features

- Saleem Real Estate branding
- Company logo
- Sidebar property controls
- Property specification cards
- Valuation metric cards
- Total valuation breakdown
- Price-per-square-foot breakdown
- Responsive wide-screen layout

---

# 🛠️ Tech Stack

### Programming & Data Analysis

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

### Deployment

- Streamlit
- Joblib

### Core Techniques

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Log Transformation
- Target Encoding
- Regression Modeling
- Hyperparameter Tuning
- Cross-Validation
- Model Deployment

---

# 📁 Project Structure

```text
365-Real-Estate/
│
├── Images/
│   ├── Bedroom Count VS Price.png
│   ├── Distribution of Target Variables.png
│   ├── logo.png
│   ├── OFF Plan VS Ready.png
│   ├── Price and price per square foot by community.png
│   ├── Price Movement Across The Six Years.png
│   ├── Running Model 1.png
│   ├── Running Model 2.png
│   └── Size VS Price.png
│
├── dld_transactions.csv
├── df_clean.csv
├── uae_real_estate_avm_tuned_model.joblib
├── app.py
├── notebook.ipynb
└── README.md
```

---

# 🚀 How to Run

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd 365-Real-Estate
```

## 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost category_encoders streamlit joblib openpyxl
```

## 3. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 💡 Key Business Insights

### 📍 Location Matters

Dubai communities showed substantial differences in both median property value and price per square foot.

### 📐 Size Drives Total Value

Larger properties generally commanded higher total prices, but size alone could not explain the full variation.

### 🏆 Luxury Properties Are Harder to Predict

High-value villas and penthouses showed wider price distributions, increasing valuation uncertainty.

### 📈 Market Conditions Matter

Median property prices and price per square foot increased substantially between 2020 and 2025.

### 📊 Log Transformation Improved Modeling

Transforming the highly skewed property-price target substantially improved regression performance.

---

# ⚠️ Model Limitations

The model does not directly capture several property-level characteristics that can materially affect value.

These include:

- Floor number
- Floor elevation
- View orientation
- Natural lighting
- Renovation status
- Interior quality
- Building-level quality
- Developer-specific characteristics
- Seller motivation
- Distressed transactions
- Off-plan payment plans

Two properties with similar layouts in the same tower can still have materially different values because of these factors.

Luxury and penthouse transactions are also relatively sparse, making these segments more difficult to predict reliably.

---

# 🔮 Future Improvements

Potential improvements include:

- Building-level features
- Floor number
- View orientation
- Renovation status
- Property condition
- Developer information
- Geospatial distance features
- Recent market indicators
- Automated model retraining
- Prediction uncertainty estimates
- Comparable-property analysis
- More granular building/project features

---

# 🔄 End-to-End Workflow

```text
Raw DLD Transactions
        ↓
Data Cleaning & Quality Checks
        ↓
Location Standardization
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Target Encoding
        ↓
Log Transformation
        ↓
Model Comparison
        ↓
LightGBM Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Model Export
        ↓
Streamlit Deployment
        ↓
Automated Property Valuation
```

---

# 📌 Project Outcome

The project demonstrates a complete **end-to-end machine learning workflow**, from raw Dubai real estate transactions to a deployed automated valuation application.

The final **Tuned LightGBM Regressor** achieved:

### **94.64% R²**
### **13.52% MAPE**
### **≈ AED 268K MAE**

This provides a strong machine-learning foundation for automated residential property valuation in Dubai.

---

# 👨‍💻 Author

## Muhammad Rohail

**Data Analyst | Business Intelligence | Machine Learning**