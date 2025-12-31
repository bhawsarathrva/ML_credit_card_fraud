# 📊 Credit Card Fraud Dataset - Complete Description

## Dataset Overview

**File:** `creditCardFraud_Data.csv`
- **Total Records:** 1,001 credit card transactions
- **Total Features:** 24 columns (23 feature columns + 1 target column)
- **Data Type:** All columns are integer type (int64)
- **Class Distribution:**
  - Non-Fraud (0): 787 transactions (78.62%)
  - Fraud (1): 214 transactions (21.38%)
  - **Imbalance Ratio:** ~3.7:1 (moderately imbalanced)

---

## Column-by-Column Explanation

### 1. **LIMIT_BAL** (Credit Limit)
**Type:** Numerical (int64)  
**Range:** 10,000 to 700,000  
**Mean:** 167,532  
**Information:**
- Amount of the credit limit given to the customer (in NT dollars)
- Represents the maximum amount a customer can borrow
- Higher limits often given to customers with better credit scores
- **Fraud Indicator:** Fraudsters may prefer accounts with higher limits

---

### 2. **SEX** (Gender)
**Type:** Categorical (encoded as int)  
**Values:**
- 1 = Male
- 2 = Female

**Distribution:** Mean ~1.59 (more females than males)  
**Information:**
- Demographic feature indicating customer's gender
- Used to understand behavioral patterns
- **Fraud Indicator:** Minimal direct correlation, but useful for demographic segmentation

---

### 3. **EDUCATION** (Education Level)
**Type:** Categorical (encoded as int)  
**Values:**
- 1 = Graduate school
- 2 = University
- 3 = High school
- 4 = Others
- 5, 6 = Unknown

**Mean:** 1.78 (most customers have university/graduate education)  
**Information:**
- Educational background of the customer
- Can indicate financial literacy and stability
- **Fraud Indicator:** Education level may correlate with fraud patterns

---

### 4. **MARRIAGE** (Marital Status)
**Type:** Categorical (encoded as int)  
**Values:**
- 1 = Married
- 2 = Single
- 3 = Others

**Mean:** 1.60 (mix of married and single)  
**Information:**
- Marital status of the customer
- Affects spending patterns and financial responsibility
- **Fraud Indicator:** Different spending patterns based on marital status

---

### 5. **AGE** (Customer Age)
**Type:** Numerical (int64)  
**Range:** 21 to 75 years  
**Mean:** 34.9 years  
**25th percentile:** 28 years  
**75th percentile:** 41 years

**Information:**
- Age of the credit card holder
- Younger customers may have different risk profiles
- **Fraud Indicator:** Age groups show different fraud susceptibility patterns

---

### 6-11. **PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6** (Repayment Status)
**Type:** Categorical (encoded as int)  
**Meaning:** Payment status for the past 6 months
- PAY_0 = Repayment status in September 2005
- PAY_2 = Repayment status in August 2005
- PAY_3 = Repayment status in July 2005
- PAY_4 = Repayment status in June 2005
- PAY_5 = Repayment status in May 2005
- PAY_6 = Repayment status in April 2005

**Values:**
- -2 = No consumption
- -1 = Paid in full
- 0 = Revolving credit (minimum payment)
- 1 = Payment delay for 1 month
- 2 = Payment delay for 2 months
- 3 = Payment delay for 3 months
- ...
- 8 = Payment delay for 8 months
- 9 = Payment delay for 9 months and above

**Mean:** Around -0.00 to -0.31 (most customers pay on time or revolve)  
**Information:**
- Payment history is one of the **MOST IMPORTANT** features for fraud detection
- Tracks customer's payment behavior over time
- **Fraud Indicator:** 
  - Customers with consistent delays are higher risk
  - Sudden changes in payment patterns may indicate fraud
  - This is a **key predictor** of default/fraud

---

### 12-17. **BILL_AMT1 to BILL_AMT6** (Bill Statement Amount)
**Type:** Numerical (int64)  
**Meaning:** Amount of bill statement for past 6 months
- BILL_AMT1 = Bill statement in September 2005
- BILL_AMT2 = Bill statement in August 2005
- BILL_AMT3 = Bill statement in July 2005
- BILL_AMT4 = Bill statement in June 2005
- BILL_AMT5 = Bill statement in May 2005
- BILL_AMT6 = Bill statement in April 2005

**Range:** -165,580 to 628,699 (negative values = overpayment)  
**Mean:** Around 38,000 to 49,000 NT dollars  
**Information:**
- Shows how much the customer owes each month
- Tracks spending patterns over time
- **Fraud Indicator:**
  - Sudden spikes in bill amounts may indicate fraud
  - Consistent high bills with low payments = risk
  - Pattern changes are important

---

### 18-23. **PAY_AMT1 to PAY_AMT6** (Previous Payment Amount)
**Type:** Numerical (int64)  
**Meaning:** Amount of previous payment for past 6 months
- PAY_AMT1 = Amount paid in September 2005
- PAY_AMT2 = Amount paid in August 2005
- PAY_AMT3 = Amount paid in July 2005
- PAY_AMT4 = Amount paid in June 2005
- PAY_AMT5 = Amount paid in May 2005
- PAY_AMT6 = Amount paid in April 2005

**Range:** 0 to 528,666 NT dollars  
**Mean:** Around 4,000 to 5,400 NT dollars  
**Information:**
- Actual amount the customer paid each month
- Shows payment capacity and willingness
- **Fraud Indicator:**
  - Low payments despite high bills = risk
  - Ratio of payment to bill amount is important
  - Payment patterns help identify creditworthiness

---

### 24. **default payment next month** (TARGET VARIABLE)
**Type:** Binary (int64)  
**Values:**
- 0 = No default (Non-Fraud) - 787 cases (78.62%)
- 1 = Default (Fraud) - 214 cases (21.38%)

**Information:**
- **This is what we're trying to predict**
- Indicates whether the customer will default on payment next month
- Binary classification target
- **Class Imbalance:** The dataset is imbalanced (3.7:1 ratio)
- **Why SMOTE is needed:** To balance this distribution for better fraud detection

---

## Key Dataset Characteristics

### 1. **Temporal Nature**
- Data covers 6 months of history (April to September 2005)
- Allows for time-series pattern analysis
- Payment behavior tracked over time

### 2. **Feature Types**
- **Demographic Features:** SEX, EDUCATION, MARRIAGE, AGE, LIMIT_BAL
- **Behavioral Features:** PAY_0 to PAY_6 (payment status)
- **Financial Features:** BILL_AMT1-6, PAY_AMT1-6 (amounts)

### 3. **Class Imbalance Challenge**
- Only 21.38% are fraud cases
- Requires special techniques (SMOTE) to handle
- Standard ML models would be biased toward non-fraud

### 4. **No Missing Values**
- All 1,001 records are complete
- No need for imputation
- Clean dataset ready for modeling

---

## Most Important Features for Fraud Detection

Based on domain knowledge and typical credit card fraud patterns:

### **High Importance:**
1. **PAY_0 to PAY_6** - Payment history is the strongest predictor
2. **BILL_AMT1 to BILL_AMT6** - Spending patterns
3. **PAY_AMT1 to PAY_AMT6** - Payment behavior

### **Medium Importance:**
4. **LIMIT_BAL** - Credit limit indicates risk tolerance
5. **AGE** - Different age groups have different risk profiles

### **Low Importance:**
6. **EDUCATION** - Indirect impact on financial behavior
7. **MARRIAGE** - Affects spending but not strongly correlated with fraud
8. **SEX** - Minimal direct correlation

---

## Real-World Interpretation

### **What This Dataset Represents:**
This is a **credit card default prediction** dataset, where:
- Each row = One credit card customer
- Each customer has 6 months of transaction and payment history
- Goal: Predict if they will default (fail to pay) next month

### **Business Use Case:**
Banks use this type of data to:
1. **Risk Assessment:** Identify high-risk customers before issuing credit
2. **Early Warning:** Detect customers likely to default
3. **Credit Limit Adjustment:** Lower limits for risky customers
4. **Collection Strategy:** Prioritize follow-up on high-risk accounts
5. **Fraud Prevention:** Catch suspicious payment patterns

### **Why Machine Learning Helps:**
- Too complex for simple rules (23 features, non-linear relationships)
- Patterns in payment history reveal creditworthiness
- ML can detect subtle anomalies humans might miss
- Can process thousands of applications quickly

---

## Data Quality Notes

### ✅ **Strengths:**
- Clean data with no missing values
- Balanced feature types (demographic + behavioral + financial)
- Temporal dimension (6 months history)
- Real-world dataset from actual credit card data

### ⚠️ **Challenges:**
- **Class imbalance** (78% vs 21%) - requires SMOTE
- Small dataset (only 1,001 samples) - may limit model generalization
- Anonymized features - hard to interpret some patterns
- Temporal data from 2005 - patterns may have changed

---

## Recommended Preprocessing Steps

1. **Handle Class Imbalance** ✅
   - Apply SMOTE to balance the dataset
   - Alternative: Use class weights in models

2. **Feature Scaling** ✅
   - Use RobustScaler (better for outliers)
   - Scale BILL_AMT and PAY_AMT features

3. **Feature Engineering** (Optional)
   - Create ratio features: PAY_AMT / BILL_AMT
   - Calculate payment consistency metrics
   - Derive trend features from 6-month history

4. **Train-Test Split** ✅
   - Use stratified split to maintain class distribution
   - 80-20 or 70-30 split recommended

---

## Summary

This credit card fraud dataset contains **1,001 customer records** with **23 features** covering:
- **Demographics** (age, gender, education, marriage)
- **Credit information** (credit limit)
- **Payment history** (6 months of payment status)
- **Bill amounts** (6 months of billing)
- **Payment amounts** (6 months of actual payments)

The goal is to predict **default payment next month** (fraud/non-fraud).

**Key Challenge:** Class imbalance (21% fraud) requires SMOTE.

**Key Features:** Payment history (PAY_0 to PAY_6) and financial amounts (BILL_AMT, PAY_AMT) are most predictive.

**Perfect for:** Binary classification, imbalanced learning, fraud detection, and credit risk assessment.

---

**This dataset is now ready for machine learning model training!** 🚀
