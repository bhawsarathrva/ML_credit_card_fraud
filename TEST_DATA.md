# Applying the below test suits or you can give your own test suits
# Test Data for Credit Card Fraud Detection Model

## Feature Order (23 features):
LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6

---

## TEST CASE 1: HIGH RISK - LIKELY FRAUD ⚠️
**Profile:** High credit limit, young male, delayed payments, maxed out credit, minimum payments only

### Values:
```
LIMIT_BAL: 500000
SEX: 1
EDUCATION: 3
MARRIAGE: 2
AGE: 25
PAY_0: 2
PAY_2: 2
PAY_3: 1
PAY_4: 1
PAY_5: 0
PAY_6: 0
BILL_AMT1: 480000
BILL_AMT2: 475000
BILL_AMT3: 470000
BILL_AMT4: 450000
BILL_AMT5: 430000
BILL_AMT6: 420000
PAY_AMT1: 5000
PAY_AMT2: 5000
PAY_AMT3: 4500
PAY_AMT4: 4000
PAY_AMT5: 3500
PAY_AMT6: 3000
```

**Why Fraud?**
- Payment delays increasing (PAY_0=2, PAY_2=2, PAY_3=1)
- Credit nearly maxed out (480K/500K = 96%)
- Very low payment amounts (~1% of bill)
- Increasing debt over time
- Young age, high risk profile

---

## TEST CASE 2: LOW RISK - LEGITIMATE ✅
**Profile:** Moderate credit limit, mature professional, always pays in full

### Values:
```
LIMIT_BAL: 100000
SEX: 2
EDUCATION: 2
MARRIAGE: 1
AGE: 42
PAY_0: -1
PAY_2: -1
PAY_3: -1
PAY_4: -1
PAY_5: -1
PAY_6: -1
BILL_AMT1: 25000
BILL_AMT2: 23000
BILL_AMT3: 28000
BILL_AMT4: 22000
BILL_AMT5: 26000
BILL_AMT6: 24000
PAY_AMT1: 25000
PAY_AMT2: 23000
PAY_AMT3: 28000
PAY_AMT4: 22000
PAY_AMT5: 26000
PAY_AMT6: 24000
```

**Why Not Fraud?**
- All payments on time (-1 = paid in full)
- Moderate credit usage (25K/100K = 25%)
- Pays full balance every month
- Stable payment history
- Mature age, married, educated

---

## TEST CASE 3: MEDIUM RISK - RISKY BEHAVIOR ⚠️
**Profile:** Moderate limit, inconsistent payments, revolving credit

### Values:
```
LIMIT_BAL: 200000
SEX: 1
EDUCATION: 2
MARRIAGE: 2
AGE: 35
PAY_0: 1
PAY_2: 0
PAY_3: 1
PAY_4: 0
PAY_5: -1
PAY_6: 0
BILL_AMT1: 150000
BILL_AMT2: 145000
BILL_AMT3: 140000
BILL_AMT4: 135000
BILL_AMT5: 130000
BILL_AMT6: 125000
PAY_AMT1: 10000
PAY_AMT2: 12000
PAY_AMT3: 8000
PAY_AMT4: 15000
PAY_AMT5: 20000
PAY_AMT6: 10000
```

**Why Medium Risk?**
- Some payment delays (PAY_0=1, PAY_3=1)
- High credit usage (75%)
- Inconsistent payment amounts
- Increasing debt trend

---

## TEST CASE 4: EXTREME FRAUD - VERY HIGH RISK 🚨
**Profile:** Maxed out credit, consistent late payments, minimal payments

### Values:
```
LIMIT_BAL: 300000
SEX: 1
EDUCATION: 3
MARRIAGE: 3
AGE: 28
PAY_0: 3
PAY_2: 3
PAY_3: 2
PAY_4: 2
PAY_5: 1
PAY_6: 1
BILL_AMT1: 298000
BILL_AMT2: 295000
BILL_AMT3: 290000
BILL_AMT4: 285000
BILL_AMT5: 280000
BILL_AMT6: 275000
PAY_AMT1: 2000
PAY_AMT2: 2500
PAY_AMT3: 3000
PAY_AMT4: 2000
PAY_AMT5: 1500
PAY_AMT6: 2000
```

**Why Extreme Fraud?**
- 3 months delayed payments (PAY_0=3, PAY_2=3)
- 99% credit utilization
- Payments less than 1% of bill
- Consistently late over 6 months
- Debt increasing every month

---

## TEST CASE 5: LOW RISK - RESPONSIBLE YOUNG USER ✅
**Profile:** Low limit student, always pays on time

### Values:
```
LIMIT_BAL: 50000
SEX: 2
EDUCATION: 1
MARRIAGE: 2
AGE: 24
PAY_0: -1
PAY_2: -1
PAY_3: -1
PAY_4: 0
PAY_5: 0
PAY_6: -1
BILL_AMT1: 8000
BILL_AMT2: 7500
BILL_AMT3: 9000
BILL_AMT4: 8500
BILL_AMT5: 7000
BILL_AMT6: 6500
PAY_AMT1: 8000
PAY_AMT2: 7500
PAY_AMT3: 9000
PAY_AMT4: 8500
PAY_AMT5: 7000
PAY_AMT6: 6500
```

**Why Not Fraud?**
- Mostly pays in full
- Low credit usage (16-18%)
- Responsible despite young age
- Recent months all paid in full
- Graduate student (Education=1)

---

## CSV FORMAT (Easy Copy-Paste)

```csv
LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6,Expected_Result
500000,1,3,2,25,2,2,1,1,0,0,480000,475000,470000,450000,430000,420000,5000,5000,4500,4000,3500,3000,FRAUD
100000,2,2,1,42,-1,-1,-1,-1,-1,-1,25000,23000,28000,22000,26000,24000,25000,23000,28000,22000,26000,24000,NON-FRAUD
200000,1,2,2,35,1,0,1,0,-1,0,150000,145000,140000,135000,130000,125000,10000,12000,8000,15000,20000,10000,MEDIUM-RISK
300000,1,3,3,28,3,3,2,2,1,1,298000,295000,290000,285000,280000,275000,2000,2500,3000,2000,1500,2000,FRAUD
50000,2,1,2,24,-1,-1,-1,0,0,-1,8000,7500,9000,8500,7000,6500,8000,7500,9000,8500,7000,6500,NON-FRAUD
```

---

## QUICK TEST VALUES (For Web Form)

### 🚨 FRAUD Example (Copy these values):
```
50000, 1, 3, 2, 25, 2, 2, 1, 1, 0, 0, 480000, 475000, 470000, 450000, 430000, 420000, 5000, 5000, 4500, 4000, 3500, 3000
```

### ✅ NON-FRAUD Example (Copy these values):
```
100000, 2, 2, 1, 42, -1, -1, -1, -1, -1, -1, 25000, 23000, 28000, 22000, 26000, 24000, 25000, 23000, 28000, 22000, 26000, 24000
```

---

## Feature Explanation

| Feature | Description | Values |
|---------|-------------|--------|
| LIMIT_BAL | Credit limit | 10,000 - 700,000 |
| SEX | Gender | 1=Male, 2=Female |
| EDUCATION | Education level | 1=Grad, 2=Uni, 3=HS, 4=Others |
| MARRIAGE | Marital status | 1=Married, 2=Single, 3=Others |
| AGE | Customer age | 21-75 years |
| PAY_0 | Payment status Sept | -1=Paid full, 0=Revolve, 1-9=Months delayed |
| PAY_2 | Payment status Aug | Same as above |
| PAY_3 | Payment status July | Same as above |
| PAY_4 | Payment status June | Same as above |
| PAY_5 | Payment status May | Same as above |
| PAY_6 | Payment status April | Same as above |
| BILL_AMT1-6 | Bill amounts (6 months) | Bill statement amount |
| PAY_AMT1-6 | Payment amounts (6 months) | Actual payment made |

---

## How to Use

1. **For Web Form Testing:**
   - Copy values from FRAUD or NON-FRAUD examples above
   - Paste into each field in order
   - Click "Predict Fraud Risk"

2. **For CSV Upload Testing:**
   - Copy the CSV format above
   - Save as `test_data.csv`
   - Upload to your app

3. **For API Testing:**
   ```bash
   curl -X POST http://localhost:5000/predict \
     -d "LIMIT_BAL=500000&SEX=1&EDUCATION=3..."
   ```

---

## Expected Model Predictions

| Test Case | Expected Prediction | Confidence |
|-----------|-------------------|------------|
| Test Case 1 | **1 (FRAUD)** | High (70-90%) |
| Test Case 2 | **0 (NON-FRAUD)** | High (80-95%) |
| Test Case 3 | **0 or 1** | Medium (50-70%) |
| Test Case 4 | **1 (FRAUD)** | Very High (85-95%) |
| Test Case 5 | **0 (NON-FRAUD)** | High (75-90%) |

---

**Use these test cases to verify your model is working correctly!** 🎯
