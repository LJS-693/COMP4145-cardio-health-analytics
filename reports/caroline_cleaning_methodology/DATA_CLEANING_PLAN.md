# 📊 Data Cleaning & Preprocessing Plan

## Executive Summary

After analyzing the existing datasets in `original_work/`, I've identified **critical data quality issues** that violate fundamental data science principles. This document outlines a **systematic, evidence-based cleaning approach** aligned with industry best practices.

---

## 🔍 Issues Identified in Current Data

### Dataset Overview
- **cardio_train.csv**: 70,000 records (raw data)
- **cardio_cleaned.csv**: 69,984 records (already normalized, but still has issues)
- **heart.csv**: 303 records (different dataset, much smaller)

### Critical Data Quality Problems

#### 1. **Extreme Outliers (3.91% of data affected)**
| Issue | Count | Medical Plausibility |
|-------|-------|---------------------|
| Negative blood pressure | 8 | ❌ Physically impossible |
| Extreme BP (sys>250 or dia>200) | 993 | ⚠️ Life-threatening range |
| Extreme low BP (sys<80 or dia<40) | 260 | ⚠️ Hypotensive shock range |
| Diastolic ≥ Systolic | 1,236 | ❌ Physiologically impossible |
| Suspicious heights (<130cm or >220cm) | 93 | ⚠️ Outside normal adult range |
| Suspicious weights (<40kg or >200kg) | 52 | ⚠️ Extreme malnutrition or obesity |
| BMI outliers (<15 or >60) | 93 | ⚠️ Medical emergency range |

**Total rows with issues: 2,735 (3.91%)**

#### 2. **Data Imbalance Issues**
- **cardio_train.csv**: 50.03% negative, 49.97% positive ✅ **Well balanced**
- **cardio_cleaned.csv**: 50.51% negative, 49.49% positive ✅ **Still balanced after cleaning**
- **heart.csv**: 45.45% negative, 54.46% positive ⚠️ **Slight imbalance**

#### 3. **Inconsistent Data Ranges**
- **cardio_cleaned.csv**: Already normalized to [0,1] range
- **cardio_train.csv**: Raw values with different scales
  - Age: days (10,798 - 23,713)
  - Height: cm (55 - 250)
  - Weight: kg (10 - 200)
  - BP: mmHg (-150 to 16,020!)

#### 4. **Mixed Datasets Problem**
- Using **TWO DIFFERENT** cardiovascular datasets (cardio_train vs heart.csv)
- Different features, different populations, different medical contexts
- **Recommendation**: Focus on ONE dataset for consistency

---

## 🎯 Recommended Data Cleaning Approach

### Strategy: **Conservative Medical-Based Filtering**

Following **clinical guidelines** and **statistical rigor**, not arbitrary thresholds.

### Method Selection Rationale

#### Why Not Use Existing Cleaned Data?
1. ❌ **Still contains outliers**: The normalized data doesn't remove impossible values
2. ❌ **Premature normalization**: Data was normalized BEFORE outlier removal
3. ❌ **Loss of interpretability**: Can't validate medical plausibility in normalized form
4. ❌ **Unknown methodology**: No documentation of what cleaning was done

#### Recommended Approach: **Domain-Informed IQR Method**

**Principle**: Combine medical domain knowledge with statistical outlier detection

---

## 📋 Detailed Cleaning Pipeline

### **Stage 1: Remove Physiologically Impossible Records**

These are **measurement errors**, not real data:

```python
# Criteria based on medical literature
Remove if:
  - ap_hi < 0 OR ap_lo < 0  (negative BP impossible)
  - ap_lo >= ap_hi  (diastolic cannot exceed systolic)
  - height < 100 OR height > 230  (outside human range)
  - weight < 30 OR weight > 300  (outside survivable range)
  - age_years < 18 OR age_years > 100  (dataset context: adults only)
```

**Justification**: Medical impossibility, likely data entry errors

### **Stage 2: Remove Extreme Clinical Outliers**

Based on **WHO/AHA clinical guidelines**:

```python
# Blood Pressure (AHA Guidelines)
Remove if:
  - ap_hi < 70 OR ap_hi > 250  (hypertensive crisis/shock)
  - ap_lo < 40 OR ap_lo > 150  (beyond emergency ranges)

# BMI (WHO Classification)
Calculate: BMI = weight / (height/100)^2
Remove if:
  - BMI < 12 OR BMI > 70  (severe malnutrition/morbid obesity extremes)
  
# Height (Adult population norms)
Remove if:
  - height < 130 OR height > 220  (outside 99.9th percentile)
```

**Justification**: While technically possible, these represent extreme medical emergencies or errors

### **Stage 3: Feature Engineering**

Create clinically meaningful derived features:

```python
# Derived Features
1. BMI = weight / (height/100)^2
2. Pulse Pressure = ap_hi - ap_lo  (cardiovascular health indicator)
3. MAP (Mean Arterial Pressure) = ap_lo + (Pulse Pressure / 3)
4. Age in years = age / 365.25
5. BMI_category = categorize(BMI)  # Underweight/Normal/Overweight/Obese
```

### **Stage 4: Encode Categorical Variables**

```python
# Already encoded in dataset:
- gender: 1=female, 2=male → Convert to: 0=female, 1=male
- cholesterol: 1=normal, 2=above normal, 3=well above normal
- gluc: 1=normal, 2=above normal, 3=well above normal
- smoke, alco, active: Already 0/1

# One-hot encode:
- cholesterol → cholesterol_1, cholesterol_2, cholesterol_3
- gluc → gluc_1, gluc_2, gluc_3
```

### **Stage 5: Normalization**

**Method**: MinMaxScaler [0,1] for consistency with neural networks

```python
# Normalize continuous features:
- age, height, weight, ap_hi, ap_lo, bmi, pulse_pressure, MAP

# Keep binary/categorical as-is:
- gender, smoke, alco, active, cholesterol_*, gluc_*
```

**Why MinMaxScaler?**
- ✅ Preserves zero values (important for binary features)
- ✅ Bounded range good for neural networks
- ✅ Interpretable (0=min, 1=max)
- ⚠️ Sensitive to outliers (but we cleaned those in Stages 1-2)

**Alternative considered**: StandardScaler
- ✅ Less sensitive to outliers
- ❌ Unbounded range (can go negative)
- ❌ Less intuitive for presentation

### **Stage 6: Handle Class Imbalance (if needed)**

Current balance: **50.03% / 49.97%** ✅ **NO RESAMPLING NEEDED**

**Decision**: Keep as-is
- Already perfectly balanced
- No SMOTE, no undersampling, no class weights needed
- Resampling would be unnecessary and could introduce synthetic bias

### **Stage 7: Train-Test Split**

**80-20 Split Strategy**:
```python
# Parameters
- test_size = 0.20
- random_state = 42 (for reproducibility)
- stratify = y (maintain class balance in both sets)

# Expected output:
- Train: ~56,000 samples (after cleaning)
- Test: ~14,000 samples (after cleaning)
```

**Why 80-20?**
- ✅ Industry standard for large datasets (>10k samples)
- ✅ Enough test data for statistical significance
- ✅ Maximum training data for model learning

**Alternative considered**: 70-30
- Used when dataset is smaller (<5k)
- Our dataset is large enough for 80-20

---

## 📁 Deliverables

### Three Files to Generate

#### 1. **`cardio_fully_cleaned.csv`**
- Complete cleaned dataset
- All stages applied
- ALL features included (original + engineered)
- Ready for EDA and feature selection

#### 2. **`cardio_train_80.csv`**
- 80% of cleaned data
- Stratified sampling (class balance preserved)
- For model training

#### 3. **`cardio_test_20.csv`**
- 20% of cleaned data
- Stratified sampling
- For final model evaluation (DO NOT use for hyperparameter tuning)

### Additional Documentation

#### 4. **`data_cleaning_report.md`**
- Detailed statistics before/after each stage
- Number of rows removed and why
- Final dataset characteristics
- Validation checks

#### 5. **`data_cleaning_code.py`**
- Complete, reproducible Python script
- Well-documented with comments
- Section-by-section corresponding to stages above

---

## 🔬 Scientific Principles Applied

### 1. **Domain-Driven Filtering**
✅ Using medical literature (AHA, WHO guidelines)  
✅ Not arbitrary statistical cutoffs  
✅ Interpretable to healthcare stakeholders

### 2. **Reproducibility**
✅ Fixed random seeds  
✅ Documented thresholds  
✅ Version-controlled code

### 3. **Transparency**
✅ Document every row removed  
✅ Justify every threshold  
✅ Report before/after statistics

### 4. **Conservative Approach**
✅ Only remove clear errors/impossibilities  
✅ Keep borderline cases (let model learn)  
✅ Avoid overfitting to "clean" data

### 5. **Clinical Validity**
✅ Consult medical ranges  
✅ Validate feature engineering  
✅ Ensure predictions are actionable

---

## 📊 Expected Outcomes

### Before Cleaning
- **70,000 rows**
- 3.91% with data quality issues
- Raw, unscaled features
- Mixture of impossible values

### After Cleaning
- **~67,000-68,000 rows** (estimated)
- 0% impossible values
- 0% extreme clinical outliers
- Normalized features [0,1]
- Engineered features (BMI, MAP, etc.)
- Perfectly balanced classes

### Quality Assurance Checks
```python
# Validation after cleaning:
1. No negative BP values
2. No ap_lo >= ap_hi
3. All heights in [130, 220]
4. All weights in [40, 200]
5. All BMI in [12, 70]
6. All BP in clinical ranges
7. Class balance maintained (49-51% each)
8. No missing values
9. All features in [0, 1] or binary
```

---

## 🎯 Why This Approach is Superior

### Compared to Existing `cardio_cleaned.csv`:

| Aspect | Existing Cleaned | Our Approach |
|--------|------------------|--------------|
| Outlier removal | ❌ Unknown | ✅ Documented medical criteria |
| Feature engineering | ❌ None | ✅ BMI, MAP, Pulse Pressure |
| Normalization timing | ❌ Before outlier removal | ✅ After outlier removal |
| Documentation | ❌ None | ✅ Complete audit trail |
| Medical validity | ❌ Unverified | ✅ Clinical guidelines-based |
| Reproducibility | ❌ Unknown process | ✅ Fixed seeds, versioned code |

---

## 👥 Team Collaboration Note

**For the team member handling descriptions/documentation:**

This cleaning approach gives you:
1. **Clear narrative**: "We applied medical domain knowledge to remove physiologically impossible measurements"
2. **Defensible choices**: Every threshold backed by clinical guidelines
3. **Academic rigor**: Follows data science best practices from literature
4. **Visual opportunities**: Before/after distributions, outlier visualizations
5. **Ethical considerations**: Conservative approach preserves real medical variation

**Key talking points for your section:**
- "Data quality is paramount in healthcare ML"
- "We prioritized clinical validity over aggressive cleaning"
- "Our approach is reproducible and auditable"
- "Feature engineering captures domain expertise"

---

## 📚 References for Methodology

1. **American Heart Association (AHA)**: Blood Pressure Categories
2. **WHO BMI Classification**: Body Mass Index standards
3. **Scikit-learn Documentation**: Preprocessing best practices
4. **Clinical Data Science (MIT)**: Medical outlier handling
5. **CRISP-DM Methodology**: Data understanding & preparation phases

---

## ⏭️ Next Steps

1. ✅ Review this plan
2. ✅ Confirm approach with team
3. ⏳ **Execute cleaning script** (ready to code)
4. ⏳ **Generate 3 output files**
5. ⏳ **Create detailed cleaning report**
6. ⏳ **Validate results**
7. ⏳ **Hand off to modeling phase**

---

**Status**: 📋 Plan Ready for Approval  
**Estimated Execution Time**: 15-20 minutes  
**Risk Level**: Low (conservative approach, well-documented)  

**Author**: Data Cleaning Team  
**Date**: 2024-11-23  
**Version**: 1.0
