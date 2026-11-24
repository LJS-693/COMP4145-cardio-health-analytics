"""
Cardiovascular Disease Data Cleaning Pipeline
==============================================

This script implements a comprehensive, medically-informed data cleaning approach
for the cardiovascular disease prediction dataset.

Author: Data Cleaning Team
Date: 2024-11-24
Version: 1.1

Methodology:
- Domain-driven filtering using clinical guidelines (WHO, AHA)
- Conservative outlier removal (only impossible/extreme values)
- Feature engineering with medical relevance
- Reproducible with fixed random seeds
- ID preservation for traceability
- Denormalization parameters export

Input: original_work/cardio_train.csv
Outputs:
  - cardio_fully_cleaned.csv (complete cleaned dataset with IDs)
  - cardio_train_80.csv (80% training split with IDs)
  - cardio_test_20.csv (20% test split with IDs)
  - denormalization_params.json (min/max values for reversing normalization)
  - data_cleaning_report.md (detailed statistics)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("CARDIOVASCULAR DISEASE DATA CLEANING PIPELINE")
print("="*80)

# =============================================================================
# STAGE 0: Load Data
# =============================================================================
print("\n📁 STAGE 0: Loading Data...")
print("-"*80)

df = pd.read_csv('../../data/raw/cardio_train.csv', sep=';')
initial_rows = len(df)
print(f"✓ Loaded {initial_rows:,} rows with {len(df.columns)} columns")
print(f"  Columns: {list(df.columns)}")

# Preserve original IDs for traceability
original_ids = df['id'].copy()
print(f"✓ Preserved {len(original_ids):,} original IDs for traceability")

# Create age in years for easier interpretation
df['age_years'] = df['age'] / 365.25

# Track cleaning statistics
cleaning_stats = {
    'initial_rows': initial_rows,
    'removed': {},
    'final_rows': 0
}

# =============================================================================
# STAGE 1: Remove Physiologically Impossible Records
# =============================================================================
print("\n🚫 STAGE 1: Removing Physiologically Impossible Records...")
print("-"*80)

df_stage1 = df.copy()

# 1.1 Negative blood pressure (physically impossible)
mask_negative_bp = (df_stage1['ap_hi'] < 0) | (df_stage1['ap_lo'] < 0)
removed_negative_bp = int(mask_negative_bp.sum())
print(f"  - Negative BP values: {removed_negative_bp} rows")
cleaning_stats['removed']['negative_bp'] = removed_negative_bp

# 1.2 Diastolic >= Systolic (physiologically impossible)
mask_bp_inverted = df_stage1['ap_lo'] >= df_stage1['ap_hi']
removed_bp_inverted = int(mask_bp_inverted.sum())
print(f"  - Diastolic ≥ Systolic: {removed_bp_inverted} rows")
cleaning_stats['removed']['bp_inverted'] = removed_bp_inverted

# 1.3 Extreme heights (outside human range)
mask_extreme_height = (df_stage1['height'] < 100) | (df_stage1['height'] > 230)
removed_extreme_height = int(mask_extreme_height.sum())
print(f"  - Extreme heights (<100 or >230 cm): {removed_extreme_height} rows")
cleaning_stats['removed']['extreme_height'] = removed_extreme_height

# 1.4 Extreme weights (outside survivable range)
mask_extreme_weight = (df_stage1['weight'] < 30) | (df_stage1['weight'] > 300)
removed_extreme_weight = int(mask_extreme_weight.sum())
print(f"  - Extreme weights (<30 or >300 kg): {removed_extreme_weight} rows")
cleaning_stats['removed']['extreme_weight'] = removed_extreme_weight

# 1.5 Age outside adult range
mask_extreme_age = (df_stage1['age_years'] < 18) | (df_stage1['age_years'] > 100)
removed_extreme_age = int(mask_extreme_age.sum())
print(f"  - Ages outside 18-100 years: {removed_extreme_age} rows")
cleaning_stats['removed']['extreme_age'] = removed_extreme_age

# Apply all Stage 1 filters
mask_stage1 = (
    ~mask_negative_bp &
    ~mask_bp_inverted &
    ~mask_extreme_height &
    ~mask_extreme_weight &
    ~mask_extreme_age
)

df_stage1 = df_stage1[mask_stage1].copy()
removed_stage1 = initial_rows - len(df_stage1)
print(f"\n✓ Stage 1 Complete: Removed {removed_stage1:,} rows ({removed_stage1/initial_rows*100:.2f}%)")
print(f"  Remaining: {len(df_stage1):,} rows")

# =============================================================================
# STAGE 2: Remove Extreme Clinical Outliers
# =============================================================================
print("\n⚠️  STAGE 2: Removing Extreme Clinical Outliers...")
print("-"*80)

df_stage2 = df_stage1.copy()

# 2.1 Blood Pressure outliers (based on AHA guidelines)
# Hypertensive crisis: >180/120
# Hypotensive shock: <70/40
# We use even more conservative ranges
mask_bp_outliers = (
    (df_stage2['ap_hi'] < 70) | (df_stage2['ap_hi'] > 250) |
    (df_stage2['ap_lo'] < 40) | (df_stage2['ap_lo'] > 150)
)
removed_bp_outliers = int(mask_bp_outliers.sum())
print(f"  - BP outliers (sys <70 or >250, dia <40 or >150): {removed_bp_outliers} rows")
cleaning_stats['removed']['bp_outliers'] = removed_bp_outliers

# 2.2 Calculate BMI and remove extreme outliers
df_stage2['bmi_temp'] = df_stage2['weight'] / ((df_stage2['height']/100) ** 2)
mask_bmi_outliers = (df_stage2['bmi_temp'] < 12) | (df_stage2['bmi_temp'] > 70)
removed_bmi_outliers = int(mask_bmi_outliers.sum())
print(f"  - BMI outliers (<12 or >70): {removed_bmi_outliers} rows")
cleaning_stats['removed']['bmi_outliers'] = removed_bmi_outliers

# 2.3 Height outliers (more refined)
mask_height_outliers = (df_stage2['height'] < 130) | (df_stage2['height'] > 220)
removed_height_outliers = int(mask_height_outliers.sum())
print(f"  - Height outliers (<130 or >220 cm): {removed_height_outliers} rows")
cleaning_stats['removed']['height_outliers'] = removed_height_outliers

# Apply all Stage 2 filters
mask_stage2 = ~mask_bp_outliers & ~mask_bmi_outliers & ~mask_height_outliers

df_stage2 = df_stage2[mask_stage2].copy()
removed_stage2 = len(df_stage1) - len(df_stage2)
print(f"\n✓ Stage 2 Complete: Removed {removed_stage2:,} rows ({removed_stage2/len(df_stage1)*100:.2f}%)")
print(f"  Remaining: {len(df_stage2):,} rows")

# =============================================================================
# STAGE 3: Feature Engineering
# =============================================================================
print("\n🔧 STAGE 3: Feature Engineering...")
print("-"*80)

df_stage3 = df_stage2.copy()

# 3.1 BMI (Body Mass Index)
df_stage3['bmi'] = df_stage3['weight'] / ((df_stage3['height']/100) ** 2)
print(f"  ✓ Created BMI feature")
print(f"    Range: {df_stage3['bmi'].min():.2f} - {df_stage3['bmi'].max():.2f}")

# 3.2 Pulse Pressure (Systolic - Diastolic)
# Indicator of arterial stiffness
df_stage3['pulse_pressure'] = df_stage3['ap_hi'] - df_stage3['ap_lo']
print(f"  ✓ Created Pulse Pressure feature")
print(f"    Range: {df_stage3['pulse_pressure'].min():.0f} - {df_stage3['pulse_pressure'].max():.0f} mmHg")

# 3.3 Mean Arterial Pressure (MAP)
# MAP = Diastolic + (Pulse Pressure / 3)
df_stage3['map'] = df_stage3['ap_lo'] + (df_stage3['pulse_pressure'] / 3)
print(f"  ✓ Created Mean Arterial Pressure (MAP) feature")
print(f"    Range: {df_stage3['map'].min():.1f} - {df_stage3['map'].max():.1f} mmHg")

# 3.4 Age in years (already created, just confirming)
print(f"  ✓ Age converted to years")
print(f"    Range: {df_stage3['age_years'].min():.1f} - {df_stage3['age_years'].max():.1f} years")

print(f"\n✓ Stage 3 Complete: Added 4 engineered features")

# =============================================================================
# STAGE 4: Encode Categorical Variables
# =============================================================================
print("\n🏷️  STAGE 4: Encoding Categorical Variables...")
print("-"*80)

df_stage4 = df_stage3.copy()

# 4.1 Gender: Convert 1,2 to 0,1 (1=female->0, 2=male->1)
df_stage4['gender'] = df_stage4['gender'] - 1
print(f"  ✓ Gender encoded: 0=female, 1=male")
print(f"    Distribution: {df_stage4['gender'].value_counts().to_dict()}")

# 4.2 One-hot encode cholesterol (1=normal, 2=above normal, 3=well above)
cholesterol_dummies = pd.get_dummies(df_stage4['cholesterol'], prefix='cholesterol')
df_stage4 = pd.concat([df_stage4, cholesterol_dummies], axis=1)
print(f"  ✓ Cholesterol one-hot encoded: cholesterol_1, cholesterol_2, cholesterol_3")
print(f"    Distribution: {df_stage4['cholesterol'].value_counts().to_dict()}")

# 4.3 One-hot encode glucose
glucose_dummies = pd.get_dummies(df_stage4['gluc'], prefix='gluc')
df_stage4 = pd.concat([df_stage4, glucose_dummies], axis=1)
print(f"  ✓ Glucose one-hot encoded: gluc_1, gluc_2, gluc_3")
print(f"    Distribution: {df_stage4['gluc'].value_counts().to_dict()}")

# Binary variables already encoded: smoke, alco, active
print(f"  ✓ Binary variables already encoded: smoke, alco, active")

print(f"\n✓ Stage 4 Complete: Categorical encoding finished")

# =============================================================================
# STAGE 5: Normalization
# =============================================================================
print("\n📏 STAGE 5: Normalization (MinMaxScaler)...")
print("-"*80)

df_stage5 = df_stage4.copy()

# Features to normalize (continuous variables)
features_to_normalize = [
    'age_years', 'height', 'weight', 'ap_hi', 'ap_lo',
    'bmi', 'pulse_pressure', 'map'
]

# Features to keep as-is (binary/categorical)
features_to_keep = [
    'gender', 'smoke', 'alco', 'active', 'cardio',
    'cholesterol_1', 'cholesterol_2', 'cholesterol_3',
    'gluc_1', 'gluc_2', 'gluc_3'
]

# Apply MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df_stage5[features_to_normalize] = scaler.fit_transform(df_stage5[features_to_normalize])

# Save denormalization parameters for future reference
denorm_params = {}
for i, feature in enumerate(features_to_normalize):
    denorm_params[feature] = {
        'min': float(scaler.data_min_[i]),
        'max': float(scaler.data_max_[i]),
        'formula': f'{feature}_original = {feature}_normalized * ({scaler.data_max_[i]} - {scaler.data_min_[i]}) + {scaler.data_min_[i]}'
    }

print(f"  ✓ Normalized {len(features_to_normalize)} continuous features to [0, 1]")
print(f"    Features: {features_to_normalize}")
print(f"  ✓ Kept {len(features_to_keep)} binary/categorical features as-is")
print(f"  ✓ Saved denormalization parameters for reversing normalization")

# Add ID column back for traceability
df_stage5['id'] = original_ids

# Create final cleaned dataset with selected columns (ID first, then features)
final_columns = ['id'] + features_to_normalize + features_to_keep
df_cleaned = df_stage5[final_columns].copy()

print(f"\n✓ Stage 5 Complete: All features normalized")
print(f"  Final dataset shape: {df_cleaned.shape}")
print(f"  ID column preserved for traceability")

# =============================================================================
# STAGE 6: Train-Test Split (80-20)
# =============================================================================
print("\n✂️  STAGE 6: Train-Test Split (80-20)...")
print("-"*80)

# Separate features and target (keep ID with features for now)
X = df_cleaned.drop('cardio', axis=1)
y = df_cleaned['cardio']

# Check class balance before split
print(f"  Class distribution before split:")
print(f"    Class 0: {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.2f}%)")
print(f"    Class 1: {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.2f}%)")

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# Recombine features and target
df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

print(f"\n  ✓ Train set: {len(df_train):,} rows ({len(df_train)/len(df_cleaned)*100:.1f}%)")
print(f"    Class 0: {(y_train==0).sum():,} ({(y_train==0).sum()/len(y_train)*100:.2f}%)")
print(f"    Class 1: {(y_train==1).sum():,} ({(y_train==1).sum()/len(y_train)*100:.2f}%)")

print(f"\n  ✓ Test set: {len(df_test):,} rows ({len(df_test)/len(df_cleaned)*100:.1f}%)")
print(f"    Class 0: {(y_test==0).sum():,} ({(y_test==0).sum()/len(y_test)*100:.2f}%)")
print(f"    Class 1: {(y_test==1).sum():,} ({(y_test==1).sum()/len(y_test)*100:.2f}%)")

print(f"\n✓ Stage 6 Complete: Stratified split maintains class balance")

# =============================================================================
# STAGE 7: Save Output Files
# =============================================================================
print("\n💾 STAGE 7: Saving Output Files...")
print("-"*80)

# Save fully cleaned dataset
output_dir = '../../data/processed/caroline_cleaned version/'
df_cleaned.to_csv(output_dir + 'cardio_fully_cleaned.csv', index=False)
print(f"  ✓ Saved: cardio_fully_cleaned.csv ({len(df_cleaned):,} rows, {len(df_cleaned.columns)} columns)")

# Save train set
df_train.to_csv(output_dir + 'cardio_train_80.csv', index=False)
print(f"  ✓ Saved: cardio_train_80.csv ({len(df_train):,} rows)")

# Save test set
df_test.to_csv(output_dir + 'cardio_test_20.csv', index=False)
print(f"  ✓ Saved: cardio_test_20.csv ({len(df_test):,} rows)")

# =============================================================================
# STAGE 8: Validation & Quality Checks
# =============================================================================
print("\n✅ STAGE 8: Validation & Quality Checks...")
print("-"*80)

validation_passed = True

# Check 1: No missing values
missing_count = df_cleaned.isnull().sum().sum()
print(f"  {'✓' if missing_count == 0 else '✗'} No missing values: {missing_count} found")
validation_passed = validation_passed and (missing_count == 0)

# Check 2: All normalized features in [0, 1]
for col in features_to_normalize:
    min_val, max_val = df_cleaned[col].min(), df_cleaned[col].max()
    is_valid = (min_val >= 0) and (max_val <= 1)
    print(f"  {'✓' if is_valid else '✗'} {col} in [0,1]: [{min_val:.4f}, {max_val:.4f}]")
    validation_passed = validation_passed and is_valid

# Check 3: Class balance maintained
balance_diff = abs((y==0).sum()/len(y) - 0.5)
is_balanced = balance_diff < 0.05  # Within 5% of perfect balance
print(f"  {'✓' if is_balanced else '✗'} Class balance maintained: {balance_diff*100:.2f}% deviation from 50-50")
validation_passed = validation_passed and is_balanced

# Check 4: Train-test split proportions
train_prop = len(df_train) / len(df_cleaned)
test_prop = len(df_test) / len(df_cleaned)
split_valid = (0.79 <= train_prop <= 0.81) and (0.19 <= test_prop <= 0.21)
print(f"  {'✓' if split_valid else '✗'} 80-20 split: {train_prop*100:.1f}%-{test_prop*100:.1f}%")
validation_passed = validation_passed and split_valid

if validation_passed:
    print(f"\n🎉 ALL VALIDATION CHECKS PASSED!")
else:
    print(f"\n⚠️  Some validation checks failed. Review output above.")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("📊 FINAL SUMMARY")
print("="*80)

total_removed = initial_rows - len(df_cleaned)
cleaning_stats['final_rows'] = len(df_cleaned)

print(f"\n📉 Data Reduction:")
print(f"  Initial rows:    {initial_rows:,}")
print(f"  Final rows:      {len(df_cleaned):,}")
print(f"  Rows removed:    {total_removed:,} ({total_removed/initial_rows*100:.2f}%)")
print(f"  Rows retained:   {len(df_cleaned)/initial_rows*100:.2f}%")

print(f"\n🗑️  Removal Breakdown:")
for reason, count in cleaning_stats['removed'].items():
    print(f"  - {reason:.<30} {count:>6,} rows")

print(f"\n📁 Output Files:")
print(f"  1. cardio_fully_cleaned.csv  - {len(df_cleaned):,} rows × {len(df_cleaned.columns)} columns")
print(f"  2. cardio_train_80.csv       - {len(df_train):,} rows (training)")
print(f"  3. cardio_test_20.csv        - {len(df_test):,} rows (testing)")

print(f"\n🎯 Class Distribution (Final):")
print(f"  Class 0 (No CVD):  {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.2f}%)")
print(f"  Class 1 (CVD):     {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.2f}%)")

print(f"\n🔧 Feature Summary:")
print(f"  Original features:     {len(df.columns) - 1}")  # -1 for target
print(f"  Engineered features:   4 (BMI, Pulse Pressure, MAP, Age in years)")
print(f"  After encoding:        {len(df_cleaned.columns) - 2}")  # -2 for target and id
print(f"  Total features:        {len(df_cleaned.columns) - 2}")  # -2 for target and id
print(f"  ID column:             Preserved for traceability")

print(f"\n✨ Pipeline Complete!")
print(f"   Ready for modeling and evaluation.")
print("="*80)

# Save cleaning statistics for the report
cleaning_stats['features'] = {
    'continuous': features_to_normalize,
    'categorical': features_to_keep
}
cleaning_stats['class_distribution'] = {
    'class_0': int((y==0).sum()),
    'class_1': int((y==1).sum())
}
cleaning_stats['denormalization'] = denorm_params

# Export stats for report generation
import json
with open(output_dir + 'cleaning_stats.json', 'w') as f:
    json.dump(cleaning_stats, f, indent=2)
print("\n📊 Cleaning statistics saved to: cleaning_stats.json")

# Export denormalization parameters separately
with open(output_dir + 'denormalization_params.json', 'w') as f:
    json.dump(denorm_params, f, indent=2)
print("📊 Denormalization parameters saved to: denormalization_params.json")
