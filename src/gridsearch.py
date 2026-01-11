import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================
# DATA PREPROCESSING
# ========================================
print("Loading and preprocessing data...")
df = pd.read_csv("data/raw/KaggleV2-May-2016.csv")

# Target
df['no_show'] = df['No-show'].map({'No': 0, 'Yes': 1})

# Dates
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# Days between scheduling and appointment
df['days_between'] = (
    df['AppointmentDay'].dt.normalize()
    - df['ScheduledDay'].dt.normalize()
).dt.days

# Remove rows with negative days_between
df = df[df['days_between'] >= 0]

df['appointment_weekday'] = df['AppointmentDay'].dt.weekday
df['is_weekend'] = df['appointment_weekday'].isin([5, 6]).astype(int)

# Demographics
df['gender'] = df['Gender'].map({'F': 0, 'M': 1})
df.loc[df['Age'] < 0, 'Age'] = np.nan

# Neighborhood frequency encoding
neighborhood_freq = df['Neighbourhood'].value_counts(normalize=True)
df['neighborhood_freq'] = df['Neighbourhood'].map(neighborhood_freq)

feature_cols = [
    'Age', 'gender', 'days_between', 'appointment_weekday', 'is_weekend',
    'SMS_received', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap',
    'neighborhood_freq'
]

#remove rows with missing feature values
df = df.dropna(subset=feature_cols)

X = df[feature_cols]
y = df['no_show']

print(f"Dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts(normalize=True)}")

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTrain size: {X_train.shape[0]}")
print(f"Validation size: {X_val.shape[0]}")

# ========================================
# MODEL DEFINITIONS
# ========================================
# Calculate scale_pos_weight for XGBoost (ratio of negative to positive samples)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\nClass imbalance ratio: {scale_pos_weight:.2f}")

models = {
    'Logistic Regression': {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'params': {
            'lr__C': np.logspace(-3, 2, 10),
            'lr__class_weight': ['balanced', None],
            'lr__penalty': ['l2'],
            'lr__solver': ['lbfgs', 'liblinear', 'saga']
        }
    },
    'Random Forest': {
        'model': Pipeline([
            ('rf', RandomForestClassifier(random_state=42))
        ]),
        'params': {
            'rf__n_estimators': [200, 300],
            'rf__max_depth': [10, 15, 20, None],
            'rf__min_samples_split': [5, 10, 15],
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__class_weight': ['balanced'],
            'rf__max_features': ['sqrt', 'log2']
        }
    },
    'XGBoost': {
        'model': Pipeline([
            ('xgb', XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                tree_method='hist',
                use_label_encoder=False
            ))
        ]),
        'params': {
            'xgb__n_estimators': [200, 300],
            'xgb__max_depth': [3, 5, 7],
            'xgb__learning_rate': [0.01, 0.05],
            'xgb__subsample': [0.8, 1.0],
            'xgb__colsample_bytree': [0.8, 1.0],
            'xgb__scale_pos_weight': [scale_pos_weight]
        }
    }
}

# ========================================
# TRAIN AND EVALUATE MODELS
# ========================================
results = {}

for model_name, model_info in models.items():
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    # GridSearchCV
    search = GridSearchCV(
        estimator=model_info['model'],
        param_grid=model_info['params'],
        cv=3,
        scoring='roc_auc', # Use ROC-AUC for evaluation due to class imbalance
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    
    # Fit on full training data
    print(f"\nStarting GridSearchCV for {model_name}...")
    search.fit(X_train, y_train)
    
    # Get best model
    best_model = search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    f1 = f1_score(y_val, y_pred)
    
    # Store results
    results[model_name] = {
        'best_params': search.best_params_,
        'best_cv_score': search.best_score_,
        'val_roc_auc': roc_auc,
        'val_f1': f1,
        'best_model': best_model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'cv_results': search.cv_results_
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {model_name}")
    print(f"{'='*60}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV ROC-AUC score: {search.best_score_:.4f}")
    print(f"Validation ROC-AUC: {roc_auc:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Show', 'No-Show']))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

# ========================================
# FINAL COMPARISON
# ========================================
print(f"\n{'='*60}")
print("FINAL MODEL COMPARISON")
print(f"{'='*60}")

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'CV ROC-AUC': [results[m]['best_cv_score'] for m in results.keys()],
    'Val ROC-AUC': [results[m]['val_roc_auc'] for m in results.keys()],
    'Val F1': [results[m]['val_f1'] for m in results.keys()]
})

print("\n", comparison_df.to_string(index=False))

# Get best model
best_model_name = max(results, key=lambda x: results[x]['val_roc_auc'])
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Validation ROC-AUC: {results[best_model_name]['val_roc_auc']:.4f}")
print(f"   Best Parameters: {results[best_model_name]['best_params']}")

# ========================================
# VISUALIZATION
# ========================================
print("\nGenerating visualizations...")

# 1. ROC Curves
plt.figure(figsize=(10, 6))
for model_name in results.keys():
    fpr, tpr, _ = roc_curve(y_val, results[model_name]['y_pred_proba'])
    auc = results[model_name]['val_roc_auc']
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Precision-Recall Curves
plt.figure(figsize=(10, 6))
for model_name in results.keys():
    precision, recall, _ = precision_recall_curve(y_val, results[model_name]['y_pred_proba'])
    plt.plot(recall, precision, label=model_name, linewidth=2)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/precision_recall_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Confusion Matrices
fig, axes = plt.subplots(1, len(results), figsize=(15, 4))
if len(results) == 1:
    axes = [axes]

for idx, (model_name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_val, result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                cbar=False, square=True)
    axes[idx].set_title(f'{model_name}\nROC-AUC: {result["val_roc_auc"]:.4f}', 
                       fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_xticklabels(['Show', 'No-Show'])
    axes[idx].set_yticklabels(['Show', 'No-Show'])

plt.tight_layout()
plt.savefig('figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Feature Importance Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest Feature Importance
if 'Random Forest' in results:
    rf_model = results['Random Forest']['best_model']
    rf_estimator = rf_model.named_steps['rf']
    
    rf_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_estimator.feature_importances_
    }).sort_values('importance', ascending=False)
    
    axes[0].barh(rf_importance['feature'], rf_importance['importance'], color='steelblue')
    axes[0].set_xlabel('Importance', fontsize=12)
    axes[0].set_ylabel('Feature', fontsize=12)
    axes[0].set_title('Random Forest - Feature Importance', fontsize=13, fontweight='bold')
    axes[0].invert_yaxis()

# XGBoost Feature Importance
if 'XGBoost' in results:
    xgb_model = results['XGBoost']['best_model']
    xgb_estimator = xgb_model.named_steps['xgb']
    
    xgb_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': xgb_estimator.feature_importances_
    }).sort_values('importance', ascending=False)
    
    axes[1].barh(xgb_importance['feature'], xgb_importance['importance'], color='darkorange')
    axes[1].set_xlabel('Importance', fontsize=12)
    axes[1].set_ylabel('Feature', fontsize=12)
    axes[1].set_title('XGBoost - Feature Importance', fontsize=13, fontweight='bold')
    axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('figures/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Print feature importance tables
if 'Random Forest' in results:
    print("\n" + "="*60)
    print("Random Forest - Top 5 Most Important Features:")
    print("="*60)
    print(rf_importance.head().to_string(index=False))

if 'XGBoost' in results:
    print("\n" + "="*60)
    print("XGBoost - Top 5 Most Important Features:")
    print("="*60)
    print(xgb_importance.head().to_string(index=False))

# 5. Model Performance Comparison Bar Chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

metrics_df = comparison_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

# CV ROC-AUC
cv_data = metrics_df[metrics_df['Metric'] == 'CV ROC-AUC']
axes[0].bar(cv_data['Model'], cv_data['Score'], color=['steelblue', 'seagreen', 'darkorange'])
axes[0].set_ylabel('ROC-AUC Score', fontsize=12)
axes[0].set_title('Cross-Validation ROC-AUC', fontsize=13, fontweight='bold')
axes[0].set_ylim([0.5, 1.0])
axes[0].tick_params(axis='x', rotation=15)
axes[0].grid(axis='y', alpha=0.3)

# Validation ROC-AUC
val_data = metrics_df[metrics_df['Metric'] == 'Val ROC-AUC']
axes[1].bar(val_data['Model'], val_data['Score'], color=['steelblue', 'seagreen', 'darkorange'])
axes[1].set_ylabel('ROC-AUC Score', fontsize=12)
axes[1].set_title('Validation ROC-AUC', fontsize=13, fontweight='bold')
axes[1].set_ylim([0.5, 1.0])
axes[1].tick_params(axis='x', rotation=15)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('figures/model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("Training and evaluation complete!")
print("Visualizations saved as PNG files:")
print("  - roc_curves_comparison.png")
print("  - precision_recall_comparison.png")
print("  - confusion_matrices.png")
print("  - feature_importance_comparison.png")
print("  - model_performance_comparison.png")
print("="*60)

# ========================================
# SAVE BEST MODEL
# ========================================
import joblib

best_model_obj = results[best_model_name]['best_model']
joblib.dump(best_model_obj, f'models/best_model_{best_model_name.replace(" ", "_").lower()}.pkl')
print(f"\n✅ Best model saved as: models/best_model_{best_model_name.replace(' ', '_').lower()}.pkl")