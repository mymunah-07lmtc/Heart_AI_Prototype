import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# --- NEW: Import SMOTE and the special Pipeline ---
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Load features
df = pd.read_csv("heart_sound_features.csv")
print("📊 Dataset loaded")
print(f"   Total samples: {len(df)}")
print(f"   Features: {df.shape[1] - 3}")

# 2. Prepare X and y
X = df[[f'MFCC_{i+1}' for i in range(13)] + 
       [f'MFCC_std_{i+1}' for i in range(13)] + 
       ['spec_cent_mean', 'spec_cent_std', 'spec_bw_mean', 'spec_bw_std', 
        'spec_rolloff_mean', 'spec_rolloff_std', 'zcr_mean', 'zcr_std', 
        'rms_mean', 'rms_std']]

y = df['Binary_Label']

print(f"\n📊 Class distribution BEFORE SMOTE:")
print(f"   Normal (0): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
print(f"   Abnormal (1): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")

# 3. Model definitions (we can still keep class weights, but SMOTE will do the heavy lifting)
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200, 
        class_weight='balanced',  # Extra safety for imbalance
        random_state=42
    ),
    'SVM_RBF': SVC(
        kernel='rbf', 
        C=10, 
        gamma='scale',
        class_weight='balanced', 
        probability=True, 
        random_state=42
    )
}

# Try XGBoost
try:
    from xgboost import XGBClassifier
    models['XGBoost'] = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        scale_pos_weight=sum(y==0)/sum(y==1),  # Still helpful
        random_state=42
    )
    print("✅ XGBoost loaded.")
except ImportError:
    print("ℹ️ XGBoost not installed. Skipping.")

# 4. Cross-validation WITH SMOTE
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n🔬 Evaluating models with 5-Fold Cross-Validation (WITH SMOTE):")
print("-" * 60)

best_f1 = -1
best_pipeline = None
best_model_name = None
best_scaler = None
best_model_obj = None

for name, model in models.items():
    # --- THE MAGIC: SMOTE -> Scaler -> Classifier ---
    # Note: We use ImbPipeline (not sklearn Pipeline) so SMOTE works inside CV folds
    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),          # Synthetic data generation
        ('scaler', StandardScaler()),               # Normalize features
        ('clf', model)                              # Classifier
    ])
    
    # Use cross_validate to get multiple metrics
    scores = cross_validate(
        pipe, X, y, 
        cv=cv, 
        scoring=['accuracy', 'f1_macro', 'recall_macro', 'precision_macro'],
        n_jobs=-1
    )
    
    mean_acc = scores['test_accuracy'].mean()
    mean_f1 = scores['test_f1_macro'].mean()
    mean_recall = scores['test_recall_macro'].mean()
    mean_precision = scores['test_precision_macro'].mean()
    
    print(f"\n📊 {name}:")
    print(f"   Accuracy:  {mean_acc:.4f} (+/- {scores['test_accuracy'].std():.4f})")
    print(f"   F1-Macro:  {mean_f1:.4f} (+/- {scores['test_f1_macro'].std():.4f})")
    print(f"   Recall:    {mean_recall:.4f}")
    print(f"   Precision: {mean_precision:.4f}")
    
    if mean_f1 > best_f1:
        best_f1 = mean_f1
        best_model_name = name
        best_model_obj = model  # Store the base model
        
        # --- Train the FINAL model on the ENTIRE dataset using SMOTE ---
        # We do this to save the best model for deployment
        scaler = StandardScaler()
        smote = SMOTE(random_state=42)
        
        # Apply SMOTE to the full dataset
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Scale the resampled data
        X_scaled = scaler.fit_transform(X_resampled)
        
        # Train the model on the balanced dataset
        model.fit(X_scaled, y_resampled)
        
        # --- Quick evaluation on original test set (using the resampled model) ---
        # Note: We scale the original X with the same scaler
        X_orig_scaled = scaler.transform(X)
        y_pred = model.predict(X_orig_scaled)
        
        print(f"\n   📊 Final Model Performance (on original data):")
        print(f"   Accuracy:  {accuracy_score(y, y_pred):.4f}")
        print(f"   F1-Score:  {f1_score(y, y_pred):.4f}")
        print(f"   Recall:    {recall_score(y, y_pred):.4f}")
        print(f"   Precision: {precision_score(y, y_pred):.4f}")
        print(f"\n   Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

print("\n" + "=" * 60)
print(f"🏆 Best Model: {best_model_name} (F1-Macro: {best_f1:.4f})")
print("=" * 60)

# 5. Save the best model components
if best_model_obj is not None:
    # We already fitted the best model in the loop above.
    # But we need to save the scaler used for the best model.
    # Since we overwrite scaler each loop, let's re-run the best one specifically.
    
    # Re-run the best model one more time to be safe
    print(f"\n🔄 Re-training the best model ({best_model_name}) for deployment...")
    final_scaler = StandardScaler()
    final_smote = SMOTE(random_state=42)
    
    X_res, y_res = final_smote.fit_resample(X, y)
    X_scaled_final = final_scaler.fit_transform(X_res)
    
    # Re-initialize the model to avoid any data leakage from the loop
    if best_model_name == 'RandomForest':
        final_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    elif best_model_name == 'SVM_RBF':
        final_model = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True, random_state=42)
    elif best_model_name == 'XGBoost':
        final_model = XGBClassifier(n_estimators=100, learning_rate=0.1, scale_pos_weight=sum(y==0)/sum(y==1), random_state=42)
    else:
        final_model = best_model_obj  # fallback
    
    final_model.fit(X_scaled_final, y_res)
    
    # Save
    joblib.dump(final_model, "heart_model_best.pkl")
    joblib.dump(final_scaler, "scaler.pkl")
    label_map = {0: "Normal (Healthy)", 1: "Abnormal (Murmur Detected)"}
    joblib.dump(label_map, "label_map.pkl")

    print("\n✅ Models saved successfully!")
    print(f"   - heart_model_best.pkl ({best_model_name} with SMOTE)")
    print("   - scaler.pkl")
    print("   - label_map.pkl")

else:
    print("❌ No model selected. Please check the errors above.")

print("\n🚀 Next step: Update app.py for heart sounds")