# main.py

from data_loader import DataLoader
from custom_model import CustomLogisticRegression, ManualScaler
from evaluator import Evaluator
import numpy as np
from imblearn.over_sampling import SMOTE
import os

# 1. Load Data (Checks for cache automatically)
loader = DataLoader(raw_data_dir='data/')
train_df, test_df = loader.get_data(max_samples_per_class=5000000, test_size=0.2)

# 2. Extract Features and Labels
X_train = train_df.drop('Label', axis=1).values
y_train = train_df['Label'].values
X_test = test_df.drop('Label', axis=1).values
y_test = test_df['Label'].values

# Clean any remaining NaNs or Infs before SMOTE
X_train = np.nan_to_num(np.array(X_train, dtype=float))
X_test = np.nan_to_num(np.array(X_test, dtype=float))

model_path = "data/model.npz"
scaler_path = "data/scaler.npz"

scaler = ManualScaler()
model = CustomLogisticRegression(lr=0.1, epochs=50, batch_size=8192)

if os.path.exists(model_path) and os.path.exists(scaler_path):
    print("Loading saved model and scaler...")
    scaler.load(scaler_path)
    model.load(model_path)
    X_test_scaled = scaler.transform(X_test)
    
    # Estimate balanced counts for plotting since we skipped SMOTE
    majority_class_count = max(np.sum(y_train == 0), np.sum(y_train == 1))
    balanced_counts = {'Benign': majority_class_count, 'Attack': majority_class_count}
else:
    # Apply SMOTE to Training Set
    print(f"Before SMOTE - X_train Shape: {X_train.shape}, Benign: {np.sum(y_train == 0)}, Attack: {np.sum(y_train == 1)}")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - X_train Shape: {X_train_resampled.shape}, Benign: {np.sum(y_train_resampled == 0)}, Attack: {np.sum(y_train_resampled == 1)}")

    # 3. Scaling
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"X_train_scaled Shape: {X_train_scaled.shape}")
    print(f"Labels - Unique values: {np.unique(y_train_resampled)}")
    print(f"Any NaNs in X: {np.isnan(X_train_scaled).any()}")
    print(f"Any Infs in X: {np.isinf(X_train_scaled).any()}")

    # 4. Train
    model.fit(X_train_scaled, y_train_resampled)
    
    print("Saving model and scaler...")
    scaler.save(scaler_path)
    model.save(model_path)
    
    balanced_counts = {'Benign': int(np.sum(y_train_resampled == 0)), 'Attack': int(np.sum(y_train_resampled == 1))}

# 5. Evaluate & Plot
eval_tool = Evaluator()
preds = model.predict(X_test_scaled)
metrics, conf_data = eval_tool.calculate_metrics(y_test, preds)

print("Results:", metrics)

# Plot interactive figures in one window
raw_counts = loader.raw_counts
print("Launching interactive chart viewer...")
eval_tool.plot_interactive_view(raw_counts, balanced_counts, metrics, conf_data, train_df)