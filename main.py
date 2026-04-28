# main.py

from data_loader import DataLoader
from custom_model import CustomLogisticRegression, ManualScaler
from evaluator import Evaluator
import numpy as np

# 1. Load Data (Checks for cache automatically)
loader = DataLoader(raw_data_dir='data/')
df = loader.get_data(max_samples_per_class=500000)

# 2. Manual Split
df = df.sample(frac=1).reset_index(drop=True)
split = int(len(df) * 0.8)
X = df.drop('Label', axis=1).values
y = df['Label'].values

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. Scaling
scaler = ManualScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train Shape: {X_train_scaled.shape}")
print(f"Labels - Unique values: {np.unique(y_train)}")
print(f"Any NaNs in X: {np.isnan(X_train_scaled).any()}")
print(f"Any Infs in X: {np.isinf(X_train_scaled).any()}")

# 4. Train
model = CustomLogisticRegression(lr=0.1, epochs=1000)
model.fit(X_train_scaled, y_train)

# 5. Evaluate & Plot
eval_tool = Evaluator()
preds = model.predict(X_test_scaled)
metrics, conf_data = eval_tool.calculate_metrics(y_test, preds)

print("Results:", metrics)

# Plot class imbalance
raw_counts = loader.raw_counts
balanced_counts = {'Benign': len(df[df['Label'] == 0]), 'Attack': len(df[df['Label'] == 1])}
eval_tool.plot_class_distribution(raw_counts, balanced_counts)

eval_tool.plot_results(metrics, conf_data)