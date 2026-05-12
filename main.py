# main.py
import os
import time

def main():
    print("--- CICDDoS2019 Detection Pipeline ---")
    
    print("Step 1: Initializing Data Loader...", end="", flush=True)
    from data_loader import DataLoader
    print(" Done.")
    
    print("Step 2: Loading model components...", end="", flush=True)
    from custom_model import CustomLogisticRegression, ManualScaler
    from evaluator import Evaluator
    import numpy as np
    print(" Done.")

    # 1. Load Data (Checks for cache automatically)
    loader = DataLoader(raw_data_dir='data/')
    print("\n[Data Acquisition]")
    full_df = loader.get_data()

    # 2. Extract Features and Labels
    # Explicitly drop helper columns from features
    X = full_df.drop(['Label', '_source_file'], axis=1, errors='ignore').values
    y = full_df['Label'].values
    
    # Clean any remaining NaNs or Infs
    X = np.nan_to_num(np.array(X, dtype=float))

    from sklearn.model_selection import StratifiedKFold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_test_metrics = []
    
    print(f"\n[Starting {n_splits}-Fold Cross Validation]")
    
    # We will use the last fold's results for the interactive dashboard
    last_fold_data = {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        before_gan_counts = {'Benign': int(np.sum(y_train == 0)), 'Attack': int(np.sum(y_train == 1))}
        test_counts = {'Benign': int(np.sum(y_test == 0)), 'Attack': int(np.sum(y_test == 1))}

        # Apply GAN Balancing to Training Set
        print("  Balancing training fold with GAN...")
        from gan_balancer import GANBalancer
        balancer = GANBalancer(epochs=50)
        X_train_resampled, y_train_resampled = balancer.balance(X_train, y_train)
        
        # Scaling
        scaler = ManualScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model = CustomLogisticRegression(lr=0.1, epochs=50, batch_size=8192)
        model.fit(X_train_scaled, y_train_resampled)
        
        # Evaluate
        eval_tool = Evaluator()
        
        train_preds = model.predict(X_train_scaled)
        train_metrics, train_conf_data = eval_tool.calculate_metrics(y_train_resampled, train_preds)
        
        test_preds = model.predict(X_test_scaled)
        test_metrics, test_conf_data = eval_tool.calculate_metrics(y_test, test_preds)
        
        all_test_metrics.append(test_metrics)
        print(f"  Fold {fold + 1} Test Accuracy: {test_metrics['Accuracy']:.4f}")

        # Store last fold for dashboard
        current_raw_counts = loader.raw_counts.copy()
        current_raw_counts['Train Samples Per File'] = full_df.iloc[train_idx]['_source_file'].value_counts().to_dict()
        current_raw_counts['Test Samples Per File'] = full_df.iloc[test_idx]['_source_file'].value_counts().to_dict()

        last_fold_data = {
            'raw_counts': current_raw_counts,
            'before_gan_counts': before_gan_counts,
            'test_counts': test_counts,
            'balanced_counts': {'Benign': int(np.sum(y_train_resampled == 0)), 'Attack': int(np.sum(y_train_resampled == 1))},
            'train_metrics': train_metrics,
            'train_conf_data': train_conf_data,
            'test_metrics': test_metrics,
            'test_conf_data': test_conf_data,
            'df': full_df.iloc[test_idx] # Show test set correlation
        }

    # Summarize results
    print("\n" + "="*30)
    print(f" {n_splits}-FOLD CV RESULTS (AVERAGE)")
    print("="*30)
    for metric in all_test_metrics[0].keys():
        values = [m[metric] for m in all_test_metrics]
        print(f"{metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
    print("="*30)

    # Launch dashboard using results from the last fold
    print("\nLaunching interactive chart viewer (showing final fold)...")
    eval_tool.plot_interactive_view(
        last_fold_data['raw_counts'], 
        last_fold_data['before_gan_counts'], 
        last_fold_data['balanced_counts'], 
        last_fold_data['test_counts'],
        last_fold_data['train_metrics'], 
        last_fold_data['train_conf_data'], 
        last_fold_data['test_metrics'], 
        last_fold_data['test_conf_data'], 
        last_fold_data['df']
    )

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal execution time: {time.time() - start_time:.2f}s")