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
    train_df, test_df = loader.get_data(max_samples_per_class=5000000, test_size=0.2)

    # 2. Extract Features and Labels
    X_train = train_df.drop('Label', axis=1).values
    y_train = train_df['Label'].values
    X_test = test_df.drop('Label', axis=1).values
    y_test = test_df['Label'].values

    before_gan_counts = {'Benign': int(np.sum(y_train == 0)), 'Attack': int(np.sum(y_train == 1))}
    test_counts = {'Benign': int(np.sum(y_test == 0)), 'Attack': int(np.sum(y_test == 1))}

    # Clean any remaining NaNs or Infs before GAN Balancing
    X_train = np.nan_to_num(np.array(X_train, dtype=float))
    X_test = np.nan_to_num(np.array(X_test, dtype=float))

    model_path = "data/model.npz"
    scaler_path = "data/scaler.npz"

    scaler = ManualScaler()
    model = CustomLogisticRegression(lr=0.1, epochs=50, batch_size=8192)

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("\n[Model Loading]")
        print("Loading saved model and scaler...")
        scaler.load(scaler_path)
        model.load(model_path)
        X_test_scaled = scaler.transform(X_test)
        
        # Skip GAN re-balancing when loading pre-trained model for speed and consistency
        print("\n[Evaluation Setup]")
        print("Using original training distribution for evaluation (Skipping GAN)...")
        X_train_resampled, y_train_resampled = X_train, y_train
        X_train_scaled = scaler.transform(X_train_resampled)
        balanced_counts = before_gan_counts.copy()
    else:
        # Apply GAN Balancing to Training Set
        print("\n[Data Balancing]")
        print("Loading Generative AI components (Torch)...", end="", flush=True)
        from gan_balancer import GANBalancer
        print(" Done.")
        print(f"Before GAN Balancing - X_train Shape: {X_train.shape}, Benign: {np.sum(y_train == 0)}, Attack: {np.sum(y_train == 1)}")
        balancer = GANBalancer(epochs=50)
        X_train_resampled, y_train_resampled = balancer.balance(X_train, y_train)
        print(f"After GAN Balancing - X_train Shape: {X_train_resampled.shape}, Benign: {np.sum(y_train_resampled == 0)}, Attack: {np.sum(y_train_resampled == 1)}")

        # 3. Scaling
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"X_train_scaled Shape: {X_train_scaled.shape}")
        
        # 4. Train
        print("\n[Model Training]")
        model.fit(X_train_scaled, y_train_resampled)
        
        print("Saving model and scaler...")
        scaler.save(scaler_path)
        model.save(model_path)
        
        balanced_counts = {'Benign': int(np.sum(y_train_resampled == 0)), 'Attack': int(np.sum(y_train_resampled == 1))}

    # 5. Evaluate & Plot
    eval_tool = Evaluator()

    # Training Evaluation
    print("\n[Evaluation]")
    print("Evaluating on training data...")
    train_preds = model.predict(X_train_scaled)
    train_metrics, train_conf_data = eval_tool.calculate_metrics(y_train_resampled, train_preds)

    # Testing Evaluation
    print("Evaluating on testing data...")
    test_preds = model.predict(X_test_scaled)
    test_metrics, test_conf_data = eval_tool.calculate_metrics(y_test, test_preds)

    print("Training Results:", train_metrics)
    print("Testing Results:", test_metrics)

    # Plot interactive figures in one window
    raw_counts = loader.raw_counts
    print("\nLaunching interactive chart viewer...")
    eval_tool.plot_interactive_view(
        raw_counts, 
        before_gan_counts, 
        balanced_counts, 
        test_counts,
        train_metrics, 
        train_conf_data, 
        test_metrics, 
        test_conf_data, 
        train_df
    )

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal execution time: {time.time() - start_time:.2f}s")