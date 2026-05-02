# data.py

import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, raw_data_dir, train_path='data/train_data.parquet', test_path='data/test_data.parquet'):
        self.raw_dir = raw_data_dir
        self.train_path = train_path
        self.test_path = test_path

    def clean_data(self, df, balance=True):
        df.columns = df.columns.str.strip()
        
        # Extract labels early
        if 'Label' in df.columns:
            df['Label'] = df['Label'].astype(str).str.upper().apply(lambda x: 0 if 'BENIGN' in x else 1)
        
        # Drop non-numeric and metadata
        drop_cols = ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 
                    'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', 
                    'Simulated Label', 'Inbound']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
        df = df.select_dtypes(include=[np.number])

        # Replace infs and drop ANY row with a NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=0, how='any', inplace=True) # Ensure rows are dropped

        if not balance:
            return df

        # Balance classes
        if 'Label' in df.columns:
            benign = df[df['Label'] == 0]
            attack = df[df['Label'] == 1]
            n_samples = min(len(benign), len(attack))
            
            if n_samples > 0:
                df = pd.concat([benign.sample(n_samples), attack.sample(n_samples)])
                return df.sample(frac=1).reset_index(drop=True)
        return pd.DataFrame()

    def get_data(self, max_samples_per_class=500000, test_size=0.2):
        import json
        self.raw_counts = {'Benign': 0, 'Attack': 0}
        counts_path = "data/counts.json"
        
        # Check cache
        if os.path.exists(self.train_path) and os.path.exists(self.test_path):
            print(f"Loading cached data from {self.train_path} and {self.test_path}...")
            if os.path.exists(counts_path):
                with open(counts_path, "r") as f:
                    self.raw_counts = json.load(f)
            return pd.read_parquet(self.train_path), pd.read_parquet(self.test_path)

        print("No cache found. Processing raw files...")
        combined_benign = []
        combined_attack = []
        b_len = 0
        a_len = 0
        
        for root, _, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith(".csv") and not file.startswith(".~"):
                    path = os.path.join(root, file)
                    print(f"Processing: {file}")
                    try:
                        # Chunking to prevent memory errors
                        chunks = pd.read_csv(path, low_memory=False, on_bad_lines='skip', chunksize=50000)

                        for chunk in chunks:
                            cleaned_chunk = self.clean_data(chunk, balance=False)
                            if not cleaned_chunk.empty and 'Label' in cleaned_chunk.columns:
                                benign = cleaned_chunk[cleaned_chunk['Label'] == 0]
                                attack = cleaned_chunk[cleaned_chunk['Label'] == 1]
                                
                                self.raw_counts['Benign'] += len(benign)
                                self.raw_counts['Attack'] += len(attack)
                                
                                if not benign.empty:
                                    combined_benign.append(benign)
                                    b_len += len(benign)
                                    if b_len > max_samples_per_class * 2:
                                        temp_b = pd.concat(combined_benign, ignore_index=True)
                                        combined_benign = [temp_b.sample(n=max_samples_per_class)]
                                        b_len = max_samples_per_class
                                        
                                if not attack.empty:
                                    combined_attack.append(attack)
                                    a_len += len(attack)
                                    if a_len > max_samples_per_class * 2:
                                        temp_a = pd.concat(combined_attack, ignore_index=True)
                                        combined_attack = [temp_a.sample(n=max_samples_per_class)]
                                        a_len = max_samples_per_class

                    except Exception as e:
                        print(f"Could not process {file}: {e}")

        if not combined_benign or not combined_attack:
            raise ValueError("No data loaded. Check raw_data_dir.")

        all_benign = pd.concat(combined_benign, ignore_index=True).drop_duplicates()
        all_attack = pd.concat(combined_attack, ignore_index=True).drop_duplicates()
        
        if len(all_benign) == 0 and len(all_attack) == 0:
            raise ValueError("No data loaded.")

        # 1. Combine all data to preserve original distribution before splitting
        full_df = pd.concat([all_benign, all_attack]).sample(frac=1, random_state=42).reset_index(drop=True)

        # 2. Split into train and test
        split_idx = int(len(full_df) * (1 - test_size))
        train_df = full_df.iloc[:split_idx]
        test_df = full_df.iloc[split_idx:]

        # 3. Apply transformations (max_samples_per_class limiting) to train_df ONLY
        train_benign = train_df[train_df['Label'] == 0]
        train_attack = train_df[train_df['Label'] == 1]
        
        n_benign = min(len(train_benign), max_samples_per_class)
        n_attack = min(len(train_attack), max_samples_per_class)
        
        final_train_benign = train_benign.sample(n_benign, random_state=42) if n_benign > 0 else train_benign
        final_train_attack = train_attack.sample(n_attack, random_state=42) if n_attack > 0 else train_attack
        
        train_df = pd.concat([final_train_benign, final_train_attack]).sample(frac=1, random_state=42).reset_index(drop=True)

        os.makedirs(os.path.dirname(self.train_path), exist_ok=True)
        train_df.to_parquet(self.train_path)
        test_df.to_parquet(self.test_path)
        
        with open(counts_path, "w") as f:
            json.dump(self.raw_counts, f)
            
        return train_df, test_df