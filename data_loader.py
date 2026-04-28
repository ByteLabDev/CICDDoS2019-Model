# data.py

import pandas as pd
import numpy as np
import os

import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, raw_data_dir, processed_path='data/processed_data.parquet'):
        self.raw_dir = raw_data_dir
        self.processed_path = processed_path

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

    def get_data(self, max_samples_per_class=500000):
        import json
        self.raw_counts = {'Benign': 0, 'Attack': 0}
        counts_path = self.processed_path + ".counts.json"
        
        # Check cache
        if os.path.exists(self.processed_path):
            print(f"Loading cached data from {self.processed_path}...")
            if os.path.exists(counts_path):
                with open(counts_path, "r") as f:
                    self.raw_counts = json.load(f)
            return pd.read_parquet(self.processed_path)

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
        
        n_samples = min(len(all_benign), len(all_attack), max_samples_per_class)
        if n_samples > 0:
            balanced_df = pd.concat([all_benign.sample(n_samples), all_attack.sample(n_samples)])
            full_df = balanced_df.sample(frac=1).reset_index(drop=True)
        else:
            raise ValueError("Not enough data to balance.")

        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        full_df.to_parquet(self.processed_path)
        
        with open(counts_path, "w") as f:
            json.dump(self.raw_counts, f)
            
        return full_df