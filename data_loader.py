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

    def get_data(self, sample_size_per_file=5000):
        # Check cache
        if os.path.exists(self.processed_path):
            print(f"Loading cached data from {self.processed_path}...")
            return pd.read_parquet(self.processed_path)

        print("No cache found. Processing raw files...")
        combined_df = []
        
        for root, _, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith(".csv") and not file.startswith(".~"):
                    path = os.path.join(root, file)
                    print(f"Processing: {file}")
                    try:
                        # Chunking to prevent memory errors
                        chunks = pd.read_csv(path, low_memory=False, on_bad_lines='skip', chunksize=50000)
                        
                        file_benign = []
                        file_attack = []
                        b_len = 0
                        a_len = 0

                        for chunk in chunks:
                            cleaned_chunk = self.clean_data(chunk, balance=False)
                            if not cleaned_chunk.empty and 'Label' in cleaned_chunk.columns:
                                benign = cleaned_chunk[cleaned_chunk['Label'] == 0]
                                attack = cleaned_chunk[cleaned_chunk['Label'] == 1]
                                
                                if not benign.empty:
                                    file_benign.append(benign)
                                    b_len += len(benign)
                                    if b_len > 200000:
                                        temp_b = pd.concat(file_benign, ignore_index=True)
                                        file_benign = [temp_b.sample(n=100000)]
                                        b_len = 100000
                                        
                                if not attack.empty:
                                    file_attack.append(attack)
                                    a_len += len(attack)
                                    if a_len > 200000:
                                        temp_a = pd.concat(file_attack, ignore_index=True)
                                        file_attack = [temp_a.sample(n=100000)]
                                        a_len = 100000

                        if file_benign and file_attack:
                            all_benign = pd.concat(file_benign, ignore_index=True)
                            all_attack = pd.concat(file_attack, ignore_index=True)
                            
                            n_samples = min(len(all_benign), len(all_attack), sample_size_per_file // 2)
                            if n_samples > 0:
                                balanced_df = pd.concat([all_benign.sample(n_samples), all_attack.sample(n_samples)])
                                combined_df.append(balanced_df.sample(frac=1).reset_index(drop=True))
                    except Exception as e:
                        print(f"Could not process {file}: {e}")

        if not combined_df:
            raise ValueError("No data loaded. Check raw_data_dir.")

        full_df = pd.concat(combined_df, ignore_index=True)
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        full_df.to_parquet(self.processed_path)
        return full_df