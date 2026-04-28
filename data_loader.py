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

    def clean_data(self, df):
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

        # Balance classes
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
                        chunks = pd.read_csv(path, low_memory=False, on_bad_lines='skip', chunksize=10000)
                        first_chunk = next(chunks)
                        
                        actual_sample = min(len(first_chunk), sample_size_per_file)
                        temp_df = first_chunk.sample(n=actual_sample)
                        
                        # Apply the cleaning logic
                        cleaned = self.clean_data(temp_df)
                        combined_df.append(cleaned)
                    except Exception as e:
                        print(f"Could not process {file}: {e}")

        if not combined_df:
            raise ValueError("No data loaded. Check raw_data_dir.")

        full_df = pd.concat(combined_df, ignore_index=True)
        os.makedirs(os.path.dirname(self.processed_path), exist_ok=True)
        full_df.to_parquet(self.processed_path)
        return full_df