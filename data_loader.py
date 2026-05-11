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
        initial_count = len(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=0, how='any', inplace=True) # Ensure rows are dropped
        dropped_count = initial_count - len(df)

        if not balance:
            return df, dropped_count

        # Balance classes
        if 'Label' in df.columns:
            benign = df[df['Label'] == 0]
            attack = df[df['Label'] == 1]
            n_samples = min(len(benign), len(attack))
            
            if n_samples > 0:
                df = pd.concat([benign.sample(n_samples), attack.sample(n_samples)])
                return df.sample(frac=1).reset_index(drop=True), dropped_count
        return pd.DataFrame(), dropped_count

    def get_data(self, max_samples_per_class=500000, test_size=0.2):
        import json
        self.raw_counts = {'Benign': 0, 'Attack': 0, 'Total Dropped': 0, 'Total Saved': 0, 'Entries Per File': {}, 'Train Samples Per File': {}, 'Test Samples Per File': {}}
        counts_path = "data/counts.json"
        
        # Check cache
        if os.path.exists(self.train_path) and os.path.exists(self.test_path):
            print(f"Loading cached data from {self.train_path} and {self.test_path}...")
            if os.path.exists(counts_path):
                with open(counts_path, "r") as f:
                    self.raw_counts = json.load(f)
            return pd.read_parquet(self.train_path), pd.read_parquet(self.test_path)

        print("No cache found. Processing raw files...")
        
        # 1. Get all CSV files
        all_files = []
        for root, _, files in os.walk(self.raw_dir):
            for file in files:
                if file.endswith(".csv") and not file.startswith(".~"):
                    all_files.append(os.path.join(root, file))
        
        if not all_files:
            raise ValueError(f"No CSV files found in {self.raw_dir}")

        # 2. Step 1: Process CSVs and save to temp parquets
        temp_dir = "data/temp_uniform"
        os.makedirs(temp_dir, exist_ok=True)
        file_info = {} # {temp_parquet_path: count}
        
        print("Step 1: Checking for cached Parquet files and cleaning new datasets...")
        self.feature_cols = None # To store the master schema
        
        for path in all_files:
            filename = os.path.basename(path)
            rel_path = os.path.relpath(path, self.raw_dir)
            unique_name = rel_path.replace(os.sep, "_").replace(".csv", ".parquet")
            temp_path = os.path.join(temp_dir, unique_name)
            
            # Skip if temp parquet already exists (in case of resume)
            if os.path.exists(temp_path):
                try:
                    import pyarrow.parquet as pq
                    meta = pq.read_metadata(temp_path)
                    file_info[temp_path] = meta.num_rows
                    # Capture schema if not already set
                    if self.feature_cols is None:
                        self.feature_cols = meta.schema.names
                    
                    # Track entries per file
                    self.raw_counts['Entries Per File'][unique_name] = meta.num_rows
                    
                    print(f"  Found cached temp file: {filename} ({meta.num_rows:,} samples)")
                    continue
                except:
                    pass # Corrupted file, re-process

            print(f"  Processing: {filename}...", flush=True)
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                writer = None
                row_count = 0
                chunks = pd.read_csv(path, low_memory=False, on_bad_lines='skip', chunksize=200000)
                
                total_dropped = 0
                for chunk in chunks:
                    cleaned, dropped = self.clean_data(chunk, balance=False)
                    total_dropped += dropped
                    if not cleaned.empty:
                        # Enforce schema consistency
                        if self.feature_cols is None:
                            self.feature_cols = list(cleaned.columns)
                        else:
                            # Reindex to match the first chunk's schema (adds missing cols as NaN, drops extra)
                            cleaned = cleaned.reindex(columns=self.feature_cols, fill_value=0)
                        
                        table = pa.Table.from_pandas(cleaned, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(temp_path, table.schema)
                        writer.write_table(table)
                        row_count += len(cleaned)
                
                if writer:
                    writer.close()
                    file_info[temp_path] = row_count
                    self.raw_counts['Total Saved'] += row_count
                    
                    # Track entries per file
                    if 'Entries Per File' not in self.raw_counts:
                        self.raw_counts['Entries Per File'] = {}
                    self.raw_counts['Entries Per File'][unique_name] = row_count

                    if total_dropped > 0:
                        self.raw_counts['Total Dropped'] += total_dropped
                        print(f"    - Total dropped {total_dropped:,} packets containing NaN or Infinity.")
                    print(f"    -> Saved {row_count:,} samples to temp cache.")
                else:
                    print(f"    -> No valid samples found.")
            except Exception as e:
                print(f"    -> Error processing {filename}: {e}")

        if not file_info:
            raise ValueError("No valid data found in any of the provided CSV files.")

        # 3. Find the smallest dataset size
        min_samples = min(file_info.values())
        print(f"\nSmallest dataset size: {min_samples:,} entries.")
        print(f"Sampling this amount from all {len(file_info)} datasets.\n")

        # 4. Step 2: Sample from parquets
        print("Step 2: Loading samples and combining...")
        all_dfs = []
        for temp_path, total_count in file_info.items():
            filename = os.path.basename(temp_path)
            print(f"  Sampling from: {filename}...")
            try:
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(temp_path)
                
                # OPTIMIZATION: If the first row group has enough rows, only read that
                if pf.num_row_groups > 1 and pf.metadata.row_group(0).num_rows >= min_samples:
                    table = pf.read_row_group(0)
                    df = table.to_pandas()
                else:
                    df = pd.read_parquet(temp_path)
                
                if len(df) > min_samples:
                    sampled_df = df.sample(n=min_samples, random_state=42).copy()
                else:
                    sampled_df = df.copy()
                
                # Tag with source file for later distribution tracking
                sampled_df['_source_file'] = filename
                
                # Track raw counts for reporting
                benign_in_file = int(np.sum(sampled_df['Label'] == 0))
                attack_in_file = int(np.sum(sampled_df['Label'] == 1))
                self.raw_counts['Benign'] += benign_in_file
                self.raw_counts['Attack'] += attack_in_file
                
                all_dfs.append(sampled_df)
                del df
                
            except Exception as e:
                print(f"    -> Error sampling {filename}: {e}")

        if not all_dfs:
            raise ValueError("Failed to collect any data during Step 2.")

        # 5. Combine and Split
        full_df = pd.concat(all_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Total samples combined: {len(full_df):,}")

        # Split into train and test
        split_idx = int(len(full_df) * (1 - test_size))
        train_df = full_df.iloc[:split_idx].copy()
        test_df = full_df.iloc[split_idx:].copy()

        # Calculate samples per file for train/test splits
        self.raw_counts['Train Samples Per File'] = train_df['_source_file'].value_counts().to_dict()
        self.raw_counts['Test Samples Per File'] = test_df['_source_file'].value_counts().to_dict()

        # Drop the helper column before saving/returning
        train_df.drop(columns=['_source_file'], inplace=True)
        test_df.drop(columns=['_source_file'], inplace=True)

        # Save to cache
        os.makedirs(os.path.dirname(self.train_path), exist_ok=True)
        train_df.to_parquet(self.train_path)
        test_df.to_parquet(self.test_path)
        
        with open(counts_path, "w") as f:
            json.dump(self.raw_counts, f)
            
        return train_df, test_df