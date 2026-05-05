import os
import pandas as pd

raw_dir = 'data/'
all_files = []
for root, _, files in os.walk(raw_dir):
    for file in files:
        if file.endswith(".csv") and not file.startswith(".~"):
            all_files.append(os.path.join(root, file))

print(f"Found {len(all_files)} files.")
for f in all_files:
    print(f" - {f}")

if all_files:
    print(f"Reading first chunk of {all_files[0]}...")
    chunk = pd.read_csv(all_files[0], nrows=10)
    print("Success.")
    print(chunk.head())
