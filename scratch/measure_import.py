import time

start = time.time()
print("Importing pandas...")
import pandas as pd
print(f"Pandas took {time.time() - start:.2f}s")

start = time.time()
print("Importing numpy...")
import numpy as np
print(f"Numpy took {time.time() - start:.2f}s")

start = time.time()
print("Importing data_loader...")
import data_loader
print(f"data_loader took {time.time() - start:.2f}s")
