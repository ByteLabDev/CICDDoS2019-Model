import time
print("Starting debug...")
s = time.time()
import numpy as np
print(f"numpy: {time.time()-s:.2f}s")
s = time.time()
from custom_model import CustomLogisticRegression, ManualScaler
print(f"custom_model: {time.time()-s:.2f}s")
s = time.time()
from evaluator import Evaluator
print(f"evaluator: {time.time()-s:.2f}s")
print("Done.")
