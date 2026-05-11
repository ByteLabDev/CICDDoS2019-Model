import time
import sys

def check_import(module_name):
    start = time.time()
    if module_name in sys.modules:
        del sys.modules[module_name]
    __import__(module_name)
    end = time.time()
    print(f"Import {module_name} took: {end - start:.4f}s")

print("Checking optimization results...")
check_import('data_loader')
check_import('evaluator')
check_import('main')
