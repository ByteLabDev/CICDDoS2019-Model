# evaluator.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    def calculate_metrics(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}, [tn, fp, fn, tp]

    def plot_results(self, metrics, conf_matrix):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Metrics Bar
        ax1.bar(metrics.keys(), metrics.values(), color='skyblue')
        ax1.set_ylim(0, 1)
        ax1.set_title("Performance Metrics")

        # Confusion Matrix
        tn, fp, fn, tp = conf_matrix
        cm = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax2)
        ax2.set_title("Confusion Matrix")
        plt.show()

    def plot_class_distribution(self, raw_counts, balanced_counts):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Raw Data Imbalance
        ax1.bar(raw_counts.keys(), raw_counts.values(), color=['lightgreen', 'salmon'])
        ax1.set_title(f"Raw Data Class Imbalance\n(Total: {sum(raw_counts.values()):,})")
        ax1.set_ylabel("Number of Samples")
        
        # Balanced Data
        ax2.bar(balanced_counts.keys(), balanced_counts.values(), color=['lightgreen', 'salmon'])
        ax2.set_title(f"Balanced Data (Used for Model)\n(Total: {sum(balanced_counts.values()):,})")
        ax2.set_ylabel("Number of Samples")
        
        plt.tight_layout()
        plt.show()