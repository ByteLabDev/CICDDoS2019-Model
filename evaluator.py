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

    def plot_interactive_view(self, raw_counts, balanced_counts, metrics, conf_matrix, df, top_n=15):
        import pandas as pd
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        current_view = [0]
        views_count = 5
        
        def draw_view(view_idx):
            ax.clear()
            if view_idx == 0:
                ax.bar(raw_counts.keys(), raw_counts.values(), color=['lightgreen', 'salmon'])
                ax.set_title(f"Raw Data Class Imbalance\n(Total: {sum(raw_counts.values()):,})")
                ax.set_ylabel("Number of Samples")
            elif view_idx == 1:
                ax.bar(balanced_counts.keys(), balanced_counts.values(), color=['lightgreen', 'salmon'])
                ax.set_title(f"Balanced Data (Used for Model)\n(Total: {sum(balanced_counts.values()):,})")
                ax.set_ylabel("Number of Samples")
            elif view_idx == 2:
                ax.bar(metrics.keys(), metrics.values(), color='skyblue')
                ax.set_ylim(0, 1.05)
                for i, v in enumerate(metrics.values()):
                    ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
                ax.set_title("Performance Metrics")
            elif view_idx == 3:
                tn, fp, fn, tp = conf_matrix
                cm = np.array([[tn, fp], [fn, tp]])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False)
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
            elif view_idx == 4:
                if 'Label' in df.columns:
                    if len(df) > 50000:
                        df_sample = df.sample(50000, random_state=42)
                    else:
                        df_sample = df
                    correlations = df_sample.corr()['Label'].abs().sort_values(ascending=False)
                    top_features = correlations.index[:top_n].tolist()
                    corr_matrix = df_sample[top_features].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
                    ax.set_title(f"Correlation Matrix (Top {top_n} Features Correlated with Label)")
                else:
                    ax.text(0.5, 0.5, "Label column not found", ha='center', va='center')
            
            fig.canvas.draw_idle()

        def next_view(*args, **kwargs):
            current_view[0] = (current_view[0] + 1) % views_count
            draw_view(current_view[0])
            
        def prev_view(*args, **kwargs):
            current_view[0] = (current_view[0] - 1) % views_count
            draw_view(current_view[0])

        # Hijack the default Matplotlib toolbar's back and forward buttons
        if fig.canvas.toolbar is not None:
            toolbar = fig.canvas.toolbar
            
            # Re-wire the actual UI buttons to our functions
            if hasattr(toolbar, '_buttons'): # Tkinter backend
                if 'Back' in toolbar._buttons:
                    toolbar._buttons['Back'].config(command=prev_view)
                if 'Forward' in toolbar._buttons:
                    toolbar._buttons['Forward'].config(command=next_view)
            elif hasattr(toolbar, '_actions'): # Qt backend
                if 'Back' in toolbar._actions:
                    try:
                        toolbar._actions['Back'].triggered.disconnect()
                        toolbar._actions['Back'].triggered.connect(prev_view)
                    except: pass
                if 'Forward' in toolbar._actions:
                    try:
                        toolbar._actions['Forward'].triggered.disconnect()
                        toolbar._actions['Forward'].triggered.connect(next_view)
                    except: pass

            # Fallback for underlying methods
            toolbar.forward = next_view
            toolbar.back = prev_view
            
            # Override set_history_buttons to ensure our hijacked buttons stay enabled
            def override_set_history_buttons(*args, **kwargs):
                if hasattr(toolbar, '_actions'):
                    if 'Back' in toolbar._actions:
                        toolbar._actions['Back'].set_enabled(True)
                    if 'Forward' in toolbar._actions:
                        toolbar._actions['Forward'].set_enabled(True)
                elif hasattr(toolbar, '_buttons'):
                    if 'Back' in toolbar._buttons:
                        toolbar._buttons['Back'].config(state='normal')
                    if 'Forward' in toolbar._buttons:
                        toolbar._buttons['Forward'].config(state='normal')
                        
            toolbar.set_history_buttons = override_set_history_buttons
            toolbar.set_history_buttons()

        draw_view(0)
        plt.show()