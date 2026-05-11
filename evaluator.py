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

    def plot_interactive_view(self, raw_counts, before_smote_counts, balanced_counts, train_metrics, train_conf_matrix, test_metrics, test_conf_matrix, df, top_n=15):
        import pandas as pd
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Capture original axes position to restore it after heatmap shrinking
        original_ax_pos = ax.get_position()
        
        current_view = [0]
        views_count = 10
        
        def draw_view(view_idx):
            # Clear any existing colorbars/extra axes from previous views
            for extra_ax in fig.axes:
                if extra_ax is not ax:
                    fig.delaxes(extra_ax)
            
            ax.clear()
            ax.set_position(original_ax_pos) # Restore original position
            ax.axis('on')  # Ensure axis is on (it might have been turned off by View 9)
            
            if view_idx == 0:
                # Filter out stats like 'Total Dropped' and 'Total Saved' for the bar chart
                plot_data = {k: v for k, v in raw_counts.items() if k in ['Benign', 'Attack']}
                ax.bar(plot_data.keys(), plot_data.values(), color=['lightgreen', 'salmon'])
                ax.set_title(f"Raw Data Class Imbalance\n(Total: {sum(plot_data.values()):,})")
                ax.set_ylabel("Number of Samples")
            elif view_idx == 1:
                ax.bar(before_smote_counts.keys(), before_smote_counts.values(), color=['lightgreen', 'salmon'])
                ax.set_title(f"Before SMOTE Data (Training Split)\n(Total: {sum(before_smote_counts.values()):,})")
                ax.set_ylabel("Number of Samples")
            elif view_idx == 2:
                ax.bar(balanced_counts.keys(), balanced_counts.values(), color=['lightgreen', 'salmon'])
                ax.set_title(f"After SMOTE Data (Used for Training)\n(Total: {sum(balanced_counts.values()):,})")
                ax.set_ylabel("Number of Samples")
            elif view_idx == 3:
                ax.bar(train_metrics.keys(), train_metrics.values(), color='mediumseagreen')
                ax.set_ylim(0, 1.05)
                for i, v in enumerate(train_metrics.values()):
                    ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
                ax.set_title("Training Performance Metrics (Balanced Data)")
            elif view_idx == 4:
                tn, fp, fn, tp = train_conf_matrix
                cm = np.array([[tn, fp], [fn, tp]])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False)
                ax.set_title("Training Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
            elif view_idx == 5:
                ax.bar(test_metrics.keys(), test_metrics.values(), color='skyblue')
                ax.set_ylim(0, 1.05)
                for i, v in enumerate(test_metrics.values()):
                    ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
                ax.set_title("Testing Performance Metrics (Original Distribution)")
            elif view_idx == 6:
                tn, fp, fn, tp = test_conf_matrix
                cm = np.array([[tn, fp], [fn, tp]])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                ax.set_title("Testing Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
            elif view_idx == 7:
                if 'Label' in df.columns:
                    df_sample = df.sample(min(50000, len(df)), random_state=42)
                    correlations = df_sample.corr()['Label'].abs().sort_values(ascending=False)
                    top_features = correlations.index[:top_n].tolist()
                    corr_matrix = df_sample[top_features].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
                    ax.set_title(f"Correlation Matrix (Top {top_n} Features Correlated with Label)")
                else:
                    ax.text(0.5, 0.5, "Label column not found", ha='center', va='center')
            elif view_idx == 8:
                if 'Label' in df.columns:
                    df_sample = df.sample(min(50000, len(df)), random_state=42)
                    corr_matrix = df_sample.corr()
                    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax, xticklabels=True, yticklabels=True)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                    ax.set_title(f"Correlation Matrix (All {len(df_sample.columns)} Features)")
                    fig.tight_layout()
                else:
                    ax.text(0.5, 0.5, "Label column not found", ha='center', va='center')
            
            # --- VIEW: DATASET FACTS ---
            elif view_idx == 9:
                ax.axis('off')
                facts_text = (
                    "CIC-DDoS2019 Dataset Facts\n"
                    "--------------------------------------------------\n\n"
                    "• Dataset Origin: Canadian Institute for Cybersecurity (UNB)\n"
                    "• Total Original Features: 87 (extracted via CICFlowMeter-V3)\n"
                    "• Numeric Features: 80 (statistical flow metrics)\n"
                    "• Non-numeric Features: 7 (Flow ID, IP Source/Dest, Timestamp, etc.)\n"
                    "• Attack Types: 12 (NTP, DNS, LDAP, MSSQL, NetBIOS, SNMP, SSDP, UDP, etc.)\n"
                    "• Primary Paper: Sharafaldin et al. (2019)\n\n"
                    "Preprocessing Applied in this Session:\n"
                    f"• Current Column Count: {len(df.columns)}\n"
                    f"• Total Valid Samples: {raw_counts.get('Total Saved', len(df)):,}\n"
                    f"• Dropped (NaN/Inf): {raw_counts.get('Total Dropped', 0):,}\n"
                    "• Balanced via SMOTE: Yes\n"
                    "• Non-numeric Labels: Removed"
                )
                ax.text(0.05, 0.95, facts_text, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            fig.canvas.draw_idle()

        def next_view(*args, **kwargs):
            current_view[0] = (current_view[0] + 1) % views_count
            draw_view(current_view[0])
            
        def prev_view(*args, **kwargs):
            current_view[0] = (current_view[0] - 1) % views_count
            draw_view(current_view[0])

        if fig.canvas.toolbar is not None:
            toolbar = fig.canvas.toolbar
            if hasattr(toolbar, '_buttons'):
                if 'Back' in toolbar._buttons: toolbar._buttons['Back'].config(command=prev_view)
                if 'Forward' in toolbar._buttons: toolbar._buttons['Forward'].config(command=next_view)
            elif hasattr(toolbar, '_actions'):
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

            toolbar.forward = next_view
            toolbar.back = prev_view
            
            def override_set_history_buttons(*args, **kwargs):
                if hasattr(toolbar, '_actions'):
                    if 'Back' in toolbar._actions: toolbar._actions['Back'].set_enabled(True)
                    if 'Forward' in toolbar._actions: toolbar._actions['Forward'].set_enabled(True)
                elif hasattr(toolbar, '_buttons'):
                    if 'Back' in toolbar._buttons: toolbar._buttons['Back'].config(state='normal')
                    if 'Forward' in toolbar._buttons: toolbar._buttons['Forward'].config(state='normal')
                        
            toolbar.set_history_buttons = override_set_history_buttons
            toolbar.set_history_buttons()

        draw_view(0)
        plt.show()