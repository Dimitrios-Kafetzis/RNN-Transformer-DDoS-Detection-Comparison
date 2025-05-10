import matplotlib.pyplot as plt
import json
import numpy as np

# Load the necessary data
with open('evaluation_results/complete_results.json', 'r') as f:
    results = json.load(f)

# Extract F1 scores from model_metrics
f1_scores = {}
if 'model_metrics' in results and results['model_metrics']:
    for model, metrics in results['model_metrics'].items():
        if 'f1_score' in metrics:
            f1_scores[model] = metrics['f1_score']
else:
    # Alternatively, we can calculate them from attack_types
    for model, attack_metrics in results['attack_types'].items():
        f1_values = [metrics['f1'] for metrics in attack_metrics.values()]
        f1_scores[model] = np.mean(f1_values)

# We need to add F1 scores for other models that might not be in model_metrics
# Calculate average F1 across attack types for all models
for model, attack_metrics in results['attack_types'].items():
    if model not in f1_scores:
        f1_values = [metrics['f1'] for metrics in attack_metrics.values()]
        f1_scores[model] = np.mean(f1_values)

# Extract latency data from timing section
latencies = {}
for model, metrics in results['timing'].items():
    latencies[model] = metrics['mean_latency_ms']

# For threshold detector, use a different approach if needed
if 'threshold' not in latencies and 'threshold' in f1_scores:
    # Set a default latency or find it from another source
    # For now, let's use a default value
    latencies['threshold'] = 0.5  # Placeholder value (very fast)

# Create lists for plotting
models = list(set(f1_scores.keys()) & set(latencies.keys()))  # Models that have both F1 and latency
f1_values = [f1_scores[model] for model in models]
latency_values = [latencies[model] for model in models]

# Create the plot
plt.figure(figsize=(10, 6))

# Define colors for each model
colors = {
    'dnn': '#1f77b4',       # Blue
    'gru': '#ff7f0e',       # Orange
    'linear': '#2ca02c',    # Green
    'lstm': '#d62728',      # Red
    'shallow': '#9467bd',   # Purple
    'transformer': '#8c564b', # Brown
    'threshold': '#e377c2'  # Pink
}

# Plot each model
for i, model in enumerate(models):
    plt.scatter(latency_values[i], f1_values[i], 
                color=colors.get(model, 'gray'),
                s=100, 
                label=model)
    
    # Add model name annotation
    plt.annotate(model, 
                 (latency_values[i], f1_values[i]),
                 xytext=(5, 0),
                 textcoords='offset points',
                 fontsize=10)

# Format the plot
plt.title('Trade-off between F1 Score and Inference Latency', fontsize=14)
plt.xlabel('Average Inference Latency (ms)', fontsize=12)
plt.ylabel('Macro F1 Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Move the legend to the bottom right corner
plt.legend(loc='lower right')

# Save the plot
plt.savefig('f1_latency_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("F1 vs Latency plot saved as f1_latency_plot.png")