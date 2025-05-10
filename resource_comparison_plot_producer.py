import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

# Load the necessary data
with open('evaluation_results/complete_results.json', 'r') as f:
    results = json.load(f)

# Extract memory usage data
memory_data = results['memory']

# Create lists for plotting
models = list(memory_data.keys())
peak_memory = [memory_data[model].get('peak_memory_mb', 0) for model in models]
memory_increase = [memory_data[model].get('memory_increase_mb', 0) for model in models]

# Define CPU utilization data - using the values from your data
cpu_percentage = {
    'dnn': 0.300,
    'gru': 0.278,
    'linear': 0.034,
    'lstm': 0.323,
    'shallow': 0.258,
    'transformer': 22.980  # Much higher than others
}

# Create a DataFrame for easier manipulation
data = pd.DataFrame({
    'Model': models,
    'Peak Memory (MB)': peak_memory,
    'Memory Increase (MB)': memory_increase,
    'CPU Utilization (%)': [cpu_percentage.get(model, 0) for model in models]
})

# Sort by memory usage for better visualization
data = data.sort_values('Peak Memory (MB)', ascending=False)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Using darker colors
memory_color = '#2E86C1'  # Darker blue
cpu_color = '#28B463'     # Darker green

# Plot 1: Memory Usage
ax1.barh(data['Model'], data['Peak Memory (MB)'], color=memory_color)
ax1.set_xlabel('Peak Memory Usage (MB)', fontsize=12)
ax1.set_ylabel('Model', fontsize=12)
ax1.set_title('Peak Memory Usage by Model', fontsize=14)
ax1.grid(axis='x', linestyle='--', alpha=0.7)

# Add memory values on the bars
for i, v in enumerate(data['Peak Memory (MB)']):
    ax1.text(v + 50, i, f"{v:.1f} MB", va='center')

# Plot 2: CPU Utilization - Simplified approach with log scale
bars2 = ax2.barh(data['Model'], data['CPU Utilization (%)'], color=cpu_color)
ax2.set_xscale('log')  # Using log scale to handle the large difference
ax2.set_xlabel('CPU Utilization (%) - Log Scale', fontsize=12)
ax2.set_title('CPU Utilization by Model', fontsize=14)
ax2.grid(axis='x', linestyle='--', alpha=0.7)

# Add CPU values on the bars
for i, v in enumerate(data['CPU Utilization (%)']):
    # Position the text based on value to ensure visibility
    if v < 1:
        text_x = v * 1.5  # For small values, position slightly to the right
    else:
        text_x = v * 0.5  # For large values, position in the middle of the bar
    
    ax2.text(text_x, i, f"{v:.3f}%", va='center', 
             color='black' if v < 1 else 'white')  # Text color based on bar darkness

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('resource_comparison_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Resource comparison plot saved as resource_comparison_plot.png")