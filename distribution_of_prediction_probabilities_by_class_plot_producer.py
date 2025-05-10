import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import os

# Load necessary data
def load_predictions_and_labels():
    # Load raw predictions from the transformer model
    transformer_probs_path = os.path.join("evaluation_results/predictions", "transformer_1746776936_raw_probs.npy")
    transformer_probs = np.load(transformer_probs_path)
    
    # Load true labels
    true_labels_path = os.path.join("evaluation_results/predictions", "y_true.npy")
    true_labels = np.load(true_labels_path)
    
    # Ensure same length (in case of any mismatches)
    min_length = min(len(transformer_probs), len(true_labels))
    transformer_probs = transformer_probs[:min_length]
    true_labels = true_labels[:min_length]
    
    return transformer_probs, true_labels

# Create the distribution plot
def create_distribution_plot(probs, labels):
    plt.figure(figsize=(10, 6))
    
    # Split probabilities by class
    normal_probs = probs[labels == 0]
    attack_probs = probs[labels == 1]
    
    # Plot distribution with kernel density estimation
    sns.histplot(normal_probs, kde=True, stat="density", 
                label="Normal Traffic (Class 0)", 
                color="blue", alpha=0.5, 
                bins=30)
    
    sns.histplot(attack_probs, kde=True, stat="density", 
                label="Attack Traffic (Class 1)", 
                color="red", alpha=0.5, 
                bins=30)
    
    # Find the range for optimal threshold
    mean_normal = normal_probs.mean()
    mean_attack = attack_probs.mean()
    middle_threshold = (mean_normal + mean_attack) / 2
    
    # Add vertical line for optimal threshold
    plt.axvline(middle_threshold, color='black', linestyle='--', 
               label=f'Optimal Threshold: {middle_threshold:.4f}')
    
    # Add annotations
    plt.text(mean_normal, plt.gca().get_ylim()[1]*0.9, 
            f'Mean (Normal): {mean_normal:.4f}', 
            color='blue', ha='center')
    
    plt.text(mean_attack, plt.gca().get_ylim()[1]*0.8, 
            f'Mean (Attack): {mean_attack:.4f}', 
            color='red', ha='center')
    
    # Format plot
    plt.title('Distribution of Prediction Probabilities by Class', fontsize=14)
    plt.xlabel('Prediction Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Adjust x-axis to focus on the relevant probability range
    min_prob = min(probs) * 0.95
    max_prob = max(probs) * 1.05
    plt.xlim(min_prob, max_prob)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('Distribution_of_Prediction_Probabilities_by_Class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved as Distribution_of_Prediction_Probabilities_by_Class.png")
    print(f"Normal class predictions range: {normal_probs.min():.5f} - {normal_probs.max():.5f}")
    print(f"Attack class predictions range: {attack_probs.min():.5f} - {attack_probs.max():.5f}")
    print(f"Suggested threshold: {middle_threshold:.5f}")

# Main execution
if __name__ == "__main__":
    # Load the data
    probs, labels = load_predictions_and_labels()
    
    # Print summary statistics
    print(f"Total samples: {len(probs)}")
    print(f"Normal samples: {np.sum(labels == 0)}")
    print(f"Attack samples: {np.sum(labels == 1)}")
    print(f"Overall probability range: {probs.min():.5f} - {probs.max():.5f}")
    
    # Create and save the plot
    create_distribution_plot(probs, labels)