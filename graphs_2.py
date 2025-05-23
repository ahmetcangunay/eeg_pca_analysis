import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Get all CSV files from the comparison_graphs directory
csv_files = glob.glob('comparison_graphs/*.csv')

# Create an empty list to store DataFrames
dfs = []

# Read each CSV file and add model name
for file in csv_files:
    # Extract model name from filename (remove _metrics.csv and convert to lowercase)
    model_name = os.path.basename(file).replace('_metrics.csv', '')
    
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Add model column
    df['model'] = model_name
    
    # Append to list
    dfs.append(df)

# Combine all DataFrames
combined_df = pd.concat(dfs, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('combined_metrics.csv', index=False)

# Set the style
plt.style.use('default')
sns.set_theme()

# Create separate plots for each metric and version
versions = combined_df['Version'].unique()
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']

for version in versions:
    # Filter data for current version
    version_data = combined_df[combined_df['Version'] == version]
    
    # Sort models by Accuracy
    version_data = version_data.sort_values('Accuracy')
    
    # Create a separate plot for each metric
    for metric in metrics:
        # Create a new figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create bar plot for single metric
        bars = ax.bar(version_data['model'], version_data[metric], width=0.6)
        
        # Customize the plot
        ax.set_title(f'{metric} Scores - Version: {version}', fontsize=16, pad=20)
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', rotation=0)
        
        # Set y-axis limits to start from 0
        ax.set_ylim(0, 1.1)
        
        # Add grid for better readability
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'individual_metric_graphs/{metric.lower()}_comparison_{version}.png', bbox_inches='tight', dpi=300)
        plt.close(fig)

print("Individual metric comparison graphs have been saved.")
