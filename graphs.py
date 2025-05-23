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

# Create a figure with subplots for each version
versions = combined_df['Version'].unique()
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']

# Set the style
plt.style.use('default')
sns.set_theme()

# Create a figure with subplots for each version
for idx, version in enumerate(versions):
    # Filter data for current version
    version_data = combined_df[combined_df['Version'] == version]
    
    # Sort models by Accuracy
    version_data = version_data.sort_values('Accuracy')
    
    # Create a new figure for each version
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Create bar plot
    version_data.plot(x='model', y=metrics, kind='bar', ax=ax, width=0.8)
    
    # Customize the plot
    ax.set_title(f'Version: {version}', fontsize=20, y=1.1)
    ax.set_xlabel('Models', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)
    
    # Add value labels on top of bars
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=8, rotation=45, labels=[f'{v:.3f}' for v in i.datavalues])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'model_comparison_{version}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

print("Comparison graphs saved as 'model_comparison_<version>.png'")
