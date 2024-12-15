import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data_file = 'CS562  Project final result - Shawn - Mistral.csv'
data = pd.read_csv(data_file, header=1)

# Inspect the first few rows of the data to understand the structure
print(data.head())

# Replace 'Category' and 'Phase 1 Label' with actual column names from your dataset
category_column = 'category'  # Replace with the actual column name for categories
phase1_label_column = 'Phase 2 label'  # Replace with the actual column name for Phase 1 labels

# Filter rows where Phase 1 Label is 2 (harmful) and count occurrences of each category
filtered_data = data[data[phase1_label_column] == 2]
grouped_data = filtered_data.groupby(category_column)[phase1_label_column].count()

# Convert the grouped data to a DataFrame
grouped_data_df = grouped_data.reset_index()
grouped_data_df.columns = [category_column, 'Count of harmful output']

# Plot the bar graph
plt.figure(figsize=(10, 6))
plt.bar(grouped_data_df[category_column], grouped_data_df['Count of harmful output'], color='skyblue')
plt.xlabel('Category')
plt.ylabel('Count of Phase 2 harmful output')
plt.title('mistral_small_latest - Count of harmful output by category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show the plot
plt.show()