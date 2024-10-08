import pandas as pd

# Read CSV file into a pandas DataFrame
df = pd.read_csv('GradCAM_Text.csv')

# Extract rows where the column has the fixed value
filtered_df = df['TestImage_name']

# Write DataFrame to a text file without truncating values
filtered_df.to_csv('output.txt', index=False)
