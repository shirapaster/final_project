import pandas as pd
import os

# Load the dataset
file_path = './src/Data_Cortex_Nuclear.csv'
data = pd.read_csv(file_path)

# Step 1: Initial data inspection
print("Data Overview:")

# Display basic information about the dataset
print(f"Number of rows: {data.shape[0]}")
print(f"Number of columns: {data.shape[1]}")

# Display data types and missing values
print("Data types and missing values per column:")
print(data.dtypes)
print(data.isnull().sum())

# Display statistical summary of numeric columns
print("Statistical summary of numeric columns:")
print(data.describe())

# Step 2: Check for missing values
# Identify columns with more than 20% missing values
missing_values = data.isnull().sum()
columns_with_many_missing = missing_values[missing_values > len(data) * 0.2].index
print("Columns with more than 20% missing values (to be dropped):")
print(columns_with_many_missing)

# Drop columns with more than 20% missing values
# This ensures the dataset is manageable and focused on reliable data
data_cleaned = data.drop(columns=columns_with_many_missing)

# Step 3: Fill missing values for relevant columns
# Filling missing values in selected columns with the mean
for column in ['BDNF_N', 'pCREB_N']:
    if data_cleaned[column].isnull().sum() > 0:
        data_cleaned[column] = data_cleaned[column].fillna(data_cleaned[column].mean())

# Verify there are no more missing values in the relevant columns
print("Missing values after filling in relevant columns:")
print(data_cleaned[['BDNF_N', 'pCREB_N']].isnull().sum())

# Step 4: Detect and handle outliers
# Function to detect outliers based on the 1.5*IQR rule
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Detect outliers in BDNF_N and pCREB_N
outliers_BDNF = detect_outliers(data_cleaned, 'BDNF_N')
outliers_pCREB = detect_outliers(data_cleaned, 'pCREB_N')

print("Number of outliers in BDNF_N:", len(outliers_BDNF))
print("Number of outliers in pCREB_N:", len(outliers_pCREB))

# Optionally remove outliers
# Outliers are removed to ensure the analysis is not skewed
data_cleaned = data_cleaned[~data_cleaned.index.isin(outliers_BDNF.index)]
data_cleaned = data_cleaned[~data_cleaned.index.isin(outliers_pCREB.index)]

# Step 5: Final cleaning validation
# Verify there are no missing values or extreme outliers left
print("Missing values after cleaning:")
print(data_cleaned.isnull().sum())

# Display cleaned data overview
print("Cleaned data overview:")
print(data_cleaned.describe())

# Step 6: Create a new focused dataset
# Select only relevant columns for the research questions
relevant_columns = ['MouseID', 'Genotype', 'Treatment', 'BDNF_N', 'pCREB_N']
filtered_data = data_cleaned[relevant_columns]

# Define the path to save the cleaned data within the src directory
cleaned_data_path = './src/cleaned_relevant_data.csv'

# Check if the file already exists
if os.path.exists(cleaned_data_path):
    print(f"File already exists at: {cleaned_data_path}. No new file created.")
else:
    # Save the cleaned and filtered dataset to the specified path
    filtered_data.to_csv(cleaned_data_path, index=False)
    print(f"Cleaned and relevant data saved to: {cleaned_data_path}")


