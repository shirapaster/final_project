import pandas as pd
import os

def load_data(file_path):
    """Load the dataset from the specified file path."""
    return pd.read_csv(file_path)

def inspect_data(data):
    """Provide an overview of the dataset."""
    print("Data Overview:")
    print(f"Number of rows: {data.shape[0]}")
    print(f"Number of columns: {data.shape[1]}")
    print("Data types and missing values per column:")
    print(data.dtypes)
    print(data.isnull().sum())
    print("Statistical summary of numeric columns:")
    print(data.describe())

def drop_columns_with_many_missing(data, threshold=0.2):
    """Drop columns with more than the specified threshold of missing values."""
    missing_values = data.isnull().sum()
    columns_to_drop = missing_values[missing_values > len(data) * threshold].index
    print("Columns with more than 20% missing values (to be dropped):")
    print(columns_to_drop)
    return data.drop(columns=columns_to_drop)

def fill_missing_values(data, columns):
    """Fill missing values in specified columns with their mean."""
    for column in columns:
        if data[column].isnull().sum() > 0:
            data[column] = data[column].fillna(data[column].mean())
    return data

def detect_outliers(data, column):
    """Detect outliers in a specified column based on the 1.5*IQR rule."""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

def remove_outliers(data, columns):
    """Remove outliers from specified columns."""
    for column in columns:
        outliers = detect_outliers(data, column)
        data = data[~data.index.isin(outliers.index)]
    return data

def save_cleaned_data(data, file_path):
    """Save the cleaned dataset to the specified path."""
    if os.path.exists(file_path):
        print(f"File already exists at: {file_path}. No new file created.")
    else:
        data.to_csv(file_path, index=False)
        print(f"Cleaned and relevant data saved to: {file_path}")

def main():
    file_path = './src/Data_Cortex_Nuclear.csv'
    cleaned_data_path = './src/cleaned_relevant_data.csv'

    # Load the dataset
    data = load_data(file_path)

    # Step 1: Inspect the data
    inspect_data(data)

    # Step 2: Drop columns with many missing values
    data_cleaned = drop_columns_with_many_missing(data)

    # Step 3: Fill missing values for relevant columns
    data_cleaned = fill_missing_values(data_cleaned, ['BDNF_N', 'pCREB_N'])

    # Step 4: Remove outliers
    data_cleaned = remove_outliers(data_cleaned, ['BDNF_N', 'pCREB_N'])

    # Step 5: Create a new focused dataset
    relevant_columns = ['MouseID', 'Genotype', 'Treatment', 'BDNF_N', 'pCREB_N']
    filtered_data = data_cleaned[relevant_columns]

    # Step 6: Save the cleaned and filtered dataset
    save_cleaned_data(filtered_data, cleaned_data_path)

if __name__ == "__main__":
    main()
