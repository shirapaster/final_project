import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from datacleaning import load_data, inspect_data, drop_columns_with_many_missing, fill_missing_values, detect_outliers, remove_outliers, save_cleaned_data

class TestDataCleaning(unittest.TestCase):

    def setUp(self):
        """Set up a mock dataset for testing."""
        # Positive Test Case: Valid dataset for general tests
        self.mock_data = pd.DataFrame({
            'MouseID': ['M1', 'M2', 'M3', 'M4'],
            'Genotype': ['A', 'B', 'A', 'B'],
            'Treatment': ['X', 'Y', 'X', 'Y'],
            'BDNF_N': [0.5, 0.7, None, 1.2],
            'pCREB_N': [0.3, 0.8, 1.1, None],
            'Extra': [None, None, None, None]  # Null Test Case: Column with all missing values
        })
        self.test_file_path = './test_cleaned_data.csv'

    def tearDown(self):
        """Clean up any created files."""
        try:
            os.remove(self.test_file_path)
        except FileNotFoundError:
            pass

    def test_load_data(self):
        """Test loading a dataset."""
        # Error Test Case: Handle missing file scenario
        mock_file_path = './src/Data_Cortex_Nuclear.csv'
        try:
            data = load_data(mock_file_path)
            self.assertIsInstance(data, pd.DataFrame)
        except FileNotFoundError:
            self.skipTest("File not found, skipping test.")

    def test_drop_columns_with_many_missing(self):
        """Test dropping columns with more than 20% missing values."""
        # Edge Test Case: Column with exactly 50% missing values
        cleaned_data = drop_columns_with_many_missing(self.mock_data, threshold=0.5)
        self.assertNotIn('Extra', cleaned_data.columns)  # Extra column should be dropped

    def test_fill_missing_values(self):
        """Test filling missing values with column mean."""
        # Positive Test Case: Columns with missing values filled with mean
        columns_to_fill = ['BDNF_N', 'pCREB_N']
        filled_data = fill_missing_values(self.mock_data.copy(), columns_to_fill)
        self.assertFalse(filled_data['BDNF_N'].isnull().any())
        self.assertFalse(filled_data['pCREB_N'].isnull().any())

    def test_detect_outliers(self):
        """Test detecting outliers based on IQR."""
        # Boundary Test Case: Check for outliers in data within IQR boundaries
        outliers = detect_outliers(self.mock_data, 'BDNF_N')
        self.assertTrue(outliers.empty or isinstance(outliers, pd.DataFrame))

    def test_remove_outliers(self):
        """Test removing outliers from specified columns."""
        # Negative Test Case: Remove valid outliers and check length
        cleaned_data = remove_outliers(self.mock_data.copy(), ['BDNF_N', 'pCREB_N'])
        self.assertTrue(len(cleaned_data) <= len(self.mock_data))

    def test_save_cleaned_data(self):
        """Test saving cleaned data to a file."""
        # Error Test Case: Ensure file saving works correctly
        save_cleaned_data(self.mock_data, self.test_file_path)
        self.assertTrue(os.path.exists(self.test_file_path))

if __name__ == '__main__':
    unittest.main()
