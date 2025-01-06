import unittest
import pandas as pd
from datacleaning import * 

class TestDataCleaning(unittest.TestCase):
    """
    Unit tests for data cleaning functions.
    """

    def setUp(self):
        """
        Set up a sample dataset for testing.
        """
        self.data = pd.DataFrame({
            'A': [1, 2, 3, None, 5],       # Column with some missing values
            'B': [10, None, 30, 40, None], # Column with missing values
            'C': [None, None, None, None, None],  # Column to be removed (all missing)
            'D': [5, 15, 25, 35, 45]       # Fully populated column
        })

    def test_remove_columns_with_missing_values(self):
        """
        Positive Test Case:
        Test that columns with more than 20% missing values are removed correctly.
        """
        cleaned_data = datacleaning.clean_data(self.data)
        self.assertNotIn('C', cleaned_data.columns, "Column 'C' should have been removed.")
        self.assertIn('A', cleaned_data.columns, "Column 'A' should not have been removed.")

    def test_fill_missing_values(self):
        """
        Positive Test Case:
        Test that missing values are filled with the mean of the respective column.
        """
        cleaned_data = datacleaning.clean_data(self.data)
        # Check if missing values in column 'A' were replaced with the mean
        self.assertAlmostEqual(cleaned_data['A'].iloc[3], 2.75, msg="Missing value in 'A' not filled with mean.")
        # Check if missing values in column 'B' were replaced with the mean
        self.assertAlmostEqual(cleaned_data['B'].iloc[1], 26.6666667, msg="Missing value in 'B' not filled with mean.")

    def test_detect_outliers(self):
        """
        Boundary Test Case:
        Test the detection of outliers using the IQR method.
        """
        data_with_outliers = pd.DataFrame({'A': [1, 2, 3, 4, 100]})  # 100 is an outlier
        outliers = datacleaning.detect_outliers(data_with_outliers, 'A')
        self.assertEqual(len(outliers), 1, "There should be exactly one outlier detected.")
        self.assertEqual(outliers.iloc[0]['A'], 100, "The detected outlier should be 100.")

    def test_file_not_found(self):
        """
        Error Test Case:
        Test behavior when the input file does not exist.
        """
        with self.assertRaises(FileNotFoundError, msg="FileNotFoundError was not raised for a missing file."):
            pd.read_csv('non_existent_file.csv')

    def test_invalid_data_type(self):
        """
        Negative Test Case:
        Test behavior when a column contains invalid data types (e.g., strings in numeric columns).
        """
        invalid_data = pd.DataFrame({'A': ['a', 'b', 'c', 4, 5]})  # Invalid strings in numeric column
        with self.assertRaises(ValueError, msg="ValueError was not raised for invalid data types."):
            datacleaning.clean_data(invalid_data)

    def test_empty_dataset(self):
        """
        Edge Test Case:
        Test behavior when the dataset is completely empty.
        """
        empty_data = pd.DataFrame()  # Empty dataset
        cleaned_data = datacleaning.clean_data(empty_data)
        self.assertTrue(cleaned_data.empty, "The cleaned dataset should be empty for an empty input.")

if __name__ == '__main__':
    unittest.main()
