import unittest
import os
import csv
import sys
sys.path.append('../src/common')
import paths
sys.path.append(os.path.join(paths.UTILS_DIR, 'video_preprocessing'))
from videoPrepHelper import read_data

class TestReadData(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = 'test_data'
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        # Clean up the temporary directory after testing
        if os.path.exists(self.test_dir):
            for root, dirs, files in os.walk(self.test_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.test_dir)

    def test_read_data_valid_range(self):
        # Create a test CSV file
        csv_file_path = os.path.join(self.test_dir, 'test_data.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Name', 'Age'])
            csvwriter.writerow(['Alice', 25])
            csvwriter.writerow(['Bob', 30])
            csvwriter.writerow(['Charlie', 40])

        # Test reading data from the CSV file within a valid range
        data = read_data(csv_file_path, start_index=1, end_index=2)
        expected_data = [['Bob', 30], ['Charlie', 40]]
        
        self.assertEqual(data, expected_data)

    def test_read_data_start_index_greater_than_end_index(self):
        # Create a test CSV file
        csv_file_path = os.path.join(self.test_dir, 'test_data.csv')

        # Ensure a FileNotFoundError is raised
        with self.assertRaises(IndexError):
            read_data(csv_file_path, start_index=10, end_index=1)

    def test_read_data_file_not_found(self):
        # Test when the CSV file does not exist
        csv_file_path = 'non_existent_file.csv'

        # Ensure a FileNotFoundError is raised
        with self.assertRaises(FileNotFoundError):
            read_data(csv_file_path, start_index=0, end_index=1)

    def test_read_data_skip_header(self):
        # Create a test CSV file with a header
        csv_file_path = os.path.join(self.test_dir, 'test_data.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Name', 'Age'])
            csvwriter.writerow(['Alice', 25])
            csvwriter.writerow(['Bob', 30])

        # Test reading data and skipping the header row
        data = read_data(csv_file_path, start_index=0, end_index=1)
        expected_data = [['Alice', 25], ['Bob', 30]]
        
        self.assertEqual(data, expected_data)

if __name__ == '__main__':
    unittest.main()
