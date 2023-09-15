import os
import yaml
import unittest
import sys
sys.path.append('/mnt/c/Users/ghorb/OneDrive/Desktop/T2V/video_preprocessing')
from yamlEditor import createOrLoadYamlFile, loadMainYamlFile, updateMainYamlFile, updateIterationYamlFile, extractKeysFromYaml

class TestYamlFunctions(unittest.TestCase):

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

    def test_createOrLoadYamlFile(self):
        file_path = os.path.join(self.test_dir, 'test_create_or_load.yaml')
        
        # Test creating a new YAML file
        data = createOrLoadYamlFile(file_path)
        self.assertTrue(os.path.exists(file_path))
        self.assertEqual(data['Source'], '')
        
        # Test loading an existing YAML file
        loaded_data = createOrLoadYamlFile(file_path)
        self.assertEqual(data, loaded_data)

    def test_loadMainYamlFile(self):
        file_path = os.path.join(self.test_dir, 'test_load_main.yaml')
        
        # Test creating a new YAML file
        data = loadMainYamlFile(file_path)
        self.assertTrue(os.path.exists(file_path))
        self.assertEqual(data['Total_video_checkpoint'], 0)
        
        # Test loading an existing YAML file
        loaded_data = loadMainYamlFile(file_path)
        self.assertEqual(data, loaded_data)

    def test_updateMainYamlFile(self):
        file_path = os.path.join(self.test_dir, 'test_update_main.yaml')
        
        # Create an initial YAML file
        initial_data = {
            'Source': 'InitialSource',
            'Total_iterations': 5,
            'Total_video_checkpoint': 10
        }
        with open(file_path, 'w') as yaml_file:
            yaml.dump(initial_data, yaml_file)
        
        # Test updating the YAML file
        updateMainYamlFile(file_path, 'UpdatedSource', 7, 20)
        updated_data = loadMainYamlFile(file_path)
        
        self.assertEqual(updated_data['Source'], 'UpdatedSource')
        self.assertEqual(updated_data['Total_iterations'], 7)
        self.assertEqual(updated_data['Total_video_checkpoint'], 20)

    def test_updateIterationYamlFile(self):
        file_path = os.path.join(self.test_dir, 'test_update_iteration.yaml')
        
        # Create an initial YAML file
        initial_data = {
            'Source': 'InitialSource',
            'failed_indexes': [1, 2, 3],
            'total_videos': 10,
            'video_checkpoint_from_to': [0, 5]
        }
        with open(file_path, 'w') as yaml_file:
            yaml.dump(initial_data, yaml_file)
        
        # Test updating the YAML file
        updateIterationYamlFile(file_path, 'UpdatedSource', [4, 5, 6], 15, [1, 10])
        updated_data = loadMainYamlFile(file_path)
        
        self.assertEqual(updated_data['Source'], 'UpdatedSource')
        self.assertEqual(updated_data['failed_indexes'], [4, 5, 6])
        self.assertEqual(updated_data['total_videos'], 15)
        self.assertEqual(updated_data['video_checkpoint_from_to'], [1, 10])

    def test_extractKeysFromYaml(self):
        data = {
            'Source': '',
            'failed_indexes': [],
            'total_videos': 0,
            'video_checkpoint_from_to': [],
            'nested_data': {
                'key1': 'value1',
                'key2': 'value2'
            }
        }
        
        keys = extractKeysFromYaml(data)
        
        # Check if all keys are extracted correctly
        self.assertIn('Source', keys)
        self.assertIn('failed_indexes', keys)
        self.assertIn('total_videos', keys)
        self.assertIn('video_checkpoint_from_to', keys)
        self.assertIn('nested_data_key1', keys)
        self.assertIn('nested_data_key2', keys)

    def test_createOrLoadYamlFile_existing_file(self):
        file_path = os.path.join(self.test_dir, 'test_existing_file.yaml')
        
        # Create an initial YAML file
        initial_data = {
            'Source': 'InitialSource',
            'Total_iterations': 5,
            'Total_video_checkpoint': 10
        }
        with open(file_path, 'w') as yaml_file:
            yaml.dump(initial_data, yaml_file)
        
        # Test creating or loading an existing YAML file
        loaded_data = createOrLoadYamlFile(file_path)
        
        self.assertTrue(os.path.exists(file_path))
        self.assertEqual(loaded_data['Source'], 'InitialSource')
        self.assertEqual(loaded_data['Total_iterations'], 5)
        self.assertEqual(loaded_data['Total_video_checkpoint'], 10)

if __name__ == '__main__':
    unittest.main()
