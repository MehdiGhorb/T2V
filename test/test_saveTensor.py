import unittest
import torch
import os
import sys

sys.path.append('../src/common')
import paths
sys.path.append(paths.TENSOR_GEN_PATH)
from saveTensor import saveTensor, loadTensor

class TestTensorFunctions(unittest.TestCase):

    def setUp(self):
        self.test_data = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        self.test_path = 'test_tensor.pt'

    def tearDown(self):
        if os.path.exists(self.test_path):
            os.remove(self.test_path)

    def test_save_and_load_tensor(self):
        # Test saveTensor and loadTensor functions
        saveTensor(self.test_data, self.test_path)
        loaded_data = loadTensor(self.test_path)

        self.assertTrue(torch.equal(self.test_data[0], loaded_data[0]))
        self.assertTrue(torch.equal(self.test_data[1], loaded_data[1]))
        self.assertEqual(len(self.test_data), len(loaded_data))

    def test_load_nonexistent_file(self):
        # Test loading a nonexistent file
        with self.assertRaises(Exception) as context:
            loadTensor('nonexistent.pt')

        self.assertTrue("No such file or directory" in str(context.exception))

if __name__ == '__main__':
    unittest.main()
