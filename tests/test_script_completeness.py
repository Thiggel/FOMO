import unittest
import os

class TestScripts(unittest.TestCase):
    def setUp(self):
        self.jobs_directory = 'jobs'
        self.required_elements = {
            'ResNet18': ['imbalance_method'],
        }
        #if we don't have key, we should obligatory have vals
        self.ifelses = {
            'no_pretrain': ['max_cycles'],
        }

    def test_shell_scripts(self):
        for root, dirs, files in os.walk(self.jobs_directory):
            for file in files:
                if file.endswith('.sh') and 'run' not in file:
                    self.check_script(os.path.join(root, file))

    def check_script(self, filepath):
        with open(filepath, 'r') as file:
            content = file.read()
            for key, vals in self.required_elements.items():
                for value in vals:
                    if key in content and value not in content:
                        self.fail(f"File {filepath} contains '{key}' but not '{value}'.")

            for key, vals in self.ifelses.items():
                if key in content:
                    for value in vals:
                        if value not in content:
                            self.fail(f"File {filepath} does not contain '{key}' so it should have '{value}'.")

if __name__ == '__main__':
    unittest.main()