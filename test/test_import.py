#!/usr/bin/env python3
"""
Test basic package import functionality
"""

import unittest


class TestImport(unittest.TestCase):
    """Test basic import functionality of the ds_policy package"""

    def test_package_import(self):
        """Test importing main package components"""
        try:
            import ds_policy
            from ds_policy import DSPolicy, NeuralODE
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import ds_policy package: {e}")

    def test_version(self):
        """Test version attribute existence"""
        import ds_policy
        self.assertTrue(hasattr(ds_policy, '__version__'))
        self.assertIsInstance(ds_policy.__version__, str)


if __name__ == '__main__':
    unittest.main() 