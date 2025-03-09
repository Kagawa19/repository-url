import unittest
from ai_services_api.services.centralized_repository.expert_matching.matcher import Normalizer

class TestNormalizer(unittest.TestCase):

    def setUp(self):
        self.normalizer = Normalizer()

    def test_normalize_name(self):
        self.assertEqual(self.normalizer._normalize_name("John Doe"), "john doe")
        self.assertEqual(self.normalizer._normalize_name("  Jane   Smith  "), "jane smith")
        self.assertEqual(self.normalizer._normalize_name("Alice"), "alice")

    def test_get_name_parts(self):
        self.assertEqual(self.normalizer._get_name_parts("John Doe"), ("John", "Doe"))
        self.assertEqual(self.normalizer._get_name_parts("Alice"), ("", "Alice"))
        self.assertEqual(self.normalizer._get_name_parts("  Bob  Marley "), ("Bob", "Marley"))

if __name__ == '__main__':
    unittest.main()