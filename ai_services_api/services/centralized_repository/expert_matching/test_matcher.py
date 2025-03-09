import unittest
from ai_services_api.services.centralized_repository.expert_matching.matcher import Matcher
from ai_services_api.services.centralized_repository.database_setup import DatabaseInitializer, ExpertManager, ExpertResourceLinker


class TestMatcher(unittest.TestCase):

    def setUp(self):
        self.matcher = Matcher()

    def test_link_experts_to_resources(self):
        experts = [
            {'id': 1, 'name': 'John Doe'},
            {'id': 2, 'name': 'Jane Smith'}
        ]
        resources = [
            {'id': 1, 'authors': ['John Doe']},
            {'id': 2, 'authors': ['Jane Smith', 'Alice Johnson']}
        ]
        expected_links = {
            1: [1],
            2: [2]
        }
        actual_links = self.matcher.link_experts_to_resources(experts, resources)
        self.assertEqual(actual_links, expected_links)

    def test_no_matching_experts(self):
        experts = [
            {'id': 1, 'name': 'John Doe'},
            {'id': 2, 'name': 'Jane Smith'}
        ]
        resources = [
            {'id': 1, 'authors': ['Alice Johnson']},
            {'id': 2, 'authors': ['Bob Brown']}
        ]
        expected_links = {}
        actual_links = self.matcher.link_experts_to_resources(experts, resources)
        self.assertEqual(actual_links, expected_links)

if __name__ == '__main__':
    unittest.main()