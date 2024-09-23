import unittest
from src.preprocessing import (
    extract_relevant_fields,
    extract_code_snippets,
    extract_images_and_links,
    preprocess_text,
    filter_single_assignee,
    remove_infrequent_assignees
)

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Sample data for tests
        self.sample_issues = [
            {
                "github_id": 1,
                "title": "Sample Issue 1",
                "body": "This is a sample body. \n```python\nprint('Hello World')\n```",
                "assignee": "user1"
            },
            {
                "github_id": 2,
                "title": "Sample Issue 2",
                "body": "Another body with an image ![image](https://example.com/image.png) and a link [example](https://example.com).",
                "assignee": "user2"
            },
            {
                "github_id": 3,
                "title": "Sample Issue 3",
                "body": "Body without code or images.",
                "assignee": None
            },
            {
                "github_id": 4,
                "title": "Sample Issue 4",
                "body": "Text with a link but no image [example](https://example.com).",
                "assignee": "user1"
            }
        ]

    def test_extract_relevant_fields(self):
        extracted = extract_relevant_fields(self.sample_issues)
        self.assertEqual(len(extracted), 4)
        self.assertEqual(extracted[0]['github_id'], 1)
        self.assertIn('title', extracted[0])
        self.assertIn('body', extracted[0])
        self.assertIn('assignee', extracted[0])

    def test_extract_code_snippets(self):
        code_snippets, cleaned_text = extract_code_snippets(self.sample_issues[0]['body'])
        self.assertEqual(len(code_snippets), 1)
        self.assertIn("print('Hello World')", code_snippets[0])
        self.assertNotIn('```', cleaned_text)

    def test_extract_images_and_links(self):
        images, links, cleaned_text = extract_images_and_links(self.sample_issues[1]['body'])
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]['url'], "https://example.com/image.png")
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]['url'], "https://example.com")
        self.assertNotIn('![image]', cleaned_text)
        self.assertNotIn('[example]', cleaned_text)

    def test_preprocess_text(self):
        processed = preprocess_text("This is a Sample text.")
        self.assertEqual(processed, "sampl text")

    def test_filter_single_assignee(self):
        filtered = filter_single_assignee(self.sample_issues)
        self.assertEqual(len(filtered), 3)  # Issue 3 has no assignee, so should be excluded

    def test_remove_infrequent_assignees(self):
        filtered = remove_infrequent_assignees(self.sample_issues, min_assignments=2)
        self.assertEqual(len(filtered), 2)  # "user1" has 2 assignments, others have fewer

if __name__ == '__main__':
    unittest.main()
