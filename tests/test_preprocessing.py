import unittest
import pandas as pd
from src.preprocessing import (
    clean_html_and_symbols,
    extract_markdown_elements,
    remove_infrequent_assignees,
    preprocess_text_classical,
    preprocess_issues,
    split_data
)

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Sample data to be used in the tests
        self.sample_data = [
            {"github_id": 1, "title": "Issue 1", "body": "This is a body with code\n```python\nprint('hello')\n```", "assignee": "user1"},
            {"github_id": 2, "title": "Issue 2", "body": "Body with image ![image](https://example.com/image.png)", "assignee": "user2"},
            {"github_id": 3, "title": "Issue 3", "body": "Body with link [link](https://example.com)", "assignee": "user1"},
            {"github_id": 4, "title": "Issue 4", "body": "No code or images.", "assignee": None}
        ]
        self.issues_df = pd.DataFrame(self.sample_data)

    def test_clean_html_and_symbols(self):
        # Test for removing HTML and encoded symbols
        text = "<p>This is a test &nbsp; with special symbols \u2022</p>"
        cleaned_text = clean_html_and_symbols(text)
        self.assertEqual(cleaned_text, "This is a test   with special symbols ")

    def test_extract_markdown_elements(self):
        # Test markdown element extraction
        text = "Here is some code: ```python\nprint('hello')\n``` and an image ![image](https://example.com/image.png)"
        code_snippets, images, links, cleaned_body = extract_markdown_elements(text)
        self.assertEqual(len(code_snippets), 1)
        self.assertIn("print('hello')", code_snippets[0])
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0], ('No Alt Text', 'https://example.com/image.png'))
        self.assertEqual(cleaned_body.strip(), "Here is some code:  and an image ")

    def test_remove_infrequent_assignees(self):
        # Test for removing infrequent assignees
        filtered_df = remove_infrequent_assignees(self.issues_df, min_assignments=2)
        self.assertEqual(filtered_df.shape[0], 2)  # Only "user1" appears twice

    def test_preprocess_text_classical(self):
        # Test text preprocessing
        raw_text = "This is a sample TEXT, with punctuation!"
        processed_text = preprocess_text_classical(raw_text)
        self.assertEqual(processed_text, "sampl text punctuat")

    def test_preprocess_issues(self):
        # Test the entire preprocessing workflow on issues
        preprocessed_df = preprocess_issues(self.issues_df)
        self.assertIn('classical_preprocessed_title', preprocessed_df.columns)
        self.assertIn('code_snippets', preprocessed_df.columns)
        self.assertIn('images', preprocessed_df.columns)
        self.assertIn('cleaned_body', preprocessed_df.columns)
        self.assertEqual(preprocessed_df.shape[0], 4)  # Ensure all rows are kept

    def test_split_data(self):
        # Test the data split function based on ranges
        train_set, test_set = split_data(self.issues_df, (1, 2), (3, 4))
        self.assertEqual(train_set.shape[0], 2)  # IDs 1 and 2 should be in train set
        self.assertEqual(test_set.shape[0], 2)   # IDs 3 and 4 should be in test set

if __name__ == '__main__':
    unittest.main()
