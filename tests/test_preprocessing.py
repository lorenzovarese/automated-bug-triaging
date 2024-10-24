import unittest
import pandas as pd
from src.preprocessing import (
    clean_html_and_symbols,
    extract_code_snippets,
    extract_images,
    extract_links,
    extract_tables,
    extract_markdown_elements,
    preprocess_text_classical,
    remove_infrequent_assignees,
)

class TestPreprocessingFunctions(unittest.TestCase):

    def test_clean_html_and_symbols(self):
        input_text = "<p>This is a test.</p> \u1234 \uabcd"
        expected_output = "This is a test.    "
        result = clean_html_and_symbols(input_text)
        self.assertEqual(result, expected_output)

    def test_extract_code_snippets(self):
        input_text = (
            "Here is some code:\n"
            "```python\n"
            "def hello():\n"
            "    print('Hello World')\n"
            "```\n"
            "End of code."
        )
        expected_cleaned_text = "Here is some code:\n\nEnd of code."
        expected_code_snippets = ["def hello():\n    print('Hello World')"]
        code_snippets, cleaned_text = extract_code_snippets(input_text)
        self.assertEqual(cleaned_text, expected_cleaned_text)
        self.assertEqual(code_snippets, expected_code_snippets)

    def test_extract_images(self):
        input_text = "This is an image: ![Alt Text](http://example.com/image.png)"
        expected_cleaned_text = "This is an image: "
        expected_images = [("Alt Text", "http://example.com/image.png")]
        images, cleaned_text = extract_images(input_text)
        self.assertEqual(cleaned_text, expected_cleaned_text)
        self.assertEqual(images, expected_images)

    def test_extract_links(self):
        input_text = "This is a link: [Example](http://example.com)"
        expected_cleaned_text = "This is a link: Example"
        expected_links = [("Example", "http://example.com")]
        links, cleaned_text = extract_links(input_text)
        self.assertEqual(cleaned_text, expected_cleaned_text)
        self.assertEqual(links, expected_links)

    def test_extract_tables(self):
        input_text = (
            "Here is a table:\n\n"
            "| Header1 | Header2 |\n"
            "|---------|---------|\n"
            "| Cell1   | Cell2   |\n"
            "| Cell3   | Cell4   |\n"
            "End of table."
        )
        expected_cleaned_text = "Here is a table:\n\nEnd of table."
        expected_tables = [
            [
                ["Header1", "Header2"],
                ["Cell1", "Cell2"],
                ["Cell3", "Cell4"]
            ]
        ]
        tables, cleaned_text = extract_tables(input_text)
        self.assertEqual(cleaned_text, expected_cleaned_text)
        self.assertEqual(tables, expected_tables)

    def test_extract_markdown_elements(self):
        input_text = (
            "Sample text with markdown elements.\n\n"
            "![Image Alt](http://example.com/image.png)\n"
            "[Link Text](http://example.com)\n"
            "```java\n"
            "public class HelloWorld {\n"
            "    public static void main(String[] args) {\n"
            "        System.out.println(\"Hello, World\");\n"
            "    }\n"
            "}\n"
            "```\n"
            "End of markdown."
        )
        expected_cleaned_text = (
            "Sample text with markdown elements.\n\n\nLink Text\n\nEnd of markdown."
        )

        expected_code_snippets = [
            "public class HelloWorld {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World\");\n    }\n}"
        ]
        expected_images = [("Image Alt", "http://example.com/image.png")]
        expected_links = [("Link Text", "http://example.com")]
        code_snippets, images, links, tables, cleaned_text = extract_markdown_elements(input_text)
        self.assertEqual(cleaned_text, expected_cleaned_text)
        self.assertEqual(code_snippets, expected_code_snippets)
        self.assertEqual(images, expected_images)
        self.assertEqual(links, expected_links)

    def test_preprocess_text_classical(self):
        input_text = "This is a Sample TEXT, with punctuation! And stopwords."
        expected_output = "sampl text punctuat stopword"
        result = preprocess_text_classical(input_text)
        self.assertEqual(result, expected_output)

class TestDataFrameFunctions(unittest.TestCase):

    def test_remove_infrequent_assignees(self):
        data = {
            'assignee': ['alice', 'bob', 'alice', 'charlie', 'bob', 'bob', 'charlie', 'dave', 'dave', 'dave'],
            'issue_id': range(1, 11)
        }
        df = pd.DataFrame(data)
        min_assignments = 3
        filtered_df = remove_infrequent_assignees(df, min_assignments)
        expected_assignees = ['bob', 'dave']
        self.assertEqual(sorted(filtered_df['assignee'].unique()), sorted(expected_assignees))
        self.assertEqual(len(filtered_df), 6)

if __name__ == '__main__':
    unittest.main()