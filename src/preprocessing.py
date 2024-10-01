import os
import re
import nltk
import marko
from marko import block, inline
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd
from pandarallel import pandarallel
from typing import Tuple, List

from pull_issues import pull_issues

import multiprocessing


# Ensure NLTK resources are downloaded
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Define stop words and stemmer globally to avoid reloading them in each function call
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

pandarallel.initialize(nb_workers=min(100, multiprocessing.cpu_count()-1), progress_bar=True)

def clean_html_and_symbols(text: str) -> str:
    """
    Remove HTML tags and special symbol encodings from the input text.

    Args:
        text (str): The input string containing potential HTML and encoded symbols.

    Returns:
        str: The cleaned text with HTML tags and special symbols removed.
    """
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\\u[\da-fA-F]{4}', '', text) 
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) 
    return text

def extract_markdown_elements(text: str) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]], str]:
    """
    Extract code snippets, images, and links from the markdown text and return them along with the cleaned text.

    This function parses the markdown using `marko` and recursively traverses its elements to extract:
    - Code snippets: Fenced code blocks are extracted as individual code snippets.
    - Images: Images are extracted with their alt text and URLs.
    - Links: Links are extracted with their link text and destination URLs.
    - Cleaned body: The body text with markdown elements such as images, links, and code snippets removed.

    Args:
        text (str): The markdown content containing potential code snippets, images, and links.

    Returns:
        Tuple:
            - List[str]: A list of extracted code snippets.
            - List[Tuple[str, str]]: A list of tuples containing image alt text and URLs.
            - List[Tuple[str, str]]: A list of tuples containing link text and URLs.
            - str: The cleaned text with markdown elements removed.
    """
    if text is None:
        return [], [], [], ''

    # Parse the markdown text using marko
    parsed = marko.parse(text)
    code_snippets = []
    images = []
    links = []
    cleaned_text = []

    def traverse_elements(element):
        if isinstance(element, block.FencedCode):
            # Extract code content from the children
            code_content = ''
            for child in element.children:
                if isinstance(child, str):
                    code_content += child
                elif hasattr(child, 'children'):
                    code_content += child.children
            code_snippets.append(code_content.strip())
        elif isinstance(element, inline.Image):
            alt_text = element.title if element.title else 'No Alt Text'
            dest = getattr(element, 'dest', 'No URL')
            images.append((alt_text, dest))
        elif isinstance(element, inline.Link):
            link_text = ''.join(child.children for child in element.children if isinstance(child, inline.RawText))
            dest = getattr(element, 'dest', 'No URL')
            links.append((link_text, dest))
        elif isinstance(element, inline.RawText):
            cleaned_text.append(element.children)
        elif hasattr(element, 'children'):
            for child in element.children:
                traverse_elements(child)

    traverse_elements(parsed)
    cleaned_body = ''.join(cleaned_text)
    cleaned_body = clean_html_and_symbols(cleaned_body)  # Clean symbols
    return code_snippets, images, links, cleaned_body

def remove_infrequent_assignees(issues_df: pd.DataFrame, min_assignments: int = 30) -> pd.DataFrame:
    """
    Filter out issues that are assigned to developers with fewer than a specified number of assignments.

    Args:
        issues_df (pd.DataFrame): The DataFrame containing issue data.
        min_assignments (int): The minimum number of assignments required to retain an assignee.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only issues with assignees who meet the minimum assignment threshold.
    """
    issues_df_clean = issues_df.dropna(subset=['assignee'])
    issues_df_clean['assignee'] = issues_df_clean['assignee'].str.strip()
    assignee_counts = issues_df_clean['assignee'].value_counts()

    print(f"\nAssignee counts before filtering:\n{assignee_counts}")
    
    frequent_assignees = assignee_counts[assignee_counts >= min_assignments].index
    filtered_issues_df = issues_df_clean[issues_df_clean['assignee'].isin(frequent_assignees)]

    print(f"\nTotal issues before filtering: {issues_df.shape[0]}")
    print(f"Total issues after filtering: {filtered_issues_df.shape[0]}")

    assignee_counts_after = filtered_issues_df['assignee'].value_counts()
    print(f"\nAssignee counts after filtering:\n{assignee_counts_after}")

    return filtered_issues_df

def preprocess_text_classical(text: str) -> str:
    """
    Preprocess the given text by converting it to lowercase, removing punctuation,
    tokenizing, removing stop words, and applying stemming using classical NLP techniques.

    Args:
        text (str): The raw text to be preprocessed.

    Returns:
        str: The preprocessed text, cleaned and transformed for further analysis.
    """
    if pd.isna(text):
        return ""
    
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and non-word characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    # Rejoin tokens into a single string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def preprocess_issues(issues_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess issue titles and bodies by applying classical text preprocessing,
    and extracting code snippets, images, and links using `marko`, with a progress bar.

    Args:
        issues_df (pd.DataFrame): DataFrame containing raw issue data including titles and bodies.

    Returns:
        pd.DataFrame: The DataFrame with additional columns containing preprocessed titles, bodies, 
                      code snippets, images, and links extracted from the issue body.
    
    New Columns in the DataFrame:
        - 'classical_preprocessed_title': Preprocessed issue titles.
        - 'code_snippets': Extracted code snippets from the issue bodies.
        - 'images': Extracted images from the issue bodies.
        - 'links': Extracted links from the issue bodies.
        - 'cleaned_body': Issue body with code snippets, images, and links removed.
        - 'classical_preprocessed_body': Preprocessed version of the cleaned issue body.
    """

    print("\nPreprocessing issue titles in parallel...")
    issues_df['classical_preprocessed_title'] = issues_df['title'].parallel_apply(preprocess_text_classical)

    # Apply markdown extraction with progress bar
    print("\nExtracting markdown elements (code snippets, images, links) from issue bodies in parallel...")

    # Create a helper function to return a tuple of (code_snippets, images, links, cleaned_body)
    def extract_all_elements(body):
        return extract_markdown_elements(body)

    # Apply the extraction function in parallel and split the results into individual columns
    markdown_results = issues_df['body'].parallel_apply(extract_all_elements)

    # Split the tuple results into separate DataFrame columns
    issues_df['code_snippets'] = markdown_results.apply(lambda x: x[0])
    issues_df['images'] = markdown_results.apply(lambda x: x[1])
    issues_df['links'] = markdown_results.apply(lambda x: x[2])
    issues_df['cleaned_body'] = markdown_results.apply(lambda x: x[3])

    issues_df.drop(columns=['body'], inplace=True)

    print("\nPreprocessing cleaned issue bodies in parallel...")
    issues_df['classical_preprocessed_body'] = issues_df['cleaned_body'].parallel_apply(preprocess_text_classical)

    return issues_df

def split_data(issues_df: pd.DataFrame, train_range: Tuple[int, int], test_range: Tuple[int, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the issues DataFrame into training and test sets based on specified ranges of GitHub IDs.

    Args:
        issues_df (pd.DataFrame): The DataFrame containing the issues.
        train_range (Tuple[int, int]): The range of GitHub IDs for the training set.
        test_range (Tuple[int, int]): The range of GitHub IDs for the test set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test DataFrames.
    """
    train_start, train_end = train_range
    test_start, test_end = test_range

    train_set = issues_df[(issues_df['github_id'] >= train_start) & (issues_df['github_id'] <= train_end)]
    test_set = issues_df[(issues_df['github_id'] >= test_start) & (issues_df['github_id'] <= test_end)]
    return train_set, test_set

def save_data(dataset_df: pd.DataFrame, path: str) -> None:
    """
    Save the DataFrame as a JSON file at the specified path.

    Args:
        dataset_df (pd.DataFrame): The DataFrame to be saved.
        path (str): The file path where the JSON file will be saved.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset_df.to_json(path, orient='records', indent=2)

def main() -> None:
    issues_df = pull_issues("microsoft/vscode")
    
    assert isinstance(issues_df, pd.DataFrame), f"Expected 'issues_df' to be a DataFrame but got {type(issues_df)}"

    issues_df = remove_infrequent_assignees(issues_df, min_assignments=50)

    issues_df = preprocess_issues(issues_df)

    train_range = (1, 210000)
    test_range = (210001, 220000)
    train_set, test_set = split_data(issues_df, train_range, test_range)

    train_path = os.path.join('data', 'train', 'train_issues.json')
    test_path = os.path.join('data', 'test', 'test_issues.json')
    
    print(f"\nSaving training dataset to {train_path} with {train_set.shape[0]} issues")
    save_data(train_set, train_path)
    
    print(f"Saving test dataset to {test_path} with {test_set.shape[0]} issues")
    save_data(test_set, test_path)

if __name__ == '__main__':
    main()
