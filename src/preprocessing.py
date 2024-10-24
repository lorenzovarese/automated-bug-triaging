import os
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from pandarallel import pandarallel
from typing import Tuple, List

import multiprocessing

PREPROCESSED_FILE = "data/issuesprep.json.zip"

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define stop words and stemmer globally to avoid reloading them in each function call
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

pandarallel.initialize(nb_workers=min(100, multiprocessing.cpu_count()-1), progress_bar=True)

def clean_html_and_symbols(text: str) -> str:
    """
    Remove HTML tags and special symbol encodings from the input text.

    Args:
        text (str): The input string containing potential HTML tags and encoded symbols.

    Returns:
        str: The cleaned text with HTML tags and special symbols removed.

    Raises:
        TypeError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    text = re.sub(r'<.*?>', '', text) # Removes HTML tags from the text.
    text = re.sub(r'\\u[\da-fA-F]{4}', '', text) # Removes Unicode escape sequences represented as raw string literals.
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Replaces non-ASCII characters with a space.
    return text

def extract_code_snippets(text: str) -> Tuple[List[str], str]:
    """
    Extract code snippets from the text and return the cleaned text without code snippets.

    Args:
        text (str): The input markdown text containing potential code snippets.

    Returns:
        Tuple[List[str], str]:
            - List[str]: A list of extracted code snippets.
            - str: The text with code snippets removed.

    Raises:
        TypeError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    code_snippets = []
    code_block_pattern = re.compile(r'```.*?\n(.*?)\n```', re.DOTALL)

    def code_replacer(match):
        code = match.group(1)
        code_snippets.append(code.strip())
        return ''

    text = code_block_pattern.sub(code_replacer, text)
    return code_snippets, text

def extract_images(text: str) -> Tuple[List[Tuple[str, str]], str]:
    """
    Extract images from the text and return the cleaned text without images.

    Args:
        text (str): The input markdown text containing potential images.

    Returns:
        Tuple[List[Tuple[str, str]], str]:
            - List[Tuple[str, str]]: A list of tuples containing image alt text and URLs.
            - str: The text with images removed.

    Raises:
        TypeError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    images = []
    image_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')

    def image_replacer(match):
        alt_text = match.group(1)
        url = match.group(2)
        images.append((alt_text, url))
        return ''

    text = image_pattern.sub(image_replacer, text)
    return images, text

def extract_links(text: str) -> Tuple[List[Tuple[str, str]], str]:
    """
    Extract links from the text and return the cleaned text without links.

    Args:
        text (str): The input markdown text containing potential links.

    Returns:
        Tuple[List[Tuple[str, str]], str]:
            - List[Tuple[str, str]]: A list of tuples containing link text and URLs.
            - str: The text with links replaced by their link text.

    Raises:
        TypeError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    links = []
    link_pattern = re.compile(r'\[(.*?)\]\((.*?)\)')

    def link_replacer(match):
        link_text = match.group(1)
        url = match.group(2)
        links.append((link_text, url))
        return link_text  # Keep the link text in the cleaned text

    text = link_pattern.sub(link_replacer, text)
    return links, text

def extract_tables(text: str) -> Tuple[List[List[List[str]]], str]:
    """
    Detect and extract tables from markdown text and return them along with the cleaned text.

    Args:
        text (str): The input markdown text containing potential tables.

    Returns:
        Tuple[List[List[List[str]]], str]:
            - List[List[List[str]]]: A list of tables, each table is a list of rows, each row is a list of cell strings.
            - str: The text with tables removed.

    Raises:
        TypeError: If the input text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    lines = text.split('\n')
    tables = []
    non_table_lines = []
    current_table = []
    inside_table = False

    for line in lines:
        # Skip code block markers
        if line.strip().startswith('```'):
            non_table_lines.append(line)
            continue

        # Detect potential table rows (contain pipes '|' and are not separator lines)
        if '|' in line:
            if re.match(r'^\s*\|?\s*(:?-+:?\s*\|)+\s*(:?-+:?\s*)?\s*$', line.strip()):
                # Skip separator lines, do not include in cleaned text
                inside_table = True
                continue
            else:
                # If a valid table row (not a separator), collect it
                current_table.append([cell.strip() for cell in line.strip().strip('|').split('|')])
                inside_table = True
        else:
            # If not inside a table, just add to non-table lines
            if inside_table:
                # End of table detected, process the current table
                if current_table:
                    tables.append(current_table)
                    current_table = []
                inside_table = False
            non_table_lines.append(line)

    # Handle case where text ends with a table
    if current_table:
        tables.append(current_table)

    # Reconstruct the cleaned text without the tables and separator lines
    cleaned_text = '\n'.join(non_table_lines)

    return tables, cleaned_text

def extract_markdown_elements(text: str) -> Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]], List[List[List[str]]], str]:
    """
    Extract code snippets, images, links, and tables from the markdown text and return them along with the cleaned text.

    Args:
        text (str): The input markdown text containing various markdown elements.

    Returns:
        Tuple[List[str], List[Tuple[str, str]], List[Tuple[str, str]], List[List[List[str]]], str]:
            - List[str]: A list of extracted code snippets.
            - List[Tuple[str, str]]: A list of tuples containing image alt text and URLs.
            - List[Tuple[str, str]]: A list of tuples containing link text and URLs.
            - List[List[List[str]]]: A list of parsed tables.
            - str: The cleaned text with markdown elements removed.

    Raises:
        TypeError: If the input text is not a string.
    """
    if text is None:
        return [], [], [], [], ''
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    # Extract code snippets
    code_snippets, text = extract_code_snippets(text)
    # Extract tables
    tables, text = extract_tables(text)
    # Extract images
    images, text = extract_images(text)
    # Extract links
    links, text = extract_links(text)
    # Clean remaining text
    cleaned_text = clean_html_and_symbols(text)
    return code_snippets, images, links, tables, cleaned_text

def remove_infrequent_assignees(issues_df: pd.DataFrame, min_assignments: int = 30, verbose: bool = False) -> pd.DataFrame:
    """
    Filter out issues assigned to developers with fewer than a specified number of assignments.

    Args:
        issues_df (pd.DataFrame): The DataFrame containing issue data.
        min_assignments (int, optional): The minimum number of assignments required to retain an assignee. Defaults to 30.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only issues with assignees who meet the minimum assignment threshold.

    Raises:
        ValueError: If 'assignee' column is missing in the DataFrame.
    """
    if 'assignee' not in issues_df.columns:
        raise ValueError("The DataFrame must contain an 'assignee' column.")

    issues_df_clean = issues_df.dropna(subset=['assignee'])
    issues_df_clean['assignee'] = issues_df_clean['assignee'].str.strip()
    assignee_counts = issues_df_clean['assignee'].value_counts()

    if verbose:
        print(f"\nAssignee counts before filtering:\n{assignee_counts}")

    frequent_assignees = assignee_counts[assignee_counts >= min_assignments].index
    filtered_issues_df = issues_df_clean[issues_df_clean['assignee'].isin(frequent_assignees)]

    if verbose:
        print(f"\nTotal issues before filtering: {issues_df.shape[0]}")
        print(f"Total issues after filtering: {filtered_issues_df.shape[0]}")

    assignee_counts_after = filtered_issues_df['assignee'].value_counts()
    if verbose:
        print(f"\nAssignee counts after filtering:\n{assignee_counts_after}")

    return filtered_issues_df

def preprocess_text_classical(text: str) -> str:
    """
    Preprocess text by lowercasing, removing punctuation, tokenizing, removing stop words, and stemming.

    Args:
        text (str): The raw text to be preprocessed.

    Returns:
        str: The preprocessed text.

    Raises:
        TypeError: If the input text is not a string.
    """
    if pd.isna(text):
        return ""
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

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

def get_preprocessed(github_repo: str, min_assignments: int = 50, force_processing: bool = False) -> pd.DataFrame:
    """
    Retrieve and preprocess issues from a specified GitHub repository.

    This function pulls closed issues from the provided GitHub repository,
    processes the titles and bodies of the issues to clean and extract relevant
    markdown elements, and removes infrequent assignees.

    Args:
        github_repo (str): The GitHub repository in the format "owner/repo".
        min_assignments (int, optional): The minimum number of assignments required 
                                          for an assignee to be retained. Defaults to 50.
        force_processing (bool, optional): If True, forces the function to pull new
                                            data from GitHub even if cached data exists.
                                            Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the processed issues
    """
    from pull_issues import pull_issues
    
    if not force_processing and os.path.exists(PREPROCESSED_FILE):
        df = pd.read_json(PREPROCESSED_FILE)
        return df
    
    issues_df: pd.DataFrame = pull_issues(github_repo)
    issues_df: pd.DataFrame = remove_infrequent_assignees(issues_df, min_assignments=min_assignments)
    issues_df['classical_preprocessed_title'] = issues_df['title'].parallel_apply(preprocess_text_classical)

    markdown_results = issues_df['body'].parallel_apply(extract_markdown_elements)
    issues_df[['code_snippets', 'images', 'links', 'tables', 'cleaned_body']] = pd.DataFrame(markdown_results.tolist(), index=issues_df.index)
    issues_df['text'] = issues_df['title'] + " " + issues_df['cleaned_body']
    issues_df['classical_preprocessed_body'] = issues_df['cleaned_body'].parallel_apply(preprocess_text_classical)
    
    issues_df.to_json(PREPROCESSED_FILE, orient="records")

    return issues_df

if __name__ == "__main__":
    # Example of usage
    df = get_preprocessed("microsoft/vscode")
    print(df.head())
    print("Number of issues processed: ", df.shape[0])

    df_train = df[df["github_id"] <= 210_000]
    df_recent = df[(190_000 <= df["github_id"]) & (df["github_id"] <= 210_000)]
    df_test = df[(210_000 < df["github_id"]) & (df["github_id"] <= 220_000)]

    print(f"Train size (id <= 210'000): {df_train.shape[0]}")
    print(f"Recent size (190'000 <= id <= 210'000): {df_recent.shape[0]}")
    print(f"Test size (210'000 < id <= 220'000): {df_test.shape[0]}")
    
