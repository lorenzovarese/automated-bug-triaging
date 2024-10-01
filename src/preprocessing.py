import os
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
from pandarallel import pandarallel
from typing import Tuple, List

from pull_issues import pull_issues

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define stop words and stemmer globally to avoid reloading them in each function call
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

pandarallel.initialize(nb_workers=7, progress_bar=True)

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

    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\\u[\da-fA-F]{4}', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

def extract_code_snippets(text: str) -> Tuple[str, List[str]]:
    """
    Extract code snippets from the text and return the cleaned text without code snippets.

    Args:
        text (str): The input markdown text containing potential code snippets.

    Returns:
        Tuple[str, List[str]]:
            - str: The text with code snippets removed.
            - List[str]: A list of extracted code snippets.

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
    return text, code_snippets

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
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this line contains a pipe '|' and is not a code block line
        if '|' in line and not line.strip().startswith('```'):
            # Potential table start
            table_lines = [line]
            i += 1
            if i < len(lines):
                separator_line = lines[i]
                # Check if the next line is a table separator line
                if re.match(r'^\s*\|?\s*(\s*:?-+:?\s*\|)+\s*(:?-+:?\s*)?\s*$', separator_line):
                    table_lines.append(separator_line)
                    i += 1
                    # Collect table body lines
                    while i < len(lines) and '|' in lines[i] and not lines[i].strip().startswith('```'):
                        table_lines.append(lines[i])
                        i += 1
                    # Add the collected table lines
                    tables.append(table_lines)
                    continue  # Skip incrementing i at the end
                else:
                    # Not a table separator, add the line to non_table_lines
                    non_table_lines.append(line)
                    i -= 1  # Step back to reprocess the separator line
            else:
                # End of text, add the line to non_table_lines
                non_table_lines.append(line)
        else:
            # Not a table line, add to non_table_lines
            non_table_lines.append(line)
        i += 1
    # Reconstruct text without tables
    cleaned_text = '\n'.join(non_table_lines)
    # Parse tables into structured data
    parsed_tables = []
    for table_lines in tables:
        parsed_table = []
        for row_line in table_lines:
            # Skip separator lines
            if re.match(r'^\s*\|?\s*(\s*:?-+:?\s*\|)+\s*(:?-+:?\s*)?\s*$', row_line):
                continue
            # Split the row into cells
            row = [cell.strip() for cell in row_line.strip().strip('|').split('|')]
            parsed_table.append(row)
        parsed_tables.append(parsed_table)
    return parsed_tables, cleaned_text

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
    text, code_snippets = extract_code_snippets(text)
    # Extract tables
    tables, text = extract_tables(text)
    # Extract images
    images, text = extract_images(text)
    # Extract links
    links, text = extract_links(text)
    # Clean remaining text
    cleaned_text = clean_html_and_symbols(text)
    return code_snippets, images, links, tables, cleaned_text

def remove_infrequent_assignees(issues_df: pd.DataFrame, min_assignments: int = 30) -> pd.DataFrame:
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

def preprocess_issues(issues_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess issue titles and bodies by extracting markdown elements and applying text preprocessing.

    Args:
        issues_df (pd.DataFrame): DataFrame containing raw issue data including titles and bodies.

    Returns:
        pd.DataFrame: The DataFrame with additional columns:
            - 'classical_preprocessed_title': Preprocessed issue titles.
            - 'code_snippets': Extracted code snippets from the issue bodies.
            - 'images': Extracted images from the issue bodies.
            - 'links': Extracted links from the issue bodies.
            - 'tables': Extracted tables from the issue bodies.
            - 'cleaned_body': Issue body with markdown elements removed.
            - 'classical_preprocessed_body': Preprocessed version of the cleaned issue body.

    Raises:
        ValueError: If 'title' or 'body' columns are missing in the DataFrame.
    """
    required_columns = {'title', 'body'}
    missing_columns = required_columns - set(issues_df.columns)
    if missing_columns:
        raise ValueError(f"The DataFrame is missing required columns: {missing_columns}")

    print("\nPreprocessing issue titles in parallel...")
    issues_df['classical_preprocessed_title'] = issues_df['title'].parallel_apply(preprocess_text_classical)

    # Apply markdown extraction with progress bar
    print("\nExtracting markdown elements (code snippets, images, links, tables) from issue bodies in parallel...")

    # Create a helper function to return a tuple of extracted elements
    def extract_all_elements(body):
        return extract_markdown_elements(body)

    # Apply the extraction function in parallel and split the results into individual columns
    markdown_results = issues_df['body'].parallel_apply(extract_all_elements)

    # Split the tuple results into separate DataFrame columns
    issues_df['code_snippets'] = markdown_results.apply(lambda x: x[0])
    issues_df['images'] = markdown_results.apply(lambda x: x[1])
    issues_df['links'] = markdown_results.apply(lambda x: x[2])
    issues_df['tables'] = markdown_results.apply(lambda x: x[3])
    issues_df['cleaned_body'] = markdown_results.apply(lambda x: x[4])

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
        Tuple[pd.DataFrame, pd.DataFrame]:
            - pd.DataFrame: The training set DataFrame.
            - pd.DataFrame: The test set DataFrame.

    Raises:
        ValueError: If 'github_id' column is missing in the DataFrame.
    """
    if 'github_id' not in issues_df.columns:
        raise ValueError("The DataFrame must contain a 'github_id' column.")

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

    Raises:
        IOError: If the file cannot be written.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dataset_df.to_json(path, orient='records', indent=2)
    except IOError as e:
        raise IOError(f"Failed to save data to {path}: {e}")

def main() -> None:
    """
    Main function to execute the data preprocessing pipeline.

    Steps:
        1. Pull issues from the specified GitHub repository.
        2. Remove issues with infrequent assignees.
        3. Preprocess issues by extracting markdown elements and cleaning text.
        4. Split data into training and test sets.
        5. Save the datasets to JSON files.

    Raises:
        Exception: If any step in the pipeline fails.
    """
    try:
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
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        raise

if __name__ == '__main__':
    main()
