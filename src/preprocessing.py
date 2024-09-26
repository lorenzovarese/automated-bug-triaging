import os
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
import pandas as pd
from pandarallel import pandarallel
from typing import Tuple, List
from pull_issues import pull_issues

# Ensure NLTK resources are downloaded
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Define stop words and stemmer globally to avoid reloading them in each function call
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Initialize pandarallel with all available cores
pandarallel.initialize(progress_bar=True)

def extract_code_snippets(text: str) -> Tuple[List[str], str]:
    """Extracts code snippets from the issue body and returns both the code snippets and the remaining text."""
    if text is None:
        return [], ''
    code_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    # Find all code blocks
    code_snippets = code_pattern.findall(text)
    # Remove the code blocks from the text
    cleaned_text = code_pattern.sub('', text)
    return code_snippets, cleaned_text

def extract_images_and_links(text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]:
    """Extracts markdown-style images and links from the text."""
    if text is None:
        return [], [], ''
    # Pattern for markdown images: ![alt_text](url)
    image_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    
    # Pattern for markdown links: [text](url), but not starting with !
    link_pattern = re.compile(r'(?<!!)\[(.*?)\]\((.*?)\)')
    
    images = image_pattern.findall(text)
    links = link_pattern.findall(text)
    
    # Remove the images and links from the text to clean it up
    text_cleaned = image_pattern.sub('', text)
    text_cleaned = link_pattern.sub('', text_cleaned)
    
    return images, links, text_cleaned

def remove_infrequent_assignees(issues_df: pd.DataFrame, min_assignments: int = 5) -> pd.DataFrame:
    """Removes issues assigned to infrequent assignees."""
    assignee_counts = issues_df['assignee'].value_counts()
    frequent_assignees = assignee_counts[assignee_counts >= min_assignments].index
    return issues_df[issues_df['assignee'].isin(frequent_assignees)]

def split_identifiers(text: str) -> str:
    """Splits identifiers in camelCase and snake_case."""
    # Split camelCase (e.g., "camelCase" -> "camel Case")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Replace underscores with spaces (e.g., "snake_case" -> "snake case")
    text = text.replace('_', ' ')
    return text

def preprocess_text(text: str) -> str:
    """Performs text preprocessing."""
    if pd.isna(text):
        return ""
    
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and non-word characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split identifiers
    text = split_identifiers(text)
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
    """Applies text preprocessing to issue titles and bodies, extracts code snippets, images, and links."""
    
    print("\nPreprocessing issue titles in parallel...")
    issues_df['preprocessed_title'] = issues_df['title'].parallel_apply(preprocess_text)
    
    print("\nExtracting code snippets from issue bodies...")
    issues_df['code_snippets'], issues_df['body_without_code'] = zip(*issues_df['body'].apply(extract_code_snippets))
    
    print("\nExtracting images and links from issue bodies...")
    issues_df['images'], issues_df['links'], issues_df['preprocessed_body'] = zip(*issues_df['body_without_code'].apply(extract_images_and_links))
    
    print("\nPreprocessing issue bodies in parallel...")
    issues_df['preprocessed_body'] = issues_df['preprocessed_body'].parallel_apply(preprocess_text)
    
    return issues_df

def split_data(issues_df: pd.DataFrame, train_range: Tuple[int, int], test_range: Tuple[int, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits DataFrame into training and test sets based on specified issue number ranges."""
    # Extract the start and end points for the training and testing ranges
    train_start, train_end = train_range
    test_start, test_end = test_range

    # Filter the DataFrame based on the specified ranges
    train_set = issues_df[(issues_df['github_id'] >= train_start) & (issues_df['github_id'] <= train_end)]
    test_set = issues_df[(issues_df['github_id'] >= test_start) & (issues_df['github_id'] <= test_end)]
    return train_set, test_set

def save_data(dataset_df: pd.DataFrame, path: str) -> None:
    """Saves the DataFrame to a JSON file."""
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset_df.to_json(path, orient='records', indent=2)

def main() -> None:
    """Main function to execute the pipeline."""
    
    # Read issues
    issues_df = pull_issues("microsoft/vscode")
    
    assert isinstance(issues_df, pd.DataFrame), f"Expected 'issues_df' to be a DataFrame but got {type(issues_df)}"

    print(f"\nTotal issues pulled: {issues_df.shape[0]}")

    # Remove infrequent assignees (developers with less than 5 assignments)
    issues_df = remove_infrequent_assignees(issues_df, min_assignments=5)

    # Preprocess text in issues
    issues_df = preprocess_issues(issues_df)

    # Split data into training and test sets
    train_range = (1, 210000)
    test_range = (210001, 220000)
    train_set, test_set = split_data(issues_df, train_range, test_range)

    # Create folder structure and save datasets
    train_path = os.path.join('data', 'train', 'train_issues.json')
    test_path = os.path.join('data', 'test', 'test_issues.json')
    
    print(f"\nSaving training dataset to {train_path} with {train_set.shape[0]} issues")
    save_data(train_set, train_path)
    
    print(f"\nSaving test dataset to {test_path} with {test_set.shape[0]} issues")
    save_data(test_set, test_path)

if __name__ == '__main__':
    main()
