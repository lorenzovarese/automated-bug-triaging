import os
import json
import zipfile
from collections import Counter
import re
import nltk
import random
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords', quiet=True)

def unzip_issues_data(zip_path, extract_path):
    """Unzips the issues JSON file."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def read_issues(json_path):
    """Reads the issues from a JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        issues = json.load(f)
    return issues

def extract_relevant_fields(issues):
    """Extracts relevant fields from each issue."""
    extracted_issues = []
    for issue in issues:
        issue_id = issue.get('github_id')
        title = issue.get('title')
        body = issue.get('body')
        assignee = issue.get('assignee')
        extracted_issues.append({
            'github_id': issue_id,
            'title': title,
            'body': body,
            'assignee': assignee
        })
    return extracted_issues

def extract_code_snippets(text):
    """Extracts code snippets from the issue body and returns both the code snippets and the remaining text."""
    code_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    # Find all code blocks
    code_snippets = code_pattern.findall(text)
    # Remove the code blocks from the text
    cleaned_text = code_pattern.sub('', text)
    return code_snippets, cleaned_text

def extract_images_and_links(text):
    """Extracts markdown-style images and links from the text."""
    # Pattern for markdown images: ![alt_text](url)
    image_pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')
    
    # Pattern for markdown links: [text](url), but not starting with !
    link_pattern = re.compile(r'(?<!!)\[(.*?)\]\((.*?)\)')

    images = image_pattern.findall(text)
    links = link_pattern.findall(text)

    # Remove the images and links from the text to clean it up
    text_cleaned = image_pattern.sub('', text)
    text_cleaned = link_pattern.sub('', text_cleaned)

    # Format images and links into readable form
    images_list = [{'alt_text': alt, 'url': url} for alt, url in images]
    links_list = [{'text': text, 'url': url} for text, url in links]

    return images_list, links_list, text_cleaned

def filter_single_assignee(issues):
    """Keeps only issues with exactly one assignee."""
    filtered_issues = [issue for issue in issues if issue['assignee'] and isinstance(issue['assignee'], str)]
    return filtered_issues

def remove_infrequent_assignees(issues, min_assignments=5):
    """Removes issues assigned to infrequent assignees."""
    assignee_counts = Counter(issue['assignee'] for issue in issues)
    frequent_assignees = {assignee for assignee, count in assignee_counts.items() if count >= min_assignments}
    filtered_issues = [issue for issue in issues if issue['assignee'] in frequent_assignees]
    return filtered_issues

def split_identifiers(text):
    """Splits identifiers in camelCase and snake_case."""
    # Split camelCase
    text = re.sub('([a-z])([A-Z])', r'\1 \2', text)
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    return text

def preprocess_text(text):
    """Performs text preprocessing."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split identifiers
    text = split_identifiers(text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Rejoin tokens
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def preprocess_issues(issues):
    """Applies text preprocessing to issue titles and bodies, extracts code snippets, images, and links."""
    preprocessed_issues = []
    
    # Add progress bar with tqdm
    for issue in tqdm(issues, desc="Preprocessing Issues", unit="issue"):
        preprocessed_title = preprocess_text(issue['title'])
        
        # Extract code snippets, images, and links
        code_snippets, body_without_code = extract_code_snippets(issue['body'])
        images, links, body_without_code_or_images = extract_images_and_links(body_without_code)
        
        preprocessed_body = preprocess_text(body_without_code_or_images)
        
        preprocessed_issue = {
            'github_id': issue['github_id'],
            'title': preprocessed_title,
            'body': preprocessed_body,
            'assignee': issue['assignee'],
            'code_snippets': code_snippets,  
            'images': images,                
            'links': links                   
        }
        preprocessed_issues.append(preprocessed_issue)
    return preprocessed_issues

def split_data(issues, train_ratio=0.8):
    """Splits data into training and test sets."""
    random.shuffle(issues)
    split_index = int(len(issues) * train_ratio)
    train_set = issues[:split_index]
    test_set = issues[split_index:]
    return train_set, test_set

def save_data(dataset, path):
    """Saves the dataset to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

def main():
    # Paths
    res_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'res'))
    zip_path = os.path.join(res_folder, 'issues.json.zip')
    extract_path = os.path.join(res_folder)
    json_path = os.path.join(extract_path, 'issues.json')

    # Unzip issues data
    unzip_issues_data(zip_path, extract_path)

    # Read issues
    issues = read_issues(json_path)

    # Extract relevant fields
    issues = extract_relevant_fields(issues)

    # Filter issues with exactly one assignee
    issues = filter_single_assignee(issues)

    # Remove infrequent assignees (developers with less than 5 assignments)
    issues = remove_infrequent_assignees(issues, min_assignments=5)

    # Preprocess text in issues
    issues = preprocess_issues(issues)

    # Split data into training and test sets
    train_set, test_set = split_data(issues, train_ratio=0.8) # TODO(lorenzovarese): fix the number of issues

    # Create folder structure and save datasets
    train_path = os.path.join('data', 'train', 'train_issues.json')
    test_path = os.path.join('data', 'test', 'test_issues.json')
    save_data(train_set, train_path)
    save_data(test_set, test_path)

if __name__ == '__main__':
    main()
