import json
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    """
    Load JSON data from the given filepath.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def preprocess_data(data):
    """
    Preprocess the data by concatenating title and classical_preprocessed_body,
    and extracting the assignee as the target variable.

    Args:
        data (list): List of issue data loaded from the JSON file.

    Returns:
        tuple: A tuple containing the processed text (X) and the target labels (y).
    """
    X = [issue['classical_preprocessed_title'] + ' ' + issue['classical_preprocessed_body'] for issue in data]
    y = [issue['assignee'] for issue in data]
    return X, y

def main(train_filepath, test_filepath):
    """
    Main function to load data, preprocess it, train the model, and evaluate it.

    Args:
        train_filepath (str): Filepath for the training dataset.
        test_filepath (str): Filepath for the test dataset.
    """
    # Load train and test datasets
    train_data = load_data(train_filepath)
    test_data = load_data(test_filepath)

    # Preprocess the data
    X_train, y_train = preprocess_data(train_data)
    X_test, y_test = preprocess_data(test_data)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train Multinomial Naive Bayes Classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)

    # Make predictions on test data
    y_pred = nb_classifier.predict(X_test_tfidf)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Overall Model Accuracy: {accuracy * 100:.2f}%\n')

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    # Ensure correct number of arguments are passed
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <train_filepath> <test_filepath>")
        sys.exit(1)

    # File paths are passed as arguments from the command line
    train_filepath = sys.argv[1]
    test_filepath = sys.argv[2]

    # Run the main function
    main(train_filepath, test_filepath)
