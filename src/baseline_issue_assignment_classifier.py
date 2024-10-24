from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import argparse

def preprocess_data(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Preprocess the DataFrame by combining the title and body of issues into a single feature set.

    Args:
        df (pd.DataFrame): DataFrame containing 'classical_preprocessed_title' and 
                           'classical_preprocessed_body' columns.

    Returns:
        tuple: A tuple containing the feature list (X) and the labels list (y) as lists.
    """
    X = df['classical_preprocessed_title'] + ' ' + df['classical_preprocessed_body']
    y = df['assignee']
    return X.tolist(), y.tolist()

def train_test_split_github_id(
    df: pd.DataFrame, 
    train_bound: tuple[int, int] = (0, 210_000), 
    test_bound: tuple[int, int] = (210_000, 220_000)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into training and testing sets based on GitHub issue IDs.

    Args:
        df (pd.DataFrame): DataFrame containing a 'github_id' column.
        train_bound (tuple, optional): Tuple specifying the lower and upper bounds for the training set.
                                       Defaults to (0, 210_000).
        test_bound (tuple, optional): Tuple specifying the lower and upper bounds for the test set.
                                      Defaults to (210_000, 220_000).

    Returns:
        tuple: Two DataFrames, one for training and one for testing.
    """
    if 'github_id' not in df.columns:
        raise ValueError("The DataFrame must contain a 'github_id' column.")
    
    train_df = df[(train_bound[0] < df["github_id"]) & (df["github_id"] <= train_bound[1])] 
    test_df = df[(test_bound[0] < df["github_id"]) & (df["github_id"] <= test_bound[1])]
    
    return train_df, test_df

def random_classifier(y_test: list[str]) -> np.ndarray:
    """
    Generate random predictions from unique classes in the test labels.

    Args:
        y_test (list): The true labels for the test data.

    Returns:
        np.ndarray: Array of random predictions from the unique classes.
    """
    unique_classes = np.unique(y_test)
    random_predictions = np.random.choice(unique_classes, size=len(y_test))
    return random_predictions

def weighted_random_classifier(y_train: pd.Series, y_test: list[str]) -> np.ndarray:
    """
    Generate weighted random predictions based on class frequencies in the training data.

    Args:
        y_train (pd.Series): The labels for the training data.
        y_test (list): The true labels for the test data.

    Returns:
        np.ndarray: Array of random predictions weighted by class frequencies in the training set.
    """
    class_weights = y_train.value_counts(normalize=True).to_dict()
    classes = list(class_weights.keys())
    weights = [class_weights[c] for c in classes]
    
    random_predictions = np.random.choice(classes, size=len(y_test), p=weights)
    return random_predictions


def main(verbose: bool = False) -> None:
    from preprocessing import get_preprocessed
    
    preprocess_issues_df = get_preprocessed("microsoft/vscode")
    train_data, test_data = train_test_split_github_id(preprocess_issues_df)

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
    y_pred_nb = nb_classifier.predict(X_test_tfidf)

    # Evaluate the Multinomial Naive Bayes Classifier
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f'\nNaive Bayes Classifier Accuracy: {accuracy_nb * 100:.2f}%\n')

    if verbose:
        print("Classification Report for Naive Bayes:")
        print(classification_report(y_test, y_pred_nb, zero_division=0))

    # Random Classifier
    y_pred_random = random_classifier(y_test)
    accuracy_random = accuracy_score(y_test, y_pred_random)
    print(f'Random Classifier Accuracy: {accuracy_random * 100:.2f}%\n')

    # Weighted Random Classifier
    y_pred_weighted_random = weighted_random_classifier(pd.Series(y_train), y_test)
    accuracy_weighted_random = accuracy_score(y_test, y_pred_weighted_random)
    print(f'Weighted Random Classifier Accuracy: {accuracy_weighted_random * 100:.2f}%\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline GitHub Issue Assignment Classifier")
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Print detailed classification reports and metrics"
    )
    args = parser.parse_args()

    main(verbose=args.verbose)
