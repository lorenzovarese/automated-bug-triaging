# Automated Bug Triaging

A machine learning tool for automating the assignment of open issues to the best-suited developers in the VSCode GitHub
repository.

## Group Information

- **Group Number**: 3
- **Participants**:
    - Fauconnet Arnaud
    - Perozzi Vittorio
    - Varese Lorenzo

## Features

- **Automated Assignment**: Automatically assigns open issues to the best-suited developers based on historical data.
- **Ranked Candidate List**: Provides a ranked list of potential assignees, displaying the most likely candidate at the
  top.
- **Contributor Statistics**: Shows the number of commits each candidate has authored in the VSCode repository.
- **Training on Historical Data**: Trains the model using closed issues with exactly one assignee and issue ID ≤ 210000.
- **Evaluation**: Evaluates the model on a test set consisting of closed issues with IDs from 210001 to 220000.
- **Command-Line Interface**: Simple shell-based interface for ease of use.

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **GitHub Personal Access Token** with permissions to read issues from the target repository
- **CUDA-compatible GPU (optional but recommended)**

### Set Up a Virtual Environment (Optional but Recommended)

It is recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate
```

**Disclaimer**: The instructions above are intended for Linux users.

### Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Set the `GITHUB_AUTH_TOKEN` environment variable with your GitHub personal access token:

```bash
export GITHUB_AUTH_TOKEN=<your_token_here>
```

**Note**: You can generate a token by navigating to your GitHub account settings under **Developer settings > Personal
access tokens**.

## How to Run the Tool

The tool consists of several scripts that should be run sequentially:

1. [Pull Issues from GitHub](#step-1-pull-issues-from-github)
2. [Preprocess the Issues Data](#step-2-preprocess-the-issues-data)
3. [Encode the Data](#step-3-encode-the-data)
4. [Train the Model](#step-4-train-the-model)

### Step 1: Pull Issues from GitHub

Run the `pull_issues.py` script to pull issues from the target GitHub repository. By default, it pulls issues from the
`microsoft/vscode` repository.

```bash
python src/pull_issues.py
```

This script will:

- Connect to the GitHub API using your personal access token.
- Fetch closed issues from the specified repository.
- Filter issues to include only those with exactly one assignee.
- Save the issues data to `res/issues.json.zip`.

If you wish to pull issues from a different repository, you can modify the `github_repo` parameter in the
`pull_issues.py` script.

### Step 2: Preprocess the Issues Data

Run the `preprocessing.py` script to preprocess the issues data. This script cleans and processes the text data,
extracts useful elements, and prepares the data for encoding.

```bash
python src/preprocessing.py
```

This script will:

- Load the issues data from `res/issues.json.zip`.
- Remove issues assigned to developers with fewer than a specified number of assignments (default is 50).
- Preprocess the issue titles and bodies:
    - Extract code snippets, images, links, and tables from markdown content.
    - Clean the text by removing HTML tags and special symbols.
    - Apply classical text preprocessing techniques (lowercasing, removing punctuation, stop words, and stemming).
- Save the preprocessed data to `data/issues.json`.

### Step 3: Encode the Data

Run the `encode_data.py` script to tokenize and encode the preprocessed data using the BERT tokenizer.

```bash
python src/encode_data.py
```

This script will:

- Load the preprocessed data from `data/issues.json`.
- Tokenize and encode the text data (titles and bodies) using the BERT tokenizer.
- Split the data into training, evaluation, and test sets.
- Save the encoded datasets to `data/encoded_data`.

**Note**: If the encoded data already exists, the script will load it from the disk unless you set the `force` parameter
to `True` in the script.

### Step 4: Train the Model

Run the `train.py` script to train the BERT-based sequence classification model.

```bash
python src/train.py
```

This script will:

- Load the encoded datasets from `data/encoded_data`.
- Initialize the BERT model for sequence classification.
- Train the model on the training dataset.
- Evaluate the model on the evaluation dataset.
- Save the trained model checkpoints to `data/checkpoints`.
- Evaluate the model on the test dataset and report accuracy.

**Notes**:

- The script is configured to use GPU (if available). Ensure you have a compatible GPU and CUDA installed.
- Adjust the `CUDA_VISIBLE_DEVICES` environment variable in `train.py` if necessary.
- Training may take several hours depending on your hardware.

## Usage Examples

### Example: Pulling Issues from GitHub

```bash
python src/pull_issues.py
```

Sample Output:

```
Processing issue 1000
Processing issue 1001
...
Number of issues pulled: 50000
Train size (id <= 210000): 40000
Recent size (190000 <= id <= 210000): 20000
Test size (210000 < id <= 220000): 10000
```

### Example: Preprocessing Issues Data

```bash
python src/preprocessing.py
```

Sample Output:

```
Assignee counts before filtering:
user1    500
user2    450
user3    300
...
Total issues before filtering: 50000
Total issues after filtering: 48000

Assignee counts after filtering:
user1    500
user2    450
...
Preprocessing issue titles in parallel...
Extracting markdown elements (code snippets, images, links, tables) from issue bodies in parallel...
Preprocessing cleaned issue bodies in parallel...
```

### Example: Encoding Data

```bash
python src/encode_data.py
```

Sample Output:

```
No cached data found at data/encoded_data.
Loading data from 'data/issues.json'...
Encoding data...
Saving encoded data to data/encoded_data...
```

### Example: Training the Model

```bash
python src/train.py
```

Sample Output:

```
----------- TRAINING -----------
Started at 2024-10-24 12:00:00
...
Training completed. Model saved to data/checkpoints.
----------- Evaluation -----------
Evaluating on evaluation dataset...
Evaluation results: {'eval_loss': 0.5, 'eval_accuracy': 0.85}
Evaluating on test dataset...
Accuracy: 54.00%
```
