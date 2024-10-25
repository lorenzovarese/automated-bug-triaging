# Automated Bug Triaging

A machine learning tool for automating the assignment of open issues to the best-suited developers in the VSCode GitHub repository.

## Group Information

- **Group Number**: 3
- **Participants**:
    - Fauconnet Arnaud
    - Perozzi Vittorio
    - Varese Lorenzo

## Features

- **Automated Assignment**: Automatically assigns open issues to the best-suited developers based on historical data.
- **Ranked Candidate List**: Provides a ranked list of potential assignees, displaying the most likely candidate at the top.
- **Contributor Statistics**: Shows the number of commits each candidate has authored in the VSCode repository.
- **Training on Historical Data**: Trains the model using closed issues with exactly one assignee and issue ID ≤ 210000.
- **Evaluation**: Evaluates the model on a test set consisting of closed issues with IDs from 210001 to 220000.
- **Command-Line Interface**: Simple shell-based interface with argument parsing for easy control of processing steps.

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **GitHub Personal Access Token** with permissions to read issues from the target repository
- **CUDA-compatible GPU (optional but recommended)**

### Set Up a Virtual Environment

It is recommended to use a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate
```

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

## How to Run the Tool

The tool consists of several scripts that you can run with various command-line arguments:

1. [Pull Issues from GitHub](#step-1-pull-issues-from-github)
2. [Preprocess the Issues Data](#step-2-preprocess-the-issues-data)
3. [Encode the Data](#step-3-encode-the-data)
4. [Train the Model](#step-4-train-the-model)
5. [Evaluate the Model](#step-5-evaluate-the-model)

### Step 1: Pull Issues from GitHub

Run the `pull_issues.py` script to pull issues from the target GitHub repository. By default, it pulls issues from the `microsoft/vscode` repository.

```
$ python3 src/pull_issues.py -h
usage: pull_issues.py [-h] [-f] [-r REPO] [--author2commits]
                      [--author2commits-path AUTHOR2COMMITS_PATH] [-v]

options:
  -h, --help            show this help message and exit
  -f, --force           Force re-pulling of data
  -r REPO, --repo REPO  The repository to pull issues from. Default is
                        'microsoft/vscode'
  --author2commits      Pull also the number of commits for each author in the
                        repository and save it to --author2commits-path
  --author2commits-path AUTHOR2COMMITS_PATH
                        Path to save the author2commits dictionary. Default is
                        'data/author2commits.json'
  -v, --verbose         Print verbose output
```

An example usage could be
```bash
python3 src/pull_issues.py
```

This script will:

- Connect to the GitHub API using your personal access token.
- Fetch closed issues from the specified repository.
- Filter issues to include only those with exactly one assignee.
- Save the issues data to `data/issues.json.zip`.

If you wish to pull issues from a different repository, you can pass a different url with the `-r` flag.

### Step 2: Preprocess the Issues Data

Run the `preprocessing.py` script to preprocess the issues data. This script cleans and processes the text data, extracts useful elements, and prepares the data for encoding.

```bash
python3 src/preprocessing.py
```

This script will:

- Load the issues data from `data/issues.json.zip` (or pulls the issues and saves if there if the file doesn't exist).
- Remove issues assigned to developers with fewer than a specified number of assignments (default is 50).
- Preprocess the issue titles and bodies:
    - Extract code snippets, images, links, and tables from markdown content.
    - Clean the text by removing HTML tags and special symbols.
    - Apply classical text preprocessing techniques (lowercasing, removing punctuation, stop words, and stemming).
- Save the preprocessed data to `data/issuesprep.json.zip`.

### Step 3: Encode the Data

Run the `encode_data.py` script to tokenize and encode the preprocessed data. You can use various arguments to customize the behavior:

```bash
python3 src/encode_data.py --force --verbose --frac-of-data 0.8
```

| Argument                  | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `--force`                  | Force re-encoding of data.                                                  |
| `--verbose`                | Print verbose output.                                                       |
| `--only-recent`            | Only encode recent data.                                                    |
| `--classical-preprocessing`| Use classical preprocessing (stemming + stopwords removal) instead of raw data. |
| `--frac-of-data`           | Specify the fraction of data to encode (default is 1).                       |
| `--encoded-data-path`      | Path to save the encoded dataset (default is `data/encoded_data`).           |
| `--num-proc`               | Number of processes to use for encoding (default is the number of available CPUs).|

### Step 4: Train the Model

Run the `train.py` script to train the model on the encoded data. You can control the behavior of the training process with arguments such as `--train-model` to start training or `--checkpoint` to load an existing model:

```bash
python3 src/train.py --train-model --frac-of-data 0.8
```

| Argument                   | Description                                                               |
|-----------------------------|---------------------------------------------------------------------------|
| `--train-model`             | Force the training of the model.                                           |
| `--frac-of-data`            | Fraction of data to use for training (default is 1).                        |
| `--only-recent`             | Use only recent data for training.                                         |
| `--checkpoint`              | Path to a checkpoint to load a pretrained model.                           |
| `--classical-preprocessing` | Use classical preprocessing (stemming + stopwords removal).                |
| `--encoded-data-path`       | Path to the encoded dataset (default is `data/encoded_data`).              |

### Step 5: Evaluate the Model

Run the `eval.py` script to evaluate the model. The script will output evaluation results such as accuracy and loss:

```bash
python3 src/eval.py --frac-of-data 0.8 --checkpoint data/checkpoints/model_checkpoint
```

| Argument                   | Description                                                               |
|-----------------------------|---------------------------------------------------------------------------|
| `--frac-of-data`            | Fraction of data to use for evaluation (default is 1).                      |
| `--checkpoint`              | Path to a checkpoint to load the model for evaluation.                     |

### Additional Step: Run Baseline Classifier for Comparison

You can run the baseline classifier for comparison against the machine learning model:

```bash
python3 src/baseline_issue_assignment_classifier.py --verbose
```

This will output accuracy results for the baseline classifiers (Naive Bayes, Random, and Weighted Random):

```text
Naive Bayes Classifier Accuracy: 19.99%
Random Classifier Accuracy: 3.13%
Weighted Random Classifier Accuracy: 5.05%
```

## Usage Examples

### Example: Pulling Issues from GitHub

```bash
python3 src/pull_issues.py
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
python3 src/preprocessing.py
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
Preprocessing issue titles in parallel...
Extracting markdown elements (code snippets, images, links, tables) from issue bodies in parallel...
Preprocessing cleaned issue bodies in parallel...
```

### Example: Encoding Data

```bash
python3 src/encode_data.py --force --frac-of-data 0.8 --verbose
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
python3 src/train.py --train-model --frac-of-data 0.8
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

### Example: Evaluating the Model

```bash
python3 src/eval.py --frac-of-data 0.8 --checkpoint data/checkpoints/model_checkpoint
```

Sample Output:

```
Evaluating on evaluation dataset...
Evaluation results: {'eval_loss': 0.5, 'eval_accuracy': 0.85}
Evaluating on test dataset...
Accuracy: 54.00%
```
