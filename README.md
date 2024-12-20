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

```bash
python3 src/pull_issues.py
```

This script will:

- Connect to the GitHub API using your personal access token.
- Fetch closed issues from the specified repository.
- Filter issues to include only those with exactly one assignee.
- Save the issues data to `data/issues.json.zip`.

| Argument                    | Description                                                               |
|-----------------------------|---------------------------------------------------------------------------|
| `-f`, `--force`             | Force re-pulling of data                                                  |
| `-r REPO`, `--repo REPO`    | The repository to pull issues from (default is 'microsoft/vscode')        |
| `--author2commits`          | Pull also the number of commits for each author in the repository and save it to --author2commits-path |
| `--author2commits-path AUTHOR2COMMITS_PATH` | Path to save the author2commits dictionary (default is 'data/author2commits.json') |
| `-v`, `--verbose` | Print verbose output |
           

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
| `-f`, `--force`                  | Force re-encoding of data.                                                  |
| `-v`, `--verbose`                | Print verbose output.                                                       |
| `-r`, `--only-recent`            | Only encode recent data.                                                    |
| `--classical-preprocessing`| Use classical preprocessing (stemming + stopwords removal) instead of raw data. |
| `--frac-of-data FRAC_OF_DATA`           | Specify the fraction of data to encode (default is 1).                       |
| `--encoded-data-path ENCODED_DATA_PATH`      | Path to save the encoded dataset (default is `data/encoded_data`).           |
| `--num-proc NUM_PROC`               | Number of processes to use for encoding (default is the number of available CPUs).|

### Step 4: Evaluate the Model

Run the `eval.py` script with `-t` to train the model on the encoded data. It will train a model and show the performance on the test set.

```bash
python3 src/eval.py -t --frac-of-data 0.8
```

If you wish to only evaluate a model, do not use the `-t` option, and just load the checkpoint you want with the `-c` argument.

```bash
python3 src/eval.py -c data/checkpoints/best-model
```

| Argument                   | Description                                                               |
|-----------------------------|---------------------------------------------------------------------------|
| `-t`, `--train-model`             | Force the training of the model.                                           |
| `--frac-of-data FRAC_OF_DATA`            | Fraction of data to use for training (default is 1).                        |
| `-r`, `--only-recent`             | Use only recent data for training.                                         |
| `-c`, `--checkpoint CHECKPOINT`              | Path to a checkpoint to load a pretrained model. Ignored if `--train-model` is used.                           |
| `--classical-preprocessing` | Use classical preprocessing (stemming + stopwords removal).                |
| `--encoded-data-path`       | Path to the encoded dataset (default is `data/encoded_data`).              |

### Step 5: Use the model

Now that we did everything, we can use the tool for it's intended purpose: get a
ranked list of users that are the most likely assignees accroding to our model.

```bash
python3 main.py -c data/checkpoints/best-model -i 232113
```

| Argument                   | Description                                                               |
|-----------------------------|---------------------------------------------------------------------------|
| `-m`, `--model MODEL`            | **Required**. Model to use for the prediction. Usually `./data/checkpoints/checkpoint-XXXX` or `./data/checkpoints/best-model`                        |
| `-i`, `--issue ISSUE` | **Required**. Issue number for which to predict assignees  |

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

### Example: Using the tool

```bash
python3 main.py -m data/checkpoints/best-model -i 232113
```

Sample Output:
```
Ranked list of candidate assignees for issue 232113
1. mjbvz (8407 commits in the repo)
2. alexr00 (2374 commits in the repo)
3. aeschli (4553 commits in the repo)
4. sandy081 (7918 commits in the repo)
5. jrieken (11665 commits in the repo)
```
