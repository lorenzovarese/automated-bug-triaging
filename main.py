from transformers import AutoModelForSequenceClassification
from src.pull_issues import load_repo
from src.preprocessing import extract_markdown_elements
from src.encode_data import TOKENIZER
import argparse
import json

REPO = "microsoft/vscode"

def load_issue_text(issue_number):
    """
    Load the issue body for the given issue number
    """

    repo = load_repo(REPO)
    issue = repo.get_issue(number=issue_number)

    *_, cleaned_body = extract_markdown_elements(issue.body)
    return issue.title + " " + cleaned_body

def number_of_commits_from(author):
    print(f"Counting commits from '{author}'...")
    repo = load_repo(REPO)
    commits = repo.get_commits(author=author)
    print(commits)
    return commits.totalCount


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give a ranked list of candidate assignees for a given issue number for the VS Code Github repository")

    parser.add_argument("-m", "--model", type=str, required=True, help="Model to use for the prediction. Usually './data/checkpoints/checkpoint-XXXX'")
    parser.add_argument("-i", "--issue", type=int, required=True, help="Issue number for which to predict assignees")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")


    args = parser.parse_args()

    if args.verbose: print(f"Loading issue {args.issue}...")
    issue = load_issue_text(args.issue)

    if args.verbose: print("Encoding issue...")
    encoded_issue = TOKENIZER(issue, return_tensors="pt")

    if args.verbose: print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model)

    if args.verbose: print("Predicting assignees...")
    output = model(**encoded_issue)

    if args.verbose: print("Loading the number of commits for each author...")
    author2commits = json.load(open("data/author2commits.json"))

    logits = output.logits
    ranks = logits.argsort(descending=True)
    print(f"Ranked list of candidate assignees for issue {args.issue}")
    for i, rank in enumerate(ranks.squeeze()[:5], start=1):
        author = model.config.id2label[rank.item()]
        print(f"{i}. {author} ({author2commits.get(author, 0)} commits in the repo)")

