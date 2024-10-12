import os
import sys
import argparse
import pandas as pd
from dotmap import DotMap
from github import Github
from github import Auth
from github.Repository import Repository


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Add labels to issues based on the time they were added."
    )

    parser.add_argument(
        "--path",
        type=str,
        default="../res/issues.json.zip",  # ../res/ o res/
        help="The path to existing issues file."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="labeled_issues.json.zip",
        help="The name of the output file. (it will be saved in the same directory as the input file)"
    )

    parser.add_argument(
        "--seconds",
        type=int,
        default=60,
        help="Seconds within which the label must have been added."
    )

    parser.add_argument(
        "--repository",
        type=str,
        default="microsoft/vscode",
        help="The repository to fetch the issues from."
    )

    parser.add_argument(
        "--token",
        type=str,
        default="INSERT_YOUR_TOKEN",
        help="The authentication token, if not provided it will be read from the environment variable"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"Error: The file '{args.path}' does not exist.")
        sys.exit(1)

    return args


def get_repo(token, repository) -> Repository:
    if token == "INSERT_YOUR_TOKEN":
        assert "GITHUB_AUTH_TOKEN" in os.environ, "Please set the GITHUB_AUTH_TOKEN environment variable"
        auth_token = os.environ["GITHUB_AUTH_TOKEN"]
        assert len(auth_token) > 0, "Please provide a valid authentication token"
    else:
        auth_token = token

    auth = Auth.Token(auth_token)
    g = Github(auth=auth)
    return g.get_repo(repository)


def get_creation_labels(issue_id, repo: Repository, seconds) -> str:
    """
    Get the labels that were added to an issue within seconds of its creation.

    Args:
        seconds: time in seconds to consider the label as created
        issue_id (int): The ID of the issue.
        repo (Repository): The GitHub repository object.

    Returns:
        str: A space-separated string of label names.
    """

    issue = repo.get_issue(number=issue_id)
    events = issue.get_timeline()
    created_labels = []
    for event in events:
        if event.event == "labeled":
            if ((event.created_at - issue.created_at).total_seconds()) < seconds:
                label_name = event.raw_data.get("label", {}).get("name")
                created_labels.append(label_name)

    return " ".join(created_labels)


def create_labeled_issues(args, testing=False):
    """
    Create new labeled file by fetching labels added within a specific time frame.

    This function reads issues from a JSON file, fetches the labels added to each issue
    within `args.seconds` seconds of its creation, and writes the updated issues
    with their labels to another JSON file.

    testing: False to run all the issues, otherwise the number of first issues to run
    """

    repo = get_repo(args.token, args.repository)
    df = pd.read_json(args.path, orient="records")
    records = df.to_dict(orient="records")
    dot_records = [DotMap(record) for record in records]

    n = -1
    for issue in dot_records:
        n += 1
        issue.creation_labels = get_creation_labels(issue.github_id, repo, args.seconds)
        print("N: " + str(n) + " G_ID: " + str(issue.github_id) + " labels: " + issue.creation_labels)
        if (testing != False) and (n >= testing): break

    updated_records = [record.toDict() for record in dot_records]
    updated_df = pd.DataFrame(updated_records)
    original_file_name = os.path.basename(args.path)
    out_path = args.path.replace(original_file_name, "labeled_issues.json")
    updated_df.to_json(out_path, orient="records", indent=4)
    return


if __name__ == "__main__":
    arguments = parse_arguments()
    create_labeled_issues(arguments)
