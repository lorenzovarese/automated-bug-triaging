import os
import pandas as pd
from dotmap import DotMap
from github import Github
from github import Auth
from github.Repository import Repository

ORIGINAL_ISSUES_FILE = "../res/issues.json.zip"  # ../res/ o res/
LABELED_ISSUES_FILE = "../res/labeled_issues.json.zip"
SECONDS_TO_LABEL = 60  # Secondi entro i quali l'etichetta deve essere stata aggiunta
REPOSITORY = "microsoft/vscode"
AUTH_TOKEN = "INSERT_YOUR_TOKEN"  # oppure puoi farlo tramite variabile d'ambiente

testing = False  # False to run all the issues, otherwise the number of firsts issues to run


def get_repo() -> Repository:
    if AUTH_TOKEN == "INSERT_YOUR_TOKEN":
        assert "GITHUB_AUTH_TOKEN" in os.environ, "Please set the GITHUB_AUTH_TOKEN environment variable"
        auth_token = os.environ["GITHUB_AUTH_TOKEN"]
        assert len(auth_token) > 0, "Please provide a valid authentication token"
    else:
        auth_token = AUTH_TOKEN

    auth = Auth.Token(auth_token)
    g = Github(auth=auth)
    return g.get_repo(REPOSITORY)


def get_creation_labels(issue_id, repo: Repository) -> str:
    """
    Get the labels that were added to an issue within SECONDS_TO_LABEL seconds of its creation.

    Args:
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
            if ((event.created_at - issue.created_at).total_seconds()) < 60:
                label_name = event.raw_data.get("label", {}).get("name")
                created_labels.append(label_name)

    return " ".join(created_labels)


def create_labeled_issues():
    """
    Create new labeled file by fetching labels added within a specific time frame.

    This function reads issues from a JSON file, fetches the labels added to each issue
    within `SECONDS_TO_LABEL` seconds of its creation, and writes the updated issues
    with their labels to another JSON file.
    """

    repo = get_repo()
    df = pd.read_json(ORIGINAL_ISSUES_FILE, orient="records")
    records = df.to_dict(orient="records")
    dot_records = [DotMap(record) for record in records]

    n = -1
    for issue in dot_records:
        n += 1
        issue.creation_labels = get_creation_labels(issue.github_id, repo)
        print("N: " + str(n) + " G_ID: " + str(issue.github_id) + " labels: " + issue.creation_labels)
        if (testing != False) and (n >= testing): break

    updated_records = [record.toDict() for record in dot_records]
    updated_df = pd.DataFrame(updated_records)
    updated_df.to_json(LABELED_ISSUES_FILE, orient="records", indent=4)
    return


if __name__ == "__main__":
    get_repo()
    create_labeled_issues()
