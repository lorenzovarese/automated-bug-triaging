import pandas as pd
from dotmap import DotMap
from github import Github
from github import Auth
from github.Repository import Repository

ORIGINAL_ISSUES_FILE = "../res/issues.json.zip"  # ../res/ o res/
LABELED_ISSUES_FILE = "../res/labeled_issues.json"
SECONDS_TO_LABEL = 60  # Secondi entro i quali l'etichetta deve essere stata aggiunta
REPOSITORY = "microsoft/vscode"
AUTH_TOKEN = "INSERT_YOUR_TOKEN"

repo: Repository
testing = False  # False to run all the issues, otherwise the number of firsts issues to run


def init_github():
    auth = Auth.Token(AUTH_TOKEN)
    g = Github(auth=auth)
    global repo
    repo = g.get_repo(REPOSITORY)


def get_creation_labels(issue_id):
    issue = repo.get_issue(number=issue_id)
    events = issue.get_timeline()
    created_labels = []
    for event in events:
        if event.event == "labeled":
            if ((event.created_at - issue.created_at).total_seconds()) < 60:
                label_name = event.raw_data.get("label", {}).get("name")
                created_labels.append(label_name)

    return " ".join(created_labels)
    # return created_labels


def create_labeled_issues():
    df = pd.read_json(ORIGINAL_ISSUES_FILE, orient="records")
    records = df.to_dict(orient="records")
    dot_records = [DotMap(record) for record in records]

    n = -1
    for issue in dot_records:
        n += 1
        issue.creation_labels = get_creation_labels(issue.github_id)
        print("N: " + str(n) + " G_ID: " + str(issue.github_id) + " labels: " + issue.creation_labels)
        if (testing != False) and (n >= testing): break

    updated_records = [record.toDict() for record in dot_records]
    updated_df = pd.DataFrame(updated_records)
    updated_df.to_json(LABELED_ISSUES_FILE, orient="records", indent=4)
    return


if __name__ == "__main__":
    init_github()
    create_labeled_issues()
