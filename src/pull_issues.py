import os
from github import Github
from github import Auth
import pandas as pd


MAX_ISSUE_ID = 220_000
ISSUES_FILE = "res/issues.json"


def pull_issues(
        github_repo: str, 
        auth_token: str, 
        force_pull=False, 
    ) -> pd.DataFrame:
    """
    Get the issues from the given url.

    Args:
        github_repo (str): the url of the github repository
        auth_token (str): the authentication token
        force_pull (bool, optional, default = False): force to pull the data. If False, the value returned is the one cached locally (if any), if True pulls and caches a new version

    Returns:
        list: A dataframe containing the issues. The columns of the dataframe are:
            - github_id: the id of the issue
            - title: the title of the issue
            - body: the body of the issue
            - assignee: the assignee of the issue
    """

    if not force_pull and os.path.exists(ISSUES_FILE):
        with open(ISSUES_FILE, 'r') as file:
            df = pd.read_json(file)
            return df

    ret = []

    assert len(auth_token) > 0, "Please provide a valid authentication token"
    auth = Auth.Token(auth_token)
    g = Github(auth=auth)

    repo = g.get_repo(github_repo)
    issues = repo.get_issues(state="closed", direction="asc")

    for issue in issues:
        if issue.number > MAX_ISSUE_ID:
            break
        if len(issue.assignees) != 1:
            continue

        issue_info = {
            "github_id": issue.number,
            "title": issue.title,
            "body": issue.body,
            "assignee": issue.assignee.login,
        }
        ret.append(issue_info)

    df = pd.DataFrame(ret)
    df.to_json(ISSUES_FILE, orient="records")

    return df


if __name__ == "__main__":
    assert "GITHUB_AUTH_TOKEN" in os.environ, "Please set the GITHUB_AUTH_TOKEN environment variable"
    token = os.environ["GITHUB_AUTH_TOKEN"]

    df = pull_issues("microsoft/vscode", token, force_pull=True)
