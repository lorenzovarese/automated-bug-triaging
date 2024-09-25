import os
from github import Github
from github import Auth
import pandas as pd
from tqdm import tqdm


MAX_ISSUE_ID = 220_000
ISSUES_FILE = "res/issues.json.zip"


def pull_issues(
        github_repo: str, 
        force_pull=False, 
    ) -> pd.DataFrame:
    """
    Get the issues from the given url.

    Args:
        github_repo (str): the url of the github repository
        force_pull (bool, optional, default = False): force to pull the data. If False, the value returned is the one cached locally (if any), if True pulls and caches a new version

    Returns:
        list: A dataframe containing the issues. The columns of the dataframe are:
            - github_id: the id of the issue
            - title: the title of the issue
            - body: the body of the issue
    """

    if not force_pull and os.path.exists(ISSUES_FILE):
        df = pd.read_json(ISSUES_FILE)
        return df

    ret = []

    assert "GITHUB_AUTH_TOKEN" in os.environ, "Please set the GITHUB_AUTH_TOKEN environment variable"
    auth_token = os.environ["GITHUB_AUTH_TOKEN"]

    assert len(auth_token) > 0, "Please provide a valid authentication token"
    auth = Auth.Token(auth_token)
    g = Github(auth=auth)

    repo = g.get_repo(github_repo)
    issues = repo.get_issues(state="closed", direction="asc")

    with tqdm(total=issues.totalCount, ncols= 200) as pbar:
        for issue in issues:
            pbar.update(1)
            pbar.set_description(f"Processing issue {issue.number}")
            if issue.number > MAX_ISSUE_ID or len(issue.assignees) != 1:
                continue

            issue_info = {
                "github_id": issue.number,
                "title": issue.title,
                "body": issue.body,
            }
            ret.append(issue_info)

    df = pd.DataFrame(ret)
    df.to_json(ISSUES_FILE, orient="records")

    return df

if __name__ == "__main__":
    # Example of usage
    df = pull_issues("microsoft/vscode")
    print(df.head())
    print(df.shape)

    df_train = df[df["github_id"] <= 210_000]
    df_recent = df[(190_000 <= df["github_id"]) & (df["github_id"] <= 210_000)]
    df_test = df[(210_000 < df["github_id"]) & (df["github_id"] <= 220_000)]

    print(f"Train size (id <= 210'000): {df_train.shape[0]}")
    print(f"Recent size (190'000 <= id <= 210'000): {df_recent.shape[0]}")
    print(f"Test size (210'000 < id <= 220'000): {df_test.shape[0]}")

    print(df.github_id.describe())
