import os
from github import Github
from github import Auth
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


MAX_ISSUE_ID = 220_000
ISSUES_FILE = "res/issues.json.zip"
N_THREADS = 5


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

    def process_issues(issues_slice, thread_id, start, end):
        """ Process a slice of issues """
        print(f"Thread-{thread_id} - Processing from {start} to {end}")
        results = []
        with tqdm(total=end-start, desc=f"Thread-{thread_id}", ncols=100) as pbar:
            for issue in issues_slice:
                results.append({
                    "github_id": issue.number,
                    "title": issue.title,
                    "body": issue.body,
                })
                pbar.update(1)
                pbar.set_description(f"Thread-{thread_id} - {issue.number}")
        return results

    # Helper to fetch issues slice for each thread
    def fetch_issues_for_thread(start_idx, end_idx):
        return issues[start_idx:end_idx]

    
    issues_per_thread = issues.totalCount // N_THREADS
    issue_ranges = [(i * issues_per_thread, (i + 1) * issues_per_thread) for i in range(N_THREADS)]

    # Adjust for any leftover issues
    issue_ranges[-1] = (issue_ranges[-1][0], issues.totalCount)
    
    all_results = []

    # Use ThreadPoolExecutor to multithread the fetching of issues
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = {
            executor.submit(process_issues, fetch_issues_for_thread(start, end), thread_id, start, end): thread_id
            for thread_id, (start, end) in enumerate(issue_ranges)
        }

        # Collect the results as threads complete
        for future in as_completed(futures):
            thread_id = futures[future]
            try:
                result = future.result()
                all_results.extend(result)
            except Exception as exc:
                print(f"Thread-{thread_id} generated an exception: {exc}")
        df = pd.DataFrame(ret)
        df.to_json(ISSUES_FILE, orient="records")

    return df

if __name__ == "__main__":
    # Example of usage
    df = pull_issues("microsoft/vscode", force_pull=True)
    print(df.head())
    print(df.shape)
    df_train = df[df["github_id"] < 210_000]

    df_recent = df[(190_000 <= df_train["github_id"]) & (df_train["github_id"] <= 210_000)]

    df_test = df[(210_000 < df["github_id"]) & (df["github_id"] <= 220_000)]

    print(f"Train: {df_train.shape[0]}")
    print(f"Recent: {df_recent.shape[0]}")
    print(f"Test: {df_test.shape[0]}")
