import json

from github import Github
from github import Auth

INITIAL_ISSUE_ID = 210000
FINAL_ISSUE_ID = 220000

if __name__ == '__main__':
    auth = Auth.Token(ACCESS_TOKEN)
    g = Github(auth=auth)
    repo = g.get_repo("microsoft/vscode")
    debug_initial_core_remaining = g.get_rate_limit().core.remaining

    initial_issue_date = repo.get_issue(number=INITIAL_ISSUE_ID).created_at

    issues = repo.get_issues(state='all', direction='asc', since=initial_issue_date)

    issues_data = []
    debug = 0
    for issue in issues.get_page(400):
        print(issue.number, debug)
        debug += 1
        if issue.number < INITIAL_ISSUE_ID:
            continue
        if issue.number > FINAL_ISSUE_ID: break
        issue_info = {
            "id": issue.number,
            "title": issue.title,
            "state": issue.state,
            "created_at": issue.created_at.isoformat(),
            "created_by": issue.user.login,
            "url": issue.html_url,
            "is_pull_request": issue.pull_request is not None,
        }
        issues_data.append(issue_info)

    with open("A.json", "w") as json_file:
        json.dump(issues_data, json_file, indent=4)

    print("USED ", debug_initial_core_remaining - g.get_rate_limit().core.remaining, g.get_rate_limit().core.remaining)
    g.close()

    ##
    # todo
    #  mancano:
    #  un controllo per quando ho finito le richieste.
    #  verificare se si possono usare ancora meno chiamate
    #  gestire il 400 automaticamente le for
    #  cambiare nome al json
    #  .
