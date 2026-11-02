"""Download the Computer Vision lecture slides into this folder.

The slides are for participants of this course only. Do not upload, share, or commit them.

Only new or changed slides are downloaded, unless --force is given. If the server requires a
login, credentials are read from CV_SLIDES_USER / CV_SLIDES_PASSWORD or asked for interactively.

Usage (from the repository root):
    python slides/download.py           # all lectures
    python slides/download.py 6 7       # only lectures 6 and 7
    python slides/download.py --force   # re-download everything
"""

import argparse
import email.utils
import getpass
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://lmb.informatik.uni-freiburg.de/lectures/computer_vision_I/slides/"
NUM_LECTURES = 15
SLIDES_DIR = Path(__file__).resolve().parent


def build_opener(user: str, password: str) -> urllib.request.OpenerDirector:
    """Create a URL opener that answers HTTP basic auth challenges with the given credentials."""
    manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, BASE_URL, user, password)
    return urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(manager))


def ask_credentials() -> tuple:
    """Read credentials from the environment, or prompt for them if running interactively."""
    user = os.environ.get("CV_SLIDES_USER")
    password = os.environ.get("CV_SLIDES_PASSWORD")
    if user and password:
        return user, password
    if not sys.stdin.isatty():
        sys.exit(
            "The server requires a login. Set CV_SLIDES_USER and CV_SLIDES_PASSWORD "
            "(credentials are announced in the lecture / on ILIAS)."
        )
    print("The server requires a login (credentials are announced in the lecture / on ILIAS).")
    return input("Username: "), getpass.getpass("Password: ")


def download(opener: urllib.request.OpenerDirector, num: int, force: bool) -> str:
    """Download the slides of one lecture.

    Returns:
        "new", "updated", "unchanged", "missing", "auth" (login required / rejected), or
        "failed (HTTP <code>)".
    """
    name = f"ComputerVision{num:02d}.pdf"
    target = SLIDES_DIR / name
    request = urllib.request.Request(BASE_URL + name)
    if target.exists() and not force:
        request.add_header(
            "If-Modified-Since", email.utils.formatdate(target.stat().st_mtime, usegmt=True)
        )
    try:
        with opener.open(request, timeout=60) as response:
            data = response.read()
            last_modified = response.headers.get("Last-Modified")
    except urllib.error.HTTPError as e:
        if e.code == 304:
            return "unchanged"
        if e.code == 404:
            return "missing"
        if e.code in (401, 403):
            return "auth"
        return f"failed (HTTP {e.code})"
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach the server: {e.reason}")

    status = "updated" if target.exists() else "new"
    tmp = target.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    if last_modified:
        # keep the server's timestamp so If-Modified-Since works on the next run
        mtime = email.utils.parsedate_to_datetime(last_modified).timestamp()
        os.utime(target, (mtime, mtime))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lectures", type=int, nargs="*", help="lecture numbers (default: all)")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    lectures = args.lectures or range(1, NUM_LECTURES + 1)

    # The server answers 401 both for protected files and for files that don't exist yet, so only
    # log in if credentials are given, or if nothing at all could be downloaded without them.
    has_env_credentials = bool(os.environ.get("CV_SLIDES_USER"))
    if has_env_credentials:
        opener = build_opener(*ask_credentials())
    else:
        opener = urllib.request.build_opener()
    statuses = {num: download(opener, num, args.force) for num in lectures}
    if not has_env_credentials and all(s == "auth" for s in statuses.values()):
        opener = build_opener(*ask_credentials())
        statuses = {num: download(opener, num, args.force) for num in lectures}

    labels = {
        "auth": "not available (not uploaded yet, or login required)",
        "missing": "not available",
    }
    for num, status in statuses.items():
        print(f"Lecture {num:2d}: {labels.get(status, status)}")
    if all(s == "auth" for s in statuses.values()):
        sys.exit("Nothing could be downloaded. Check the username and password.")
    if any(s in ("new", "updated", "unchanged") for s in statuses.values()):
        print(
            "\nThe slides are for participants of this course only. "
            "Please do not upload, share, or commit them."
        )


if __name__ == "__main__":
    main()
