"""Create/refresh the Label Studio annotation project.

Prereqs (once):
    uv tool install label-studio
    LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \\
    LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<repo>/dataset/ls_media \\
      label-studio start
    # then grab your API token from Account & Settings -> Access Token

Run:
    LS_URL=http://localhost:8080 LS_TOKEN=... uv run python tuning/labelstudio_import.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tuning.eval_lib.dataset import DATASET_DIR

LABEL_CONFIG = """
<View>
  <Header value="$uid / $rid"/>
  <Header value="speaker: $speaker_id — $n_speakers speaker(s), $duration s, $timestamp"/>
  <Audio name="audio" value="$audio"/>
  <Text name="excerpt" value="$transcript_excerpt"/>
  <Choices name="quality" toName="audio" choice="single" required="true">
    <Choice value="clean" hint="one person's voice"/>
    <Choice value="mixed" hint="diarization merged two people"/>
    <Choice value="garbage" hint="noise / TV / music / not a voice"/>
  </Choices>
  <TextArea name="person" toName="audio" required="true" maxSubmissions="1"
            placeholder="person key (lowercase-slug): tarik / ahmed / guest-cafe-1"/>
</View>
"""


def session_login(base: str, username: str, password: str) -> requests.Session:
    """Cookie-session login — survives Label Studio's token-regime changes."""
    s = requests.Session()
    r = s.get(f"{base}/user/login/", timeout=30)
    r.raise_for_status()
    csrf = s.cookies.get("csrftoken", "")
    r = s.post(
        f"{base}/user/login/",
        data={"email": username, "password": password, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{base}/user/login/"},
        timeout=30,
        allow_redirects=True,
    )
    r.raise_for_status()
    whoami = s.get(f"{base}/api/current-user/whoami", timeout=30)
    if whoami.status_code != 200:
        raise SystemExit(f"login failed for {username} (HTTP {whoami.status_code})")
    return s


def main() -> None:
    base = os.environ.get("LS_URL", "http://localhost:8080").rstrip("/")
    username = os.environ.get("LS_USERNAME")
    password = os.environ.get("LS_PASSWORD")
    if not (username and password):
        raise SystemExit("set LS_USERNAME and LS_PASSWORD")
    s = session_login(base, username, password)
    csrf = {"X-CSRFToken": s.cookies.get("csrftoken", ""), "Referer": base}

    tasks = json.loads((DATASET_DIR / "ls_tasks.json").read_text())

    r = s.post(
        f"{base}/api/projects",
        headers=csrf,
        json={
            "title": "impronta speaker annotation",
            "description": "Who is speaking, and is it one clean voice?",
            "label_config": LABEL_CONFIG,
        },
        timeout=30,
    )
    r.raise_for_status()
    project_id = r.json()["id"]
    print(f"created project {project_id}")

    r = s.post(
        f"{base}/api/projects/{project_id}/import",
        headers=csrf,
        json=tasks,
        timeout=300,
    )
    r.raise_for_status()
    print(f"imported {r.json().get('task_count', len(tasks))} tasks")
    print(f"annotate at {base}/projects/{project_id}")


if __name__ == "__main__":
    main()
