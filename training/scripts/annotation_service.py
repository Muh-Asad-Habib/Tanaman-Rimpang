import argparse
import hashlib
import json
import os
import secrets
import sys
import urllib.error
import urllib.request
from pathlib import Path

from training.common import digest, read_json
from training.operations import atomic_json


def credentials(service):
    path = service / "credentials.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Private annotation credentials are missing; start the service first.")
    if sys.platform == "linux" and path.stat().st_mode & 0o077:
        raise ValueError("Annotation credentials must be readable only by the SSH account (chmod 600).")
    return read_json(path)


def start(args):
    if sys.platform != "linux":
        raise ValueError("Run the annotation service on the Linux SSH server, not in the web process.")
    base, images, service = args.base.resolve(), args.images.resolve(), args.service.resolve()
    if not images.is_dir() or not images.is_relative_to(base) or not service.is_relative_to(base):
        raise ValueError("Images and service state must remain inside the isolated project workspace.")
    os.umask(0o077)
    service.mkdir(parents=True, exist_ok=True)
    path = service / "credentials.json"
    if not path.exists():
        value = {"username": "reviewer@rimpang.local", "password": secrets.token_urlsafe(30),
                 "token": secrets.token_hex(20)}
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream)
            stream.write("\n")
    account = credentials(service)
    env = os.environ.copy()
    env.update({
        "LABEL_STUDIO_USERNAME": account["username"],
        "LABEL_STUDIO_PASSWORD": account["password"],
        "LABEL_STUDIO_USER_TOKEN": account["token"],
        "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED": "true",
        "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT": str(images),
        "LABEL_STUDIO_COLLECT_ANALYTICS": "false",
        "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK": "true",
        "LABEL_STUDIO_SENTRY_DSN": "", "LABEL_STUDIO_FRONTEND_SENTRY_DSN": "",
        "LABEL_STUDIO_HOST": f"http://localhost:{args.port}",
    })
    executable = Path(sys.executable).with_name("label-studio")
    if not executable.is_file():
        raise ValueError("Install training/requirements-annotations.txt in the separate annotation environment.")
    command = [str(executable), "start", "--internal-host", "127.0.0.1", "--port", str(args.port),
               "--data-dir", str(service / "state"), "--no-browser", "--enable-legacy-api-token"]
    print(f"Private annotation service: http://localhost:{args.port}; credentials: {path}", flush=True)
    os.execve(executable, command, env)


def api(service, port, route, payload=None):
    account = credentials(service)
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}",
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Authorization": f"Token {account['token']}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Local annotation API returned HTTP {error.code} for {route}; inspect the service log.") from error


def task_identity(task):
    if not isinstance(task, dict) or not isinstance(task.get("data"), dict):
        raise ValueError("Annotation tasks must contain a data object.")
    return hashlib.sha256(json.dumps(task["data"], sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def exported_tasks(service, port, project_id):
    rows = api(service, port, f"/api/projects/{project_id}/export?exportType=JSON&download_all_tasks=true")
    if not isinstance(rows, list):
        raise ValueError("Annotation API did not return a complete JSON task export.")
    return rows


def import_tasks(args):
    service = args.service.resolve()
    tasks = read_json(args.tasks)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("The generated task list must not be empty.")
    wanted = {task_identity(row): row for row in tasks}
    if len(wanted) != len(tasks):
        raise ValueError("The input contains duplicate annotation tasks.")
    state_path = args.project_state
    new_project = not state_path.exists()
    if state_path.exists():
        project = read_json(state_path)
        if project["tasksSha256"] != digest(args.tasks) or project["configSha256"] != digest(args.label_config):
            raise ValueError("Project task/config changed; use a new project-state file instead of mixing versions.")
    else:
        created = api(service, args.port, "/api/projects/", {
            "title": args.title, "description": "Review every instance, class, source and global group before training.",
            "label_config": args.label_config.read_text(encoding="utf-8"),
        })
        project = {"schemaVersion": 1, "id": created["id"], "tasksSha256": digest(args.tasks),
                   "configSha256": digest(args.label_config), "expectedTasks": len(tasks), "status": "importing"}
        atomic_json(state_path, project)
    existing = [] if new_project else exported_tasks(service, args.port, project["id"])
    identities = [task_identity(row) for row in existing]
    known = set(identities)
    if len(known) != len(identities) or not known.issubset(wanted):
        raise ValueError("Existing project contains duplicated or unexpected tasks; review it before importing.")
    remaining = [row for identity, row in wanted.items() if identity not in known]
    for offset in range(0, len(remaining), 100):
        api(service, args.port, f"/api/projects/{project['id']}/import", remaining[offset:offset + 100])
        print(f"Submitted annotation tasks: {min(offset + 100, len(remaining))}/{len(remaining)}", flush=True)
    actual = exported_tasks(service, args.port, project["id"])
    if len(actual) != len(tasks) or {task_identity(row) for row in actual} != set(wanted):
        raise ValueError("Imported task inventory does not match the generated annotation package.")
    project.update(status="awaiting-human-review", importedTasks=len(actual))
    atomic_json(state_path, project)
    print(json.dumps({"projectId": project["id"], "tasks": len(actual), "status": project["status"],
                      "url": f"http://localhost:{args.port}/projects/{project['id']}/data"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Operate a loopback-only, private Label Studio instance.")
    parser.add_argument("--service", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8087)
    commands = parser.add_subparsers(dest="action", required=True)
    serve = commands.add_parser("start")
    serve.add_argument("--base", type=Path, required=True)
    serve.add_argument("--images", type=Path, required=True)
    load = commands.add_parser("import")
    load.add_argument("--tasks", type=Path, required=True)
    load.add_argument("--label-config", type=Path, required=True)
    load.add_argument("--project-state", type=Path, required=True)
    load.add_argument("--title", default="Rimpang - review sebelum training")
    export = commands.add_parser("export")
    export.add_argument("--project-state", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use a non-privileged port.")
    if args.action == "start":
        start(args)
    elif args.action == "import":
        import_tasks(args)
    else:
        if args.output.exists():
            parser.error("Choose a new export file; existing reviews are never overwritten.")
        project = read_json(args.project_state)
        rows = exported_tasks(args.service.resolve(), args.port, project["id"])
        if len(rows) != project["expectedTasks"]:
            raise ValueError("Export is incomplete; the project task count changed.")
        atomic_json(args.output, rows)
        print(f"Exported {len(rows)} tasks. The review importer must still validate all annotations.")


if __name__ == "__main__":
    main()
