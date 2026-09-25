import argparse
import datetime
import os
import signal
import subprocess
import sys
from pathlib import Path

from training.common import read_json
from training.operations import atomic_json, identifier


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def process_identity(pid):
    file = Path(f"/proc/{pid}/stat")
    try:
        value = file.read_text()
    except FileNotFoundError:
        return None
    # The command name can contain spaces and parentheses; fields follow its final ')'.
    fields = value[value.rfind(")") + 2:].split()
    if not fields or fields[0] == "Z":
        return None
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return f"{boot}:{fields[19]}"


def status(job_dir):
    value = read_json(Path(job_dir) / "status.json")
    if value["status"] == "running":
        live = process_identity(value["supervisorPid"]) == value["supervisorIdentity"]
        if not live:
            child = value.get("childPid")
            expected_child = value.get("childIdentity")
            child_alive = child is not None and expected_child is not None and process_identity(child) == expected_child
            value = {
                **value, "status": "orphaned" if child_alive else "interrupted",
                "note": "Child still running without its supervisor; inspect the specific child PID before any restart."
                if child_alive else "Supervisor is no longer running; inspect log before explicit resume.",
            }
    return value


def run(args):
    if sys.platform != "linux":
        raise ValueError("Persistent server jobs must run on Linux through SSH.")
    import fcntl
    base, cwd, directory = args.base.resolve(), args.cwd.resolve(), args.job_dir.resolve()
    if not base.is_dir() or not cwd.is_dir() or not cwd.is_relative_to(base) or not directory.is_relative_to(base):
        raise ValueError("Job and source directories must be inside the existing project workspace.")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("No command was supplied.")
    if not 1 <= args.cpu_count <= 8 or not 1 <= args.threads <= 4:
        raise ValueError("Server jobs support at most eight CPUs and four numerical threads.")
    identifier(args.stage)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This job already has an active supervisor.") from error
        state_path = directory / "status.json"
        previous = status(directory) if state_path.exists() else None
        if previous and (not args.resume_job or previous["status"] not in ("failed", "interrupted")):
            raise ValueError("Existing job: use a new stage or explicitly resume a failed/interrupted job.")
        allowed = sorted(os.sched_getaffinity(0))[:args.cpu_count]
        os.sched_setaffinity(0, allowed)
        priority = os.nice(5)
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = str(args.threads)
        env.update({
            "CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "TORCH_HOME": str(base / "cache" / "torch"), "HF_HOME": str(base / "cache" / "huggingface"),
            "XDG_CACHE_HOME": str(base / "cache"), "MPLCONFIGDIR": str(base / "cache" / "matplotlib"),
            "YOLO_CONFIG_DIR": str(base / "cache" / "ultralytics"), "YOLO_AUTOINSTALL": "False",
            "HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_MODE": "disabled", "COMET_DISABLE_AUTO_LOGGING": "1",
        })
        state = {
            "schemaVersion": 1, "stage": args.stage, "status": "running", "startedAt": timestamp(),
            "finishedAt": None, "exitCode": None, "supervisorPid": os.getpid(),
            "supervisorIdentity": process_identity(os.getpid()), "cwd": str(cwd),
            "command": command, "gpu": args.gpu, "cpus": allowed, "threads": args.threads,
            "nice": priority, "attempt": previous["attempt"] + 1 if previous else 1, "childPid": None,
        }
        atomic_json(state_path, state)
        child = None
        interrupted = False

        def terminate(signum, _frame):
            nonlocal interrupted
            interrupted = True
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signum)

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        try:
            with (directory / "stdout.log").open("ab", buffering=0) as log:
                log.write(f"\n--- attempt {state['attempt']} at {state['startedAt']} ---\n".encode())
                child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state.update(childPid=child.pid, childIdentity=process_identity(child.pid))
                atomic_json(state_path, state)
                code = child.wait()
            state.update(status="interrupted" if interrupted else "completed" if code == 0 else "failed",
                         exitCode=code, finishedAt=timestamp())
            atomic_json(state_path, state)
            atomic_json(directory / f"attempt-{state['attempt']}.json", state)
            return code if code >= 0 else 128 - code
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            state.update(status="failed", finishedAt=timestamp(), error=str(error))
            atomic_json(state_path, state)
            atomic_json(directory / f"attempt-{state['attempt']}.json", state)
            raise


def main():
    parser = argparse.ArgumentParser(description="Supervise one project job with resource limits and persistent status; launch through tmux.")
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("run")
    start.add_argument("--base", type=Path, required=True)
    start.add_argument("--cwd", type=Path, required=True)
    start.add_argument("--job-dir", type=Path, required=True)
    start.add_argument("--stage", required=True)
    start.add_argument("--cpu-count", type=int, default=8)
    start.add_argument("--threads", type=int, default=4)
    start.add_argument("--gpu", default="")
    start.add_argument("--resume-job", action="store_true")
    start.add_argument("command", nargs=argparse.REMAINDER)
    inspect = commands.add_parser("status")
    inspect.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run":
        sys.exit(run(args))
    else:
        import json
        print(json.dumps(status(args.job_dir), indent=2))


if __name__ == "__main__":
    main()
