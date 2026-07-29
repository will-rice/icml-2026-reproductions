# Direct Worker Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continuously maintain 10 direct Agy and 5 direct Codex persistent paper-owner workers with stable identities, quota-aware restart behavior, and no credential leakage.

**Architecture:** A user-level systemd timer runs an idempotent Python reconciler every 30 seconds. The reconciler owns stable tmux sessions, recognizes only live `agy` or `codex` foreground processes as healthy, rotates compatible Agy model profiles after quota failures, and stores sanitized runtime state outside Git.

**Tech Stack:** Python 3.10 standard library, tmux, systemd user units, pytest, direct `agy` and `codex` CLIs.

## Global Constraints

- Maintain exactly `agy-paper-owner-01` through `agy-paper-owner-10` and `codex-paper-owner-01` through `codex-paper-owner-05`.
- Launch `agy` and `codex` directly; never use OpenCode or another dispatch layer.
- Do not interrupt a healthy existing worker during installation or reconciliation.
- Agy receives automatic tool approval, internet, workspace writes, and controller credential access.
- Codex receives workspace writes, explicit internet access, and controller credential access.
- Never serialize credential values, prompts, or environment dumps.
- The supervisor must not read, infer, claim, release, or repair paper attempts.
- Runtime state belongs under `~/.local/state/icml-worker-supervisor/`, outside Git.
- Parent validation continues to exclude the archival `submissions/nape/` snapshot.

---

## File Structure

- Create `ops/__init__.py`: mark the host-operations package.
- Create `ops/worker_supervisor.py`: configuration, tmux adapter, reconciliation, status persistence, CLI, and systemd installation.
- Create `ops/systemd/icml-worker-supervisor.service`: one-shot user service.
- Create `ops/systemd/icml-worker-supervisor.timer`: persistent 30-second timer.
- Create `tests/test_worker_supervisor.py`: isolated unit and CLI tests with fake host adapters.
- Modify `docs/REMOTE_SETUP.md`: document installation, status, dry-run, and safe stop commands.

### Task 1: Desired Workers, Direct Commands, and Redaction

**Files:**
- Create: `ops/__init__.py`
- Create: `ops/worker_supervisor.py`
- Test: `tests/test_worker_supervisor.py`

**Interfaces:**
- Produces: `WorkerSpec`, `ModelProfile`, `desired_workers()`, `launch_shell_command()`, and `sanitize_text()`.
- `WorkerSpec` fields: `worker_id: str`, `agent: Literal["agy", "codex"]`, `session_name: str`.
- `ModelProfile` fields: `name: str`, `argv: tuple[str, ...]`.
- `launch_shell_command(spec, profile, repo_root) -> str` returns a shell command containing credential lookup expressions but no credential values.

- [ ] **Step 1: Write failing desired-pool and direct-command tests**

```python
from pathlib import Path

from ops import worker_supervisor as supervisor


def test_desired_workers_are_stable_ten_agy_and_five_codex():
    workers = supervisor.desired_workers()
    assert [w.worker_id for w in workers[:10]] == [
        f"agy-paper-owner-{index:02d}" for index in range(1, 11)
    ]
    assert [w.worker_id for w in workers[10:]] == [
        f"codex-paper-owner-{index:02d}" for index in range(1, 6)
    ]
    assert len({w.session_name for w in workers}) == 15


def test_agy_launch_is_direct_full_approval_and_credential_safe():
    spec = supervisor.desired_workers()[0]
    profile = supervisor.agy_profiles()[0]
    command = supervisor.launch_shell_command(spec, profile, Path("/repo"))
    assert " opencode " not in f" {command} "
    assert " agy " in f" {command} "
    assert "--dangerously-skip-permissions" in command
    assert "$(hf auth token)" in command
    assert "$(gh auth token)" in command
    assert "Use the shared icml-repro-loop skill directly" in command
    assert spec.worker_id in command


def test_codex_launch_has_workspace_network_and_direct_prompt():
    spec = supervisor.desired_workers()[10]
    command = supervisor.launch_shell_command(
        spec, supervisor.codex_profile(), Path("/repo")
    )
    assert " codex exec " in f" {command} "
    assert "--sandbox workspace-write" in command
    assert "sandbox_workspace_write.network_access=true" in command
    assert "-C /repo" in command
    assert spec.worker_id in command
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  tests/test_worker_supervisor.py::test_desired_workers_are_stable_ten_agy_and_five_codex \
  tests/test_worker_supervisor.py::test_agy_launch_is_direct_full_approval_and_credential_safe \
  tests/test_worker_supervisor.py::test_codex_launch_has_workspace_network_and_direct_prompt
```

Expected: FAIL because `ops.worker_supervisor` does not exist.

- [ ] **Step 3: Implement immutable specifications and command construction**

Create these core definitions:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import shlex

AgentName = Literal["agy", "codex"]
PROMPT = (
    "Use the shared icml-repro-loop skill directly and keep running its "
    "paper-owner loop. Read and follow "
    "/home/will/.agents/skills/icml-repro-loop/SKILL.md. "
    "Persistent worker ID: {worker_id}."
)


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    agent: AgentName
    session_name: str


@dataclass(frozen=True)
class ModelProfile:
    name: str
    argv: tuple[str, ...]


def desired_workers() -> tuple[WorkerSpec, ...]:
    agy = tuple(
        WorkerSpec(f"agy-paper-owner-{i:02d}", "agy", f"agy-paper-owner-{i:02d}")
        for i in range(1, 11)
    )
    codex = tuple(
        WorkerSpec(
            f"codex-paper-owner-{i:02d}", "codex", f"codex-paper-owner-{i:02d}"
        )
        for i in range(1, 6)
    )
    return agy + codex


def agy_profiles() -> tuple[ModelProfile, ...]:
    return (
        ModelProfile(
            "gemini-3.1-pro-high",
            ("agy", "--dangerously-skip-permissions", "--effort", "high",
             "--model", "gemini-3.1-pro-high", "--mode", "accept-edits",
             "--print-timeout", "24h", "--output-format", "stream-json"),
        ),
        ModelProfile(
            "gemini-3.6-flash-high",
            ("agy", "--dangerously-skip-permissions", "--effort", "high",
             "--model", "gemini-3.6-flash-high", "--mode", "accept-edits",
             "--print-timeout", "24h", "--output-format", "stream-json"),
        ),
        ModelProfile(
            "claude-sonnet-4-6",
            ("agy", "--dangerously-skip-permissions",
             "--model", "claude-sonnet-4-6", "--mode", "accept-edits",
             "--print-timeout", "24h", "--output-format", "stream-json"),
        ),
    )


def codex_profile() -> ModelProfile:
    return ModelProfile(
        "gpt-5.5-high",
        ("codex", "exec", "--ignore-user-config", "--ephemeral", "--json",
         "--sandbox", "workspace-write", "-c",
         "sandbox_workspace_write.network_access=true", "-c",
         'model_reasoning_effort="high"', "-m", "gpt-5.5"),
    )
```

Implement `launch_shell_command()` using `shlex.join()` for argv and
`shlex.quote()` for the prompt and paths. Agy commands prefix:

```text
HF_TOKEN="$(hf auth token)" GH_TOKEN="$(gh auth token)"
HF_HOME="/tmp/icml-agy-hf-XX" UV_CACHE_DIR="/tmp/icml-repro-uv-cache"
```

Codex receives the same `HF_TOKEN` and `GH_TOKEN` lookup expressions, followed
by `-C REPO_ROOT PROMPT`. Never resolve either token inside Python.

- [ ] **Step 4: Add and satisfy redaction tests**

```python
def test_sanitize_text_redacts_tokens_and_environment_assignments():
    raw = (
        "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz "
        "GH_TOKEN=github_pat_abcdefghijklmnopqrstuvwxyz "
        "Authorization: Bearer secret-value"
    )
    clean = supervisor.sanitize_text(raw)
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in clean
    assert "github_pat_abcdefghijklmnopqrstuvwxyz" not in clean
    assert "secret-value" not in clean
    assert clean.count("<redacted>") == 3
```

Implement compiled patterns for Hugging Face tokens, GitHub token prefixes,
and bearer headers. Keep lookup expressions such as `$(hf auth token)` intact.

- [ ] **Step 5: Run focused tests**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q tests/test_worker_supervisor.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add ops/__init__.py ops/worker_supervisor.py tests/test_worker_supervisor.py
git commit -m "feat: define direct paper-owner worker pool"
```

### Task 2: Tmux Health Adapter, Lock, and Atomic Runtime State

**Files:**
- Modify: `ops/worker_supervisor.py`
- Modify: `tests/test_worker_supervisor.py`

**Interfaces:**
- Consumes: `WorkerSpec` from Task 1.
- Produces: `HostAdapter.session_health(spec) -> SessionHealth`,
  `HostAdapter.ensure_session(spec, command) -> None`,
  `exclusive_lock(path)`, `atomic_write_json(path, value)`, and
  `runtime_directory() -> Path`.
- `SessionHealth` fields: `exists: bool`, `pane_dead: bool`,
  `foreground_command: str`, `recent_output: str`.

- [ ] **Step 1: Write failing health-classification tests**

```python
def test_only_expected_live_foreground_agent_is_healthy():
    agy = supervisor.desired_workers()[0]
    assert supervisor.is_healthy(
        agy, supervisor.SessionHealth(True, False, "agy", "")
    )
    assert not supervisor.is_healthy(
        agy, supervisor.SessionHealth(True, False, "bash", "")
    )
    assert not supervisor.is_healthy(
        agy, supervisor.SessionHealth(True, True, "agy", "")
    )
    assert not supervisor.is_healthy(
        agy, supervisor.SessionHealth(False, False, "", "")
    )
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  tests/test_worker_supervisor.py::test_only_expected_live_foreground_agent_is_healthy
```

Expected: FAIL because `SessionHealth` is undefined.

- [ ] **Step 3: Implement the subprocess-backed tmux adapter**

Use argument arrays and
`subprocess.run(argv, check=False, text=True, capture_output=True)` for:

```text
tmux list-panes -t SESSION -F
#{pane_dead}\t#{pane_current_command}\t#{pane_pid}
tmux capture-pane -pt SESSION -S -80
tmux new-session -d -s SESSION -c REPO_ROOT
tmux send-keys -t SESSION COMMAND C-m
```

`session_health()` maps tmux's missing-session exit into `exists=False`; all
other command failures raise `HostCommandError` with sanitized stderr.
`ensure_session()` creates only a missing session, then sends the launch
command. It reuses an existing idle shell rather than creating a duplicate.

- [ ] **Step 4: Add fake-adapter tests for adoption and session creation**

Create `FakeHost` in the test module with dictionaries for health and lists for
created/sent sessions. Verify:

```python
def test_ensure_session_adopts_live_process_and_reuses_idle_shell():
    fake_host = FakeHost()
    live, idle, missing = supervisor.desired_workers()[:3]
    fake_host.health[live.session_name] = supervisor.SessionHealth(
        True, False, "agy", ""
    )
    fake_host.health[idle.session_name] = supervisor.SessionHealth(
        True, False, "bash", ""
    )
    fake_host.health[missing.session_name] = supervisor.SessionHealth(
        False, False, "", ""
    )

    assert supervisor.is_healthy(live, fake_host.session_health(live))
    fake_host.ensure_session(idle, "exec agy")
    fake_host.ensure_session(missing, "exec agy")

    assert fake_host.created == [missing.session_name]
    assert fake_host.sent == [
        (idle.session_name, "exec agy"),
        (missing.session_name, "exec agy"),
    ]
```

Assert every invocation uses argument arrays and that no command diagnostics
contain token-shaped strings.

- [ ] **Step 5: Add lock and atomic-state tests**

```python
def test_atomic_status_replaces_complete_json(tmp_path):
    path = tmp_path / "status.json"
    supervisor.atomic_write_json(path, {"workers": [{"id": "one"}]})
    assert json.loads(path.read_text())["workers"] == [{"id": "one"}]
    assert list(tmp_path.glob(".status.json.*.tmp")) == []


def test_exclusive_lock_rejects_second_holder(tmp_path):
    with supervisor.exclusive_lock(tmp_path / "supervisor.lock"):
        with pytest.raises(supervisor.AlreadyRunning):
            with supervisor.exclusive_lock(tmp_path / "supervisor.lock"):
                pass
```

Implement atomic JSON with `tempfile.NamedTemporaryFile` in the target
directory, `flush()`, `os.fsync()`, `os.replace()`, and directory creation mode
`0o700`. Implement a nonblocking `fcntl.flock`.

- [ ] **Step 6: Run the test module**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q tests/test_worker_supervisor.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add ops/worker_supervisor.py tests/test_worker_supervisor.py
git commit -m "feat: inspect and persist supervised worker health"
```

### Task 3: Idempotent Reconciliation and Quota-Aware Backoff

**Files:**
- Modify: `ops/worker_supervisor.py`
- Modify: `tests/test_worker_supervisor.py`

**Interfaces:**
- Consumes: worker profiles, `HostAdapter`, locking, and status persistence.
- Produces: `RuntimeState`, `LaneState`, `parse_quota_reset()`,
  `next_profile()`, and
  `reconcile(host, state, now, repo_root, dry_run=False) -> ReconcileResult`.
- `RuntimeState.empty() -> RuntimeState` creates no lanes;
  `RuntimeState.with_lane(worker_id, *, profile_index=0,
  profile_backoff=None) -> RuntimeState` creates one testable lane.
- `LaneState` fields are `profile_index`, `profile_backoff`, `restart_count`,
  `ordinary_failures`, `next_retry_at`, and `last_error`.
- `ReconcileResult` fields are `state`, `status`, `started`, and `proposed`;
  worker ID collections use tuples.
- Runtime timestamps are UTC ISO-8601 strings; calculations use aware
  `datetime` objects.

- [ ] **Step 1: Write failing idempotence and adoption tests**

```python
def test_reconcile_does_not_restart_healthy_workers(fake_host, fixed_now):
    for spec in supervisor.desired_workers():
        fake_host.health[spec.session_name] = supervisor.SessionHealth(
            True, False, spec.agent, ""
        )
    result = supervisor.reconcile(
        fake_host, supervisor.RuntimeState.empty(), fixed_now, Path("/repo")
    )
    assert result.started == ()
    assert fake_host.sent == []
    assert len(result.status["workers"]) == 15


def test_two_reconciliations_do_not_duplicate_started_lane(fake_host, fixed_now):
    state = supervisor.RuntimeState.empty()
    first = supervisor.reconcile(fake_host, state, fixed_now, Path("/repo"))
    fake_host.mark_started(first.started)
    second = supervisor.reconcile(
        fake_host, first.state, fixed_now + timedelta(seconds=30), Path("/repo")
    )
    assert len(first.started) == 15
    assert second.started == ()
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  tests/test_worker_supervisor.py::test_reconcile_does_not_restart_healthy_workers \
  tests/test_worker_supervisor.py::test_two_reconciliations_do_not_duplicate_started_lane
```

Expected: FAIL because reconciliation types are undefined.

- [ ] **Step 3: Implement minimal reconciliation**

For every desired worker:

- Observe current `SessionHealth`.
- Return `healthy` without mutation when `is_healthy()` is true.
- Return `backed_off` without mutation when `now < next_retry_at`.
- Choose the persisted Agy profile index or the sole Codex profile.
- In dry-run mode, append a proposed action only.
- Otherwise call `ensure_session()`, increment `restart_count`, and set a
  15-second provisional retry guard.
- Serialize only worker ID, agent, health, model, restart count, next retry,
  and sanitized last error.

- [ ] **Step 4: Write failing quota parsing and fallback tests**

```python
def test_quota_error_rotates_profile_and_honors_reported_reset(fake_host, fixed_now):
    spec = supervisor.desired_workers()[0]
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True,
        False,
        "bash",
        "Individual quota reached. Resets in 46m20s.",
    )
    state = supervisor.RuntimeState.with_lane(spec.worker_id, profile_index=0)
    result = supervisor.reconcile(fake_host, state, fixed_now, Path("/repo"))
    lane = result.state.lanes[spec.worker_id]
    assert lane.profile_index == 1
    assert lane.profile_backoff["gemini-3.1-pro-high"] == (
        fixed_now + timedelta(minutes=46, seconds=20)
    )
    assert result.started == (spec.worker_id,)


def test_all_agy_profiles_exhausted_waits_for_earliest_reset(fake_host, fixed_now):
    spec = supervisor.desired_workers()[0]
    resets = {
        profile.name: fixed_now + timedelta(minutes=minutes)
        for profile, minutes in zip(supervisor.agy_profiles(), (12, 8, 20))
    }
    state = supervisor.RuntimeState.with_lane(
        spec.worker_id, profile_index=0, profile_backoff=resets
    )
    result = supervisor.reconcile(fake_host, state, fixed_now, Path("/repo"))
    lane = result.state.lanes[spec.worker_id]
    assert spec.worker_id not in result.started
    assert lane.next_retry_at == fixed_now + timedelta(minutes=8)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Individual quota reached. Resets in 50m33s.", timedelta(minutes=50, seconds=33)),
        ("Individual quota reached. Resets in 2h3m.", timedelta(hours=2, minutes=3)),
        ("process exited with code 1", None),
    ],
)
def test_parse_quota_reset_formats(message, expected, fixed_now):
    parsed = supervisor.parse_quota_reset(message, fixed_now)
    assert parsed == (None if expected is None else fixed_now + expected)


def test_ordinary_crash_uses_bounded_worker_specific_backoff(fake_host, fixed_now):
    spec = supervisor.desired_workers()[10]
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True, False, "bash", "process exited with code 1"
    )
    state = supervisor.RuntimeState.with_lane(spec.worker_id)
    result = supervisor.reconcile(fake_host, state, fixed_now, Path("/repo"))
    lane = result.state.lanes[spec.worker_id]
    assert lane.ordinary_failures == 1
    assert fixed_now + timedelta(seconds=15) <= lane.next_retry_at
    assert lane.next_retry_at <= fixed_now + timedelta(seconds=25)
```

- [ ] **Step 5: Implement quota and crash backoff**

Use:

```python
QUOTA_RE = re.compile(
    r"quota reached.*?Resets in "
    r"(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?",
    re.IGNORECASE | re.DOTALL,
)
```

On quota:

- Back off the active profile until the parsed reset.
- Select the first profile whose backoff is absent or expired.
- If none is available, do not launch and retry at the earliest reset.

On an ordinary exit, use `min(15 * 2**failures, 900)` seconds plus deterministic
per-worker jitter derived from `sha256(worker_id)` in the range 0–10 seconds.
Reset ordinary failure count after observing a healthy process.

- [ ] **Step 6: Test dry-run and sanitized status**

```python
def test_dry_run_reports_without_host_mutation_or_state_change(
    fake_host, fixed_now
):
    initial = supervisor.RuntimeState.empty()
    result = supervisor.reconcile(
        fake_host, initial, fixed_now, Path("/repo"), dry_run=True
    )
    assert len(result.proposed) == 15
    assert result.started == ()
    assert result.state == initial
    assert fake_host.created == []
    assert fake_host.sent == []


def test_status_contains_no_prompt_token_or_environment_dump(
    fake_host, fixed_now
):
    spec = supervisor.desired_workers()[0]
    fake_host.health[spec.session_name] = supervisor.SessionHealth(
        True,
        False,
        "bash",
        "HF_TOKEN=hf_secret GH_TOKEN=github_pat_secret",
    )
    result = supervisor.reconcile(
        fake_host, supervisor.RuntimeState.empty(), fixed_now, Path("/repo")
    )
    encoded = json.dumps(result.status)
    assert "Use the shared" not in encoded
    assert "HF_TOKEN" not in encoded
    assert "hf_secret" not in encoded
    assert "github_pat_" not in encoded
```

- [ ] **Step 7: Run reconciliation tests**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q tests/test_worker_supervisor.py
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add ops/worker_supervisor.py tests/test_worker_supervisor.py
git commit -m "feat: reconcile worker pools with quota backoff"
```

### Task 4: CLI, Systemd User Units, and Safe Installation

**Files:**
- Modify: `ops/worker_supervisor.py`
- Create: `ops/systemd/icml-worker-supervisor.service`
- Create: `ops/systemd/icml-worker-supervisor.timer`
- Modify: `tests/test_worker_supervisor.py`

**Interfaces:**
- Consumes: Task 3 reconciliation.
- Produces: `main(argv=None) -> int`, commands `reconcile`, `status`,
  `install`, `smoke-test`, and `stop --confirm`; systemd unit templates.

- [ ] **Step 1: Write failing CLI contract tests**

Use `main(["status"], host=fake_host, home=tmp_path, now=fixed_now)` as an
injectable test seam:

```python
def test_status_prints_compact_worker_counts(capsys, fake_host, tmp_path):
    code = supervisor.main(
        ["status"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )
    assert code == 0
    assert "agy 10/10" in capsys.readouterr().out
    assert "codex 5/5" in capsys.readouterr().out


def test_stop_requires_explicit_confirmation(fake_host, tmp_path):
    assert supervisor.main(["stop"], host=fake_host, home=tmp_path) == 2
    assert fake_host.stopped == []


def test_install_adopts_before_enabling_timer(fake_host, tmp_path):
    code = supervisor.main(
        ["install"], host=fake_host, home=tmp_path, now=FIXED_NOW
    )
    assert code == 0
    assert fake_host.events.index("reconcile") < fake_host.events.index(
        "systemctl --user enable --now icml-worker-supervisor.timer"
    )


def test_smoke_request_restores_only_disposable_session(fake_host, tmp_path):
    fake_host.production_pids = {
        spec.session_name: index
        for index, spec in enumerate(supervisor.desired_workers(), start=100)
    }
    code = supervisor.main(
        ["smoke-test", "--timeout", "1"],
        host=fake_host,
        home=tmp_path,
        now=FIXED_NOW,
    )
    assert code == 0
    assert fake_host.restored == ["icml-supervisor-smoke-test"]
    assert fake_host.stopped == ["icml-supervisor-smoke-test"]
    assert fake_host.production_pids == {
        spec.session_name: index
        for index, spec in enumerate(supervisor.desired_workers(), start=100)
    }
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  tests/test_worker_supervisor.py -k 'status_prints or stop_requires or install_adopts'
```

Expected: FAIL because the CLI commands are missing.

- [ ] **Step 3: Implement CLI and runtime state loading**

Use `argparse` subcommands. Default paths:

```python
state_dir = home / ".local/state/icml-worker-supervisor"
status_path = state_dir / "status.json"
runtime_path = state_dir / "runtime.json"
lock_path = state_dir / "supervisor.lock"
```

`reconcile` loads runtime state, acquires the lock, performs one reconciliation,
atomically writes runtime and status, and exits nonzero only on supervisor
failure—not merely because a provider is backed off.

`status` reads status JSON and prints aggregate counts plus one line per
degraded lane.

`stop --confirm` first disables the timer, then stops only tmux session names
present in `desired_workers()`. It does not kill unrelated tmux sessions.

`smoke-test --timeout 45` snapshots the 15 production pane PIDs, creates the
fixed disposable session `icml-supervisor-smoke-test` with foreground command
`/usr/bin/sleep 300`, interrupts only that disposable command, and writes a
strict smoke request containing only a random nonce and stage. The next timer
reconciliation recognizes that fixed request, restores `exec /usr/bin/sleep
300`, and atomically marks the request restored. The foreground CLI waits up to
the requested timeout, verifies the disposable command and unchanged production
PIDs, then removes only the disposable session and request. No request field is
used as a command.

- [ ] **Step 4: Create the systemd units**

`ops/systemd/icml-worker-supervisor.service`:

```ini
[Unit]
Description=Reconcile ICML direct paper-owner workers
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
WorkingDirectory=@REPO_ROOT@
ExecStart=@PYTHON@ @REPO_ROOT@/ops/worker_supervisor.py reconcile
```

`ops/systemd/icml-worker-supervisor.timer`:

```ini
[Unit]
Description=Continuously maintain ICML direct paper-owner workers

[Timer]
OnBootSec=10s
OnUnitActiveSec=30s
AccuracySec=2s
Persistent=true
Unit=icml-worker-supervisor.service

[Install]
WantedBy=timers.target
```

`install` substitutes absolute `@REPO_ROOT@` and the current Python executable,
writes to `~/.config/systemd/user/` atomically, runs an adoption reconciliation,
then invokes:

```text
systemctl --user daemon-reload
systemctl --user enable --now icml-worker-supervisor.timer
systemctl --user start icml-worker-supervisor.service
```

- [ ] **Step 5: Test exact unit rendering, command order, and smoke isolation**

Assert rendered units contain no placeholders, use absolute paths, set the
30-second interval, and contain no tokens. Assert the fake host records
reconciliation before `enable --now`. Assert a pending smoke request is
restored by `reconcile`, arbitrary request keys are rejected, production pane
PIDs are unchanged, timeout returns nonzero, and cleanup targets only
`icml-supervisor-smoke-test`.

- [ ] **Step 6: Run the full supervisor test module**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q tests/test_worker_supervisor.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add ops/worker_supervisor.py ops/systemd tests/test_worker_supervisor.py
git commit -m "feat: install continuous direct worker supervision"
```

### Task 5: Operations Documentation and Full Verification

**Files:**
- Modify: `docs/REMOTE_SETUP.md`
- Modify: `tests/test_worker_supervisor.py`

**Interfaces:**
- Consumes: the installed supervisor CLI and units.
- Produces: one documented, repeatable installation and smoke-test procedure.

- [ ] **Step 1: Write a failing documentation policy test**

```python
def test_remote_setup_documents_direct_supervisor_operations():
    text = Path("docs/REMOTE_SETUP.md").read_text()
    assert "ops/worker_supervisor.py install" in text
    assert "ops/worker_supervisor.py status" in text
    assert "ops/worker_supervisor.py reconcile --dry-run" in text
    assert "10 Agy" in text
    assert "5 Codex" in text
    assert "OpenCode" in text
```

- [ ] **Step 2: Run it and verify failure**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q \
  tests/test_worker_supervisor.py::test_remote_setup_documents_direct_supervisor_operations
```

Expected: FAIL because the operations section is absent.

- [ ] **Step 3: Document installation and operations**

Add a “Direct worker supervisor” section with:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py reconcile --dry-run
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py install
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py status
systemctl --user status icml-worker-supervisor.timer
```

State explicitly that it launches direct CLIs, targets 10 Agy and 5 Codex,
does not use OpenCode, does not mutate coordinator state, and preserves healthy
lanes.

Document the destructive boundary separately:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py stop --confirm
```

- [ ] **Step 4: Run focused and repository validation**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q tests/test_worker_supervisor.py
UV_CACHE_DIR=/tmp/icml-repro-uv-cache uv run pytest -q --ignore=submissions/nape
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
PRE_COMMIT_HOME=/tmp/icml-repro-pre-commit \
uv run pre-commit run -a
git diff --check
```

Expected: all commands exit 0. Do not run or format `submissions/nape/`.

- [ ] **Step 5: Run non-mutating host preflight**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py reconcile --dry-run
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py status
```

Expected: dry-run lists only missing or unhealthy lanes; status contains no
credential values or prompt text.

- [ ] **Step 6: Commit Task 5**

```bash
git add docs/REMOTE_SETUP.md tests/test_worker_supervisor.py
git commit -m "docs: add direct worker supervisor operations"
```

### Task 6: Install and Prove Continuous Recovery

**Files:**
- Runtime only: `~/.config/systemd/user/icml-worker-supervisor.*`
- Runtime only: `~/.local/state/icml-worker-supervisor/`

**Interfaces:**
- Consumes: completed and verified Tasks 1–5.
- Produces: active user timer and fresh operational evidence.

- [ ] **Step 1: Capture current healthy sessions**

Run:

```bash
for n in 01 02 03 04 05 06 07 08 09 10; do
  tmux list-panes -t "agy-paper-owner-$n" \
    -F "agy-paper-owner-$n #{pane_pid} #{pane_dead} #{pane_current_command}"
done
for n in 01 02 03 04 05; do
  tmux list-panes -t "codex-paper-owner-$n" \
    -F "codex-paper-owner-$n #{pane_pid} #{pane_dead} #{pane_current_command}"
done
```

Record which lanes are healthy. Do not stop them.

- [ ] **Step 2: Install and immediately verify the timer**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py install
systemctl --user is-active icml-worker-supervisor.timer
systemctl --user list-timers icml-worker-supervisor.timer --no-pager
```

Expected: timer is `active` with a next activation within 30 seconds.

- [ ] **Step 3: Verify all desired foreground processes**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py status
```

Expected: `agy 10/10`, `codex 5/5`, or an explicitly reported provider
backoff with a retry timestamp. Every lane counted healthy has foreground
command exactly `agy` or `codex`.

- [ ] **Step 4: Prove timer recovery using the disposable smoke session**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py smoke-test --timeout 45
```

Expected: the later reconciliation records exactly one restart and does not
alter any of the 15 production pane PIDs captured in Step 1.

- [ ] **Step 5: Recheck production adoption and credential hygiene**

Run:

```bash
UV_CACHE_DIR=/tmp/icml-repro-uv-cache \
  uv run python ops/worker_supervisor.py status
systemctl --user status icml-worker-supervisor.service --no-pager
```

Compare healthy pre-install pane PIDs with current pane PIDs. Expected: adopted
healthy workers retain their PIDs; systemd and status output contain no token
values, prompts, or environment dumps.

- [ ] **Step 6: Record final implementation commit if installation exposed no changes**

No repository commit is needed for runtime-only installation. If the smoke test
reveals a code defect, return to the owning task, write a failing regression
test, implement the minimal fix, rerun Task 5 verification, and commit the
specific fix before repeating installation.
