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
