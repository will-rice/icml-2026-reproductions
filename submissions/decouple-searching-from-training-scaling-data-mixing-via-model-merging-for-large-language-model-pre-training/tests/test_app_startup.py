import os
from pathlib import Path
import signal
import subprocess
import sys
import time


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
SPACE_PORT = 7860


def _listening_ipv4_addresses(pid: int, port: int) -> set[str]:
    addresses = set()
    port_hex = f"{port:04X}"

    with Path(f"/proc/{pid}/net/tcp").open() as tcp:
        next(tcp)
        for line in tcp:
            fields = line.split()
            address, listening_port = fields[1].split(":")
            if listening_port == port_hex and fields[3] == "0A":
                addresses.add(address)

    return addresses


def test_space_app_listens_on_docker_interface():
    environment = os.environ.copy()
    environment["GRADIO_ANALYTICS_ENABLED"] = "False"
    process = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=SUBMISSION_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    try:
        deadline = time.monotonic() + 15
        addresses = set()
        while process.poll() is None and time.monotonic() < deadline:
            addresses = _listening_ipv4_addresses(process.pid, SPACE_PORT)
            if addresses:
                break
            time.sleep(0.1)

        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise AssertionError(f"Space app exited during startup:\n{output}")

        assert "00000000" in addresses, (
            f"Space port {SPACE_PORT} is not listening on 0.0.0.0: "
            f"{sorted(addresses)}"
        )
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
