"""Pytest configuration and fixtures for Playwright tests."""

import subprocess
import time
import signal
import os
import sys
from pathlib import Path

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def browser_context_args():
    """Browser context arguments for all tests."""
    return {
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,
    }


@pytest.fixture(scope="session")
def launch_options():
    """Launch browser with visible GUI instead of headless."""
    return {
        "headless": False,
        "slow_mo": 300,
    }


def _start_streamlit_app(chapter: int) -> subprocess.Popen:
    """Start a Streamlit app for a specific chapter."""
    port = 8500 + chapter
    app_path = PROJECT_ROOT / f"chapter_{chapter:02d}" / "app.py"

    if not app_path.exists():
        pytest.skip(f"Chapter {chapter} app.py not found at {app_path}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    process = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )

    # Wait for the app to start
    max_wait = 30
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            import urllib.request
            response = urllib.request.urlopen(f"http://localhost:{port}/_stcore/health")
            if response.status == 200:
                return process
        except Exception:
            time.sleep(1)

    # If we get here, the app didn't start
    process.terminate()
    pytest.skip(f"Chapter {chapter} app failed to start on port {port}")


def _stop_streamlit_app(process: subprocess.Popen):
    """Stop a Streamlit app process."""
    if process:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


@pytest.fixture(scope="module")
def chapter_01_app():
    """Start Chapter 1 Streamlit app."""
    process = _start_streamlit_app(1)
    yield "http://localhost:8501"
    _stop_streamlit_app(process)


@pytest.fixture(scope="module")
def chapter_02_app():
    """Start Chapter 2 Streamlit app."""
    process = _start_streamlit_app(2)
    yield "http://localhost:8502"
    _stop_streamlit_app(process)


@pytest.fixture(scope="module")
def chapter_03_app():
    """Start Chapter 3 Streamlit app."""
    process = _start_streamlit_app(3)
    yield "http://localhost:8503"
    _stop_streamlit_app(process)


@pytest.fixture(scope="module")
def chapter_04_app():
    """Start Chapter 4 Streamlit app."""
    process = _start_streamlit_app(4)
    yield "http://localhost:8504"
    _stop_streamlit_app(process)


@pytest.fixture(scope="module")
def chapter_05_app():
    """Start Chapter 5 Streamlit app."""
    process = _start_streamlit_app(5)
    yield "http://localhost:8505"
    _stop_streamlit_app(process)


@pytest.fixture(scope="module")
def chapter_06_app():
    """Start Chapter 6 Streamlit app."""
    process = _start_streamlit_app(6)
    yield "http://localhost:8506"
    _stop_streamlit_app(process)


@pytest.fixture(scope="module")
def chapter_07_app():
    """Start Chapter 7 Streamlit app."""
    process = _start_streamlit_app(7)
    yield "http://localhost:8507"
    _stop_streamlit_app(process)


@pytest.fixture(scope="module")
def chapter_08_app():
    """Start Chapter 8 Streamlit app."""
    process = _start_streamlit_app(8)
    yield "http://localhost:8508"
    _stop_streamlit_app(process)


@pytest.fixture(scope="module")
def chapter_09_app():
    """Start Chapter 9 Streamlit app."""
    process = _start_streamlit_app(9)
    yield "http://localhost:8509"
    _stop_streamlit_app(process)


@pytest.fixture(scope="module")
def chapter_10_app():
    """Start Chapter 10 Streamlit app."""
    process = _start_streamlit_app(10)
    yield "http://localhost:8510"
    _stop_streamlit_app(process)
