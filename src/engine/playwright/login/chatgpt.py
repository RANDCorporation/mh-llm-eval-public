# Copyright (C) 2026 RAND Corporation. MIT License — see LICENSE file.
"""One-time login capture for ChatGPT.

Opens Chrome with a persistent profile so you can log in manually.
The login state is saved and reused by the batch runner.

Usage:
    python engine/playwright/login/chatgpt.py
    # Log in to chatgpt.com in the Chrome window that opens, then close it.
"""

import os
import subprocess
import sys
import platform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from src.engine.playwright.batch_runner.chatgpt import ChatGPTRunner

PORT = 9222

runner = ChatGPTRunner()
chrome_path = runner.get_chrome_path()
if not chrome_path:
    print("ERROR: Google Chrome not found.")
    sys.exit(1)

profile_dir = os.path.join(os.getcwd(), ".chrome", f"chatgpt_{PORT}")
os.makedirs(profile_dir, exist_ok=True)

print(f"\nOpening Chrome for ChatGPT login...")
print(f"Profile: {profile_dir}")
print(f"  1. Log in to https://chatgpt.com in the Chrome window.")
print(f"  2. Close the window when done.")
print(f"  3. Your session will be saved for batch runs.\n")

subprocess.call([
    chrome_path,
    f"--remote-debugging-port={PORT}",
    f"--user-data-dir={profile_dir}",
    "--no-first-run",
    "--no-default-browser-check",
    "https://chatgpt.com",
])

print("Login session saved.")
