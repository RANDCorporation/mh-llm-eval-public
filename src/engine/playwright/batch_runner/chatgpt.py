# Copyright (C) 2026 RAND Corporation. MIT License — see LICENSE file.
"""Batch runner: mental health prompts × iterations against ChatGPT consumer UI.

Usage:
    python engine/playwright/batch_runner/chatgpt.py --run-id chatgpt_01 --model-id gpt-5.3

    Optional flags:
      --port N            Chrome debug port (default: 9222)
      --iterations N      Repetitions per prompt (default: 100)
      --iteration-start N First iteration number (for multi-account splits)
      --dry-run           Print the iteration plan without running
      --limit N           Stop after N executions (smoke test)
      --convert-only      Convert existing JSONL checkpoint to parquet (no browser)

Output:
    output/responses/<run_id>.parquet
    output/responses/.checkpoints/<run_id>.jsonl

Prerequisites:
    Close all Chrome windows first, then:
        pip install playwright pyyaml pandas pyarrow
        playwright install chromium
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from src.engine.playwright.batch_runner.base import (
    PlaywrightRunner, MAX_WAIT_SECONDS, SETTLE_SECONDS,
)


class ChatGPTRunner(PlaywrightRunner):
    URL = "https://chatgpt.com"
    DEBUG_PORT = 9222
    DEFAULT_PROVIDER = "chatgpt_ui"
    PROFILE_PREFIX = "chatgpt"
    RATE_LIMIT_PHRASES = [
        "you've hit the limit for messages",
        "hit the limit for messages",
        "wait for your limit to reset",
        "want to keep using gpt-5",
        "upgrade your plan to keep",
    ]
    RATE_LIMIT_WAIT_SECONDS = 300

    def find_input(self, page):
        candidates = [
            page.locator("#prompt-textarea"),
            page.locator('div[contenteditable="true"]').first,
            page.get_by_placeholder("Message ChatGPT").first,
            page.get_by_placeholder("Ask anything").first,
            page.locator("textarea").first,
        ]
        for loc in candidates:
            try:
                loc.wait_for(state="visible", timeout=10_000)
                return loc
            except Exception:
                continue
        raise RuntimeError("Could not find the ChatGPT message input.")

    def send_message(self, page, input_loc, text):
        input_loc.click()
        input_loc.fill(text)
        time.sleep(0.4)
        try:
            page.get_by_test_id("send-button").click(timeout=3_000)
        except Exception:
            try:
                page.locator('button[aria-label="Send prompt"]').click(timeout=3_000)
            except Exception:
                input_loc.press("Enter")

    def wait_for_response(self, page):
        deadline = time.time() + MAX_WAIT_SECONDS
        last_text = ""
        stable_since = None

        while time.time() < deadline:
            text = self.get_response_text(page)
            if text and text == last_text:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= SETTLE_SECONDS:
                    return
            else:
                stable_since = None
                last_text = text
            time.sleep(1.0)

    def get_response_text(self, page):
        # Safety block takes priority
        try:
            body = page.evaluate("() => document.body.innerText")
            if body and "shown for safety reasons" in body.lower():
                return "This content couldn't be shown for safety reasons."
        except Exception:
            pass

        for sel in [
            '[data-message-author-role="assistant"] .markdown',
            '[data-message-author-role="assistant"]',
            '.agent-turn .markdown',
            '.agent-turn',
        ]:
            elements = page.locator(sel)
            count = elements.count()
            if count > 0:
                texts = []
                for i in range(count):
                    try:
                        t = elements.nth(i).inner_text().strip()
                        if t:
                            texts.append(t)
                    except Exception:
                        continue
                if texts:
                    return "\n\n---\n\n".join(texts)

        # Fallback: main content area
        try:
            main_el = page.locator("main").first
            full_text = main_el.inner_text().strip()
            skip_phrases = [
                "you said:", "chatgpt said:", "chatgpt can make mistakes",
                "check important info", "ask anything",
                "log in", "sign up for free", "new chat",
            ]
            lines = [
                line.strip() for line in full_text.split("\n")
                if line.strip()
                and not any(sp in line.strip().lower() for sp in skip_phrases)
            ]
            return "\n".join(lines) if lines else ""
        except Exception:
            return ""

    def get_crisis_banner_text(self, page):
        for sel in ['[data-testid="crisis-resource"]', '[class*="crisis"]', '[class*="safety"]']:
            try:
                el = page.locator(sel).first
                if el.is_visible(timeout=2_000):
                    return el.inner_text().strip()
            except Exception:
                continue
        try:
            el = page.get_by_text("Help is available").first
            if el.is_visible(timeout=1_000):
                return el.locator("..").first.inner_text().strip()
        except Exception:
            pass
        return ""

    def setup_instructions(self, model_id):
        return [
            "1) Handle any CAPTCHAs in the Chrome window.",
            "2) Log in if you want (or stay logged out).",
            "3) Make sure you see the ChatGPT chat input.",
        ]


if __name__ == "__main__":
    ChatGPTRunner().run()
