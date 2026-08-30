"""End-to-end UI test. Requires the compose stack running (frontend on :3000,
backend reachable through the nginx proxy). Run with:

    pytest tests/test_ui_e2e.py -m e2e
"""

import os

import pytest
from playwright.sync_api import expect, sync_playwright

BASE_URL = os.environ.get("BASE_URL", "http://localhost:3000")


@pytest.mark.e2e
def test_classify_tweet_shows_a_label():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(BASE_URL)

        page.fill("#tweet-input", "Massive wildfire forces evacuation of the entire town")
        page.click("#classify-btn")

        result = page.locator("#result")
        expect(result).to_be_visible(timeout=10_000)

        label = page.locator("#label").get_attribute("data-label")
        assert label in ("disaster", "not disaster")

        confidence = page.locator("#confidence").inner_text()
        assert confidence.endswith("%")

        browser.close()
