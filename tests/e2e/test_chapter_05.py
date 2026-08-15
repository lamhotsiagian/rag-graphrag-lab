"""Real end-to-end interactive Playwright test for Chapter 5: Advanced Retrieval & Hybrid RAG."""

import re
from pathlib import Path
from playwright.sync_api import Page, expect

SAMPLE_FILE = str(Path(__file__).parent.parent / "test_data" / "sample.txt")


def test_chapter_05_full_flow(page: Page, chapter_05_app):
    """Test indexing, strategy selection, chat retrieval, and side-by-side strategy comparison."""
    page.goto(chapter_05_app)
    page.wait_for_selector(".stApp", timeout=15000)
    expect(page.get_by_text("Advanced Retrieval")).to_be_visible(timeout=30000)

    # 1. Upload file and Index
    file_input = page.locator("input[type='file']")
    file_input.set_input_files(SAMPLE_FILE)
    expect(page.locator(".stApp")).to_contain_text("sample.txt", timeout=15000)
    page.wait_for_timeout(1000)

    index_btn = page.get_by_role("button", name="Index")
    index_btn.click()

    expect(page.locator(".stApp")).to_contain_text("Successfully indexed", timeout=60000)

    # 2. Test Chat tab with Hybrid retrieval
    chat_input = page.locator("textarea[data-testid='stChatInputTextArea']")
    chat_input.fill("What is Retrieval Augmented Generation?")
    chat_input.press("Enter")

    expect(page.locator(".stChatMessage").nth(1)).to_be_visible(timeout=60000)
    expect(page.locator(".stApp")).to_contain_text("Retrieved", timeout=30000)

    # 3. Test Compare tab
    page.get_by_role("tab", name="Compare").click()
    query_input = page.locator("div[data-testid='stTextInput'] input")
    query_input.fill("What are machine learning types?")

    compare_btn = page.get_by_role("button", name="Compare")
    compare_btn.click()

    expect(page.locator(".stApp")).to_contain_text("Comparing all strategies", timeout=15000)
