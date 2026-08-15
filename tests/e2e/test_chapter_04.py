"""Real end-to-end interactive Playwright test for Chapter 4: Complete Basic RAG."""

import re
from pathlib import Path
from playwright.sync_api import Page, expect

SAMPLE_FILE = str(Path(__file__).parent.parent / "test_data" / "sample.md")


def test_chapter_04_full_flow(page: Page, chapter_04_app):
    """Test full basic RAG pipeline: document ingestion, index stats, chat query, and sources display."""
    page.goto(chapter_04_app)
    page.wait_for_selector(".stApp", timeout=15000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
        timeout=30000,
    )

    # 1. Upload document
    file_input = page.locator("input[type='file']")
    file_input.set_input_files(SAMPLE_FILE)

    # 2. Click Index Documents
    index_btn = page.get_by_role("button", name="Index Documents")
    index_btn.click()

    # 3. Verify success message
    expect(page.locator(".stApp")).to_contain_text("Documents indexed!", timeout=45000)

    # 4. Fill chat input and submit question
    chat_input = page.locator("textarea[data-testid='stChatInputTextArea']")
    chat_input.fill("What is the main topic of the uploaded document?")
    chat_input.press("Enter")

    # 5. Verify response generated and sources expander present
    expect(page.locator(".stChatMessage").nth(1)).to_be_visible(timeout=60000)
    expect(page.get_by_text("Sources")).to_be_visible(timeout=30000)
