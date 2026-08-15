"""Real end-to-end interactive Playwright test for Chapter 1: RAG Foundations."""

import re
from pathlib import Path
from playwright.sync_api import Page, expect

SAMPLE_FILE = str(Path(__file__).parent.parent / "test_data" / "sample.txt")


def test_chapter_01_full_flow(page: Page, chapter_01_app):
    """Test full flow: page load, document upload, indexing, and chat interaction."""
    page.goto(chapter_01_app)
    page.wait_for_selector(".stApp", timeout=15000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
        timeout=30000,
    )

    # 1. Verify title
    expect(page).to_have_title(re.compile("Chapter 1"))

    # 2. Upload document
    file_input = page.locator("input[type='file']")
    file_input.set_input_files(SAMPLE_FILE)

    # 3. Process document
    process_btn = page.get_by_role("button", name=re.compile("Process Document"))
    expect(process_btn).to_be_visible()
    process_btn.click()

    # 4. Verify processing success message
    expect(page.locator(".stApp")).to_contain_text("Document processed into", timeout=30000)

    # 5. Fill chat input and submit question
    chat_input = page.locator("textarea[data-testid='stChatInputTextArea']")
    expect(chat_input).to_be_visible()
    chat_input.fill("What is Retrieval-Augmented Generation?")
    chat_input.press("Enter")

    # 6. Verify assistant response and sources display
    expect(page.locator(".stChatMessage").nth(1)).to_be_visible(timeout=60000)
    expect(page.locator(".stApp")).to_contain_text("Source 1", timeout=30000)
