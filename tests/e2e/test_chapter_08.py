"""Real end-to-end interactive Playwright test for Chapter 8: GraphRAG & Hybrid Retrieval."""

import re
from pathlib import Path
from playwright.sync_api import Page, expect

SAMPLE_FILE = str(Path(__file__).parent.parent / "test_data" / "sample.txt")


def test_chapter_08_full_flow(page: Page, chapter_08_app):
    """Test ingestion into vector and graph DBs, hybrid GraphRAG chat, and mode comparison."""
    page.goto(chapter_08_app)
    page.wait_for_selector(".stApp", timeout=15000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
        timeout=30000,
    )

    # 1. Upload file
    file_input = page.locator("input[type='file']")
    file_input.set_input_files(SAMPLE_FILE)
    expect(page.locator(".stApp")).to_contain_text("sample.txt", timeout=15000)

    # 2. Click Ingest Documents
    ingest_btn = page.get_by_role("button", name="Ingest Documents")
    ingest_btn.click()

    # 3. Verify success message
    expect(page.locator(".stApp")).to_contain_text("Ingestion complete!", timeout=60000)

    # 4. Fill chat input and submit
    chat_input = page.locator("textarea[data-testid='stChatInputTextArea']")
    chat_input.fill("What is GraphRAG?")
    chat_input.press("Enter")

    # 5. Verify assistant response generated
    expect(page.locator(".stChatMessage").nth(1)).to_be_visible(timeout=60000)

    # 6. Test Compare Modes tab
    page.get_by_role("tab", name="Compare Modes").click()
    compare_input = page.locator("div[data-testid='stTextInput'] input")
    compare_input.fill("What is knowledge graph?")

    run_comp_btn = page.get_by_role("button", name="Run Comparison")
    run_comp_btn.click()

    expect(page.locator(".stApp")).to_contain_text("Basic RAG", timeout=60000)
    expect(page.locator(".stApp")).to_contain_text("Hybrid GraphRAG", timeout=60000)
