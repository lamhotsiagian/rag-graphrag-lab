"""Real end-to-end interactive Playwright test for Chapter 3: Embeddings & Vector Database."""

import re
from pathlib import Path
from playwright.sync_api import Page, expect

SAMPLE_FILE = str(Path(__file__).parent.parent / "test_data" / "sample.txt")


def test_chapter_03_full_flow(page: Page, chapter_03_app):
    """Test document indexing into pgvector and semantic vector search query."""
    page.goto(chapter_03_app)
    page.wait_for_selector(".stApp", timeout=15000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
        timeout=30000,
    )

    # 1. Upload document
    file_input = page.locator("input[type='file']")
    file_input.set_input_files(SAMPLE_FILE)
    expect(page.locator(".stApp")).to_contain_text("sample.txt", timeout=15000)

    # 2. Click Index Documents
    index_btn = page.get_by_role("button", name="Index Documents")
    index_btn.click()

    # 3. Verify success message
    expect(page.locator(".stApp")).to_contain_text("indexed successfully", timeout=45000)

    # 4. Fill search query
    query_input = page.locator("div[data-testid='stTextInput'] input")
    query_input.fill("vector embeddings and semantic search")

    # 5. Click Search
    search_btn = page.get_by_role("button", name="Search")
    search_btn.click()

    # 6. Verify search result cards with similarity score
    expect(page.locator(".stApp")).to_contain_text("Score:", timeout=30000)
