"""Real end-to-end interactive Playwright test for Chapter 2: Document Ingestion & Chunking."""

import re
from pathlib import Path
from playwright.sync_api import Page, expect

SAMPLE_FILE = str(Path(__file__).parent.parent / "test_data" / "sample.txt")


def test_chapter_02_full_flow(page: Page, chapter_02_app):
    """Test document upload, strategy processing, chunk inspection and statistics charts."""
    page.goto(chapter_02_app)
    page.wait_for_selector(".stApp", timeout=15000)
    expect(page.locator(".stApp")).to_contain_text("Chunking Strategies", timeout=45000)

    # 1. Upload document
    file_input = page.locator("input[type='file']")
    file_input.set_input_files(SAMPLE_FILE)
    expect(page.locator(".stApp")).to_contain_text("sample.txt", timeout=15000)

    # 2. Click Process
    process_btn = page.get_by_role("button", name="Process")
    process_btn.click()

    # 3. Verify success message
    expect(page.locator(".stApp")).to_contain_text("Processing complete!", timeout=30000)

    # 4. Check Compare Strategies tab
    page.get_by_role("tab", name="Compare Strategies").click()
    expect(page.locator(".stApp")).to_contain_text("Fixed Size:")
    expect(page.locator(".stApp")).to_contain_text("Recursive:")

    # 5. Check Chunk Inspector tab
    page.get_by_role("tab", name="Chunk Inspector").click()
    expect(page.get_by_text("Inspect Chunks")).to_be_visible()
    expect(page.locator(".stApp")).to_contain_text("Chunk 1")

    # 6. Check Statistics tab
    page.get_by_role("tab", name="Statistics").click()
    expect(page.get_by_text("Number of Chunks per Strategy")).to_be_visible(timeout=15000)
    expect(page.locator(".js-plotly-plot").first).to_be_visible()
