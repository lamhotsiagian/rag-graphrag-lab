"""Real end-to-end interactive Playwright test for Chapter 7: Knowledge Graph Fundamentals."""

import re
from pathlib import Path
from playwright.sync_api import Page, expect

SAMPLE_FILE = str(Path(__file__).parent.parent / "test_data" / "sample.txt")


def test_chapter_07_full_flow(page: Page, chapter_07_app):
    """Test building Knowledge Graph in Neo4j, inspecting entities, and running Cypher query."""
    page.goto(chapter_07_app)
    page.wait_for_selector(".stApp", timeout=15000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
        timeout=30000,
    )

    # 1. Upload file
    file_input = page.locator("input[type='file']")
    file_input.set_input_files(SAMPLE_FILE)
    expect(page.locator(".stApp")).to_contain_text("sample.txt", timeout=15000)

    # 2. Click Build Knowledge Graph
    build_btn = page.get_by_role("button", name="Build Knowledge Graph")
    build_btn.click()

    # 3. Verify success message
    expect(page.locator(".stApp")).to_contain_text("Graph built successfully!", timeout=60000)

    # 4. Check Entity Explorer tab
    page.get_by_role("tab", name="Entity Explorer").click()
    expect(page.get_by_role("tab", name="Entity Explorer")).to_be_visible()

    # 5. Check Cypher Query tab
    page.get_by_role("tab", name="Cypher Query").click()
    run_btn = page.get_by_role("button", name="Run Query")
    expect(run_btn).to_be_visible()
    run_btn.click()
    expect(page.locator(".stApp")).to_contain_text("MATCH", timeout=15000)
