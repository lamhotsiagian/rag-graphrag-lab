"""Real end-to-end interactive Playwright test for Chapter 6: RAG Evaluation Dashboard."""

import re
from playwright.sync_api import Page, expect


def test_chapter_06_full_flow(page: Page, chapter_06_app):
    """Test running evaluation pipeline over golden dataset and verifying metrics dashboard."""
    page.goto(chapter_06_app)
    page.wait_for_selector(".stApp", timeout=15000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
        timeout=30000,
    )

    # 1. Click Run Evaluation
    eval_btn = page.get_by_role("button", name="Run Evaluation")
    expect(eval_btn).to_be_visible()
    eval_btn.click()

    # 2. Verify completion success message
    expect(page.locator(".stApp")).to_contain_text("Evaluation complete!", timeout=60000)

    # 3. Verify Overview metrics cards
    expect(page.locator(".stApp")).to_contain_text("Avg Precision@3")
    expect(page.locator(".stApp")).to_contain_text("Avg Faithfulness")

    # 4. Check Per-Query Results tab
    page.get_by_text("Per-Query Results").click()
    expect(page.locator(".stApp")).to_contain_text("Expected Answer:", timeout=15000)

    # 5. Check Metrics Charts tab
    page.get_by_text("Metrics Charts").click()
    expect(page.get_by_text("Precision@3 per Query")).to_be_visible(timeout=15000)
    expect(page.locator(".js-plotly-plot").first).to_be_visible()
