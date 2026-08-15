"""Real end-to-end interactive Playwright test for Chapter 10: Production System Design."""

import re
from playwright.sync_api import Page, expect


def test_chapter_10_full_flow(page: Page, chapter_10_app):
    """Test Production System Design dashboard: Architecture navigation, reliability patterns, and interview practice evaluation."""
    page.goto(chapter_10_app)
    page.wait_for_selector(".stApp", timeout=15000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
        timeout=30000,
    )

    # 1. Check Architecture section
    expect(page.get_by_text("System Architecture")).to_be_visible()

    # 2. Check Scalability section
    page.get_by_text("Scalability", exact=True).click()
    expect(page.get_by_text("Scalability Patterns")).to_be_visible()

    # 3. Check Reliability section
    page.get_by_text("Reliability", exact=True).click()
    expect(page.get_by_role("button", name="Test Circuit Breaker")).to_be_visible()

    # 4. Check Interview Practice section
    page.get_by_text("Interview Practice", exact=True).click()
    expect(page.get_by_text("System Design Interview Practice")).to_be_visible()

    # 5. Fill design answer and click Evaluate
    answer_input = page.locator("textarea").first
    expect(answer_input).to_be_visible()
    answer_input.fill("We design a multi-tiered RAG architecture using Redis semantic cache for query caching, PostgreSQL with pgvector for dense retrieval, and exponential backoff retry logic for LLM resilience.")

    eval_btn = page.get_by_role("button", name="Evaluate")
    eval_btn.click()

    # 6. Verify evaluation feedback
    expect(page.locator(".stApp")).to_contain_text("Evaluation placeholder", timeout=15000)
