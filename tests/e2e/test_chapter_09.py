"""Real end-to-end interactive Playwright test for Chapter 9: Agentic GraphRAG."""

import re
from pathlib import Path
from playwright.sync_api import Page, expect

SAMPLE_FILE = str(Path(__file__).parent.parent / "test_data" / "sample.txt")


def test_chapter_09_full_flow(page: Page, chapter_09_app):
    """Test Agentic GraphRAG LangGraph workflow: query classification, multi-step reasoning, and response verification."""
    page.goto(chapter_09_app)
    page.wait_for_selector(".stApp", timeout=15000)
    page.wait_for_function(
        "() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
        timeout=30000,
    )

    # 1. Upload file and ingest
    file_input = page.locator("input[type='file']")
    file_input.set_input_files(SAMPLE_FILE)

    ingest_btn = page.get_by_role("button", name="Ingest")
    ingest_btn.click()

    expect(page.locator(".stApp")).to_contain_text("Ingestion successful", timeout=15000)

    # 2. Submit question in Agent Chat
    chat_input = page.locator("textarea[data-testid='stChatInputTextArea']")
    chat_input.fill("How does Agentic RAG decide which retriever to use?")
    chat_input.press("Enter")

    # 3. Verify Assistant answer and confidence progress bar
    expect(page.locator(".stChatMessage").nth(1)).to_be_visible(timeout=60000)
    expect(page.get_by_text("Reasoning Steps")).to_be_visible(timeout=30000)

    # 4. Check Workflow Visualization tab
    page.get_by_role("tab", name="Workflow Visualization").click()
    expect(page.locator(".stApp")).to_contain_text("classify_query", timeout=15000)

    # 5. Check Agent Decisions tab
    page.get_by_role("tab", name="Agent Decisions").click()
    expect(page.get_by_text("Decision logs")).to_be_visible(timeout=15000)
