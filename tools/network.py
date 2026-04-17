import os
from typing import Literal
from tavily import TavilyClient


tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


def crawl_page(url: str, max_depth: int = 1, include_raw_content: bool = True):
    """Crawl a specific URL and extract its content.

    Args:
        url: The target URL to crawl.
        max_depth: Maximum crawl depth (1 = only the given URL).
        include_raw_content: Whether to include raw HTML content.
    """
    return tavily_client.crawl(
        url,
        max_depth=max_depth,
        include_raw_content=include_raw_content,
    )