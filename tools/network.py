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


def crawl_page(
    url: str,
    extract_depth: str = "basic",
    format: str = "text",
    include_images: bool = False,
):
    """Crawl a specific URL and extract its content.

    Args:
        url: The target URL to crawl.
        extract_depth: Extraction depth, "basic" or "advanced".
        format: Output format, "text" or "html".
        include_images: Whether to include images in the result.
    """
    return tavily_client.extract(
        urls=[url],
        extract_depth=extract_depth,
        format=format,
        include_images=include_images,
    )