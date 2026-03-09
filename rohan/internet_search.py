"""
internet_search.py — DuckDuckGo web search (no API key)
"""
DDGS_OK = False
try:
    from ddgs import DDGS
    DDGS_OK = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_OK = True
    except ImportError:
        pass

def search_web(query: str, max_results: int = 4):
    """Return list of {title, body, url} dicts. Empty list on failure."""
    if not DDGS_OK:
        return []
    try:
        with DDGS() as d:
            results = d.text(query + " medicine drug", max_results=max_results)
            return [{"title": r.get("title",""), "body": r.get("body",""), "url": r.get("href","")}
                    for r in results if r.get("body")]
    except Exception as e:
        print(f"  [web] DDG error: {e}")
        return []

def web_context_text(results):
    return "\n".join(f"- {r['title']}: {r['body'][:200]}" for r in results)
