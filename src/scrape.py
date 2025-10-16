import openai
from playwright.sync_api import sync_playwright, Page
from sitemap_utils import get_filtered_sitemap_urls
from selectolax.parser import HTMLParser
from dotenv import load_dotenv

from src.llm import categorize_urls_with_llm, extract_structured_data


def prune_html_for_llm(html_content: str, keywords: list[str] = None) -> str:
    """
    Prunes HTML content to only the most relevant sections for an LLM.
    This significantly reduces token count and improves accuracy.
    """
    tree = HTMLParser(html_content)

    # Step 1: Isolate the main content container
    main_container = None

    # Try finding the <main> tag first
    if tree.body:
        main_container = tree.body.css_first('main')

    # If no <main>, try finding a div/section with role="main"
    if not main_container and tree.body:
        main_container = tree.body.css_first('[role="main"]')

    # Fallback: if keywords are provided, find the best container holding them
    if not main_container and keywords and tree.body:
        best_candidate = None
        max_score = 0
        for node in tree.body.css('div, section'):
            score = sum(node.text(deep=True).lower().count(kw) for kw in keywords)
            if score > max_score:
                max_score = score
                best_candidate = node
        if max_score > 0:
            main_container = best_candidate

    # If we found a specific container, use it. Otherwise, use the whole body.
    root_node = main_container if main_container else tree.body

    if not root_node:
        return ""  # Return empty string if no body or content found

    # Step 2: Remove noise tags from the chosen container
    noise_tags = ['script', 'style', 'svg', 'nav', 'footer', 'header']
    for tag in noise_tags:
        for node in root_node.css(tag):
            node.decompose()  # decompose() removes the node and its children

    # Return the clean HTML of the container
    return root_node.html


pages_to_scrape = [
    "https://www.bioritmo.com.pe/",
    "https://limayoga.com/",
    "https://www.nasceyoga.com/",
    "https://www.zendayoga.com/",
    "https://matmax.world/",
    "https://anjali.pe/",
    "https://www.purepilatesperu.com/",
    "https://balancestudio.pe/",
    "https://curvaestudio.com/",  # no robots.txt
    "https://fitstudioperu.com/",
    "https://www.funcionalstudio.pe/",
    "https://pilatesesencia.com/",
    "https://twopilatesstudio.wixsite.com/twopilatesstudio",  # no sitemap in robots.txt
    "https://iliveko.com/",  # no sitemap in robots.txt
    "https://raise.pe/",
    "https://shadow.pe/",  #  no sitemap in robots.txt
    "https://elevatestudio.my.canva.site/",  # no robots.txt
    "https://www.boost-studio.com/"
]


def scrape_single_url(client: openai.OpenAI, page: Page, url: str, url_type: str):
    """Scrapes a single URL using a provided Playwright page object."""
    print(f"  -> Scraping {url}")
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        # Here you can add logic to wait for specific elements if needed
        # For example: expect(page.locator("body")).to_be_visible()

        raw_html = page.content()

        # Pass keywords to help the pruner find relevant content
        pruned_html = prune_html_for_llm(raw_html)

        print(f"     Original HTML length: {len(raw_html)} chars")
        print(f"     Pruned HTML length:   {len(pruned_html)} chars")

        # Here you would send the pruned_html to the LLM
        extracted_data = extract_structured_data(client, url, url_type, pruned_html, gym_name="Generic Gym")
        return extracted_data
    except Exception as e:
        print(f"     ❌ Failed to scrape {url}: {e}")


def main():
    load_dotenv()
    client = openai.Client()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for site_url in pages_to_scrape:
            print(f"Scraping {site_url}")
            urls_to_scrape = get_filtered_sitemap_urls(site_url)
            print(urls_to_scrape)
            filtered_urls = categorize_urls_with_llm(urls_to_scrape, client)
            filtered_urls["homepage"] = site_url
            print(filtered_urls)
            for page_type, sub_urls in filtered_urls.items():
                page = browser.new_page()
                try:
                    for sub_url in sub_urls:
                        extracted_data = scrape_single_url(client, page, sub_url, page_type)
                finally:
                    page.close()
        browser.close()
    print("Scraping complete.")


if __name__ == "__main__":
    main()
