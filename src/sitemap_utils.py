import httpx
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin


def get_filtered_sitemap_urls(base_url: str) -> list[str]:
    """
    Finds the sitemap from a site's robots.txt, fetches all page URLs,
    and returns a unique list of URLs filtered to the original base domain.

    Args:
        base_url: The starting URL of the website (e.g., "https://bioritmo.com.pe").

    Returns:
        A list of unique URLs from the sitemap that belong to the base_url's domain.
    """
    try:
        base_netloc = urlparse(base_url).netloc
        robots_url = urljoin(base_url, "/robots.txt")

        with httpx.Client(follow_redirects=True) as client:
            # 1. Fetch robots.txt to find sitemap URLs
            print(f"🔍 Fetching {robots_url}...")
            response = client.get(robots_url)

            if response.status_code != 200:
                print(f"⚠️ Could not fetch robots.txt (Status: {response.status_code}).")
                return []

            sitemap_urls = [
                line.split(":", 1)[1].strip()
                for line in response.text.splitlines()
                if line.strip().lower().startswith("sitemap:")
            ]

            if not sitemap_urls:
                print("⚠️ No sitemap URL found in robots.txt.")
                return []

            print(f"✅ Found sitemap(s): {sitemap_urls}")

            # 2. Process all sitemaps (handles sitemap indexes)
            urls_to_process = sitemap_urls
            all_page_urls = set()

            # Namespace for sitemap XML
            ns = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

            while urls_to_process:
                s_url = urls_to_process.pop(0)
                print(f"  -> Processing {s_url}...")
                s_response = client.get(s_url)
                if s_response.status_code != 200:
                    continue

                try:
                    root = ET.fromstring(s_response.content)

                    # Check if it's a sitemap index file
                    if root.tag.endswith('sitemapindex'):
                        # Add nested sitemap URLs to the processing queue
                        for sitemap in root.findall('sm:sitemap', ns):
                            loc = sitemap.find('sm:loc', ns)
                            if loc is not None:
                                urls_to_process.append(loc.text.strip())

                    # Check if it's a regular sitemap file
                    elif root.tag.endswith('urlset'):
                        # Add page URLs to our result set
                        for url_entry in root.findall('sm:url', ns):
                            loc = url_entry.find('sm:loc', ns)
                            if loc is not None:
                                all_page_urls.add(loc.text.strip())
                except ET.ParseError:
                    print(f"  -> ⚠️ Failed to parse XML from {s_url}")

            # 3. Filter the collected URLs to match the base domain
            print(f"Found {len(all_page_urls)} total URLs. Filtering for domain '{base_netloc}'...")

            filtered_urls = [
                url for url in all_page_urls
                if urlparse(url).netloc == base_netloc
            ]

            print(f"✅ Returning {len(filtered_urls)} filtered URLs.")
            return filtered_urls

    except httpx.RequestError as e:
        print(f"❌ An error occurred during the request: {e}")
        return []


if __name__ == "__main__":
    # Example usage with the tricky bioritmo site
    target_site = "https://www.bioritmo.com.pe"
    urls = get_filtered_sitemap_urls(target_site)

    print("\n--- Filtered URLs ---")
    if urls:
        for url in urls[:10]:  # Print first 10 for brevity
            print(url)
    else:
        print("No URLs found.")
