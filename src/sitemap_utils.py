import logging

import httpx
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, urljoin


def get_filtered_sitemap_urls(base_url: str) -> list[dict]:
    """
    Finds the sitemap from a site's robots.txt, fetches all page URLs with their metadata
    (lastmod, changefreq, priority), and returns a unique list filtered to the base domain.

    Args:
        base_url: The starting URL of the website (e.g., "https://bioritmo.com.pe").

    Returns:
        A list of dicts: [{"loc": str, "lastmod": Optional[str], "changefreq": Optional[str], "priority": Optional[str]}]
    """
    try:
        base_netloc = urlparse(base_url).netloc
        robots_url = urljoin(base_url, "/robots.txt")

        with httpx.Client(follow_redirects=True, timeout=15.0) as client:
            # 1️⃣ Fetch robots.txt
            logging.info(f"🔍 Fetching {robots_url}...")
            response = client.get(robots_url)
            if response.status_code != 200:
                logging.warning(f"⚠️ Could not fetch robots.txt (Status: {response.status_code}).")
                return []

            # Find all sitemap URLs
            sitemap_urls = [
                line.split(":", 1)[1].strip()
                for line in response.text.splitlines()
                if line.strip().lower().startswith("sitemap:")
            ]
            if not sitemap_urls:
                logging.warning("⚠️ No sitemap URL found in robots.txt.")
                return []

            logging.info(f"✅ Found sitemap(s): {sitemap_urls}")

            # 2️⃣ Process sitemap(s)
            urls_to_process = sitemap_urls
            all_page_entries = []

            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

            while urls_to_process:
                s_url = urls_to_process.pop(0)
                logging.info(f"  -> Processing {s_url}...")
                s_response = client.get(s_url)
                if s_response.status_code != 200:
                    continue

                try:
                    root = ET.fromstring(s_response.content)

                    # Handle sitemap index
                    if root.tag.endswith("sitemapindex"):
                        for sitemap in root.findall("sm:sitemap", ns):
                            loc = sitemap.find("sm:loc", ns)
                            if loc is not None:
                                urls_to_process.append(loc.text.strip())

                    # Handle regular URL set
                    elif root.tag.endswith("urlset"):
                        for url_entry in root.findall("sm:url", ns):
                            loc = url_entry.find("sm:loc", ns)
                            if loc is not None:
                                entry = {
                                    "loc": loc.text.strip(),
                                    "lastmod": None,
                                    "changefreq": None,
                                    "priority": None,
                                }

                                lastmod = url_entry.find("sm:lastmod", ns)
                                changefreq = url_entry.find("sm:changefreq", ns)
                                priority = url_entry.find("sm:priority", ns)

                                if lastmod is not None:
                                    entry["lastmod"] = lastmod.text.strip()
                                if changefreq is not None:
                                    entry["changefreq"] = changefreq.text.strip()
                                if priority is not None:
                                    entry["priority"] = priority.text.strip()

                                all_page_entries.append(entry)

                except ET.ParseError:
                    logging.error(f"  ⚠️ Failed to parse XML from {s_url}")

            # 3️⃣ Filter by base domain
            logging.info(f"Found {len(all_page_entries)} total entries. Filtering for domain '{base_netloc}'...")
            filtered_entries = [
                e for e in all_page_entries
                if urlparse(e["loc"]).netloc == base_netloc
            ]

            # Deduplicate
            seen = set()
            unique_entries = []
            for e in filtered_entries:
                if e["loc"] not in seen:
                    seen.add(e["loc"])
                    unique_entries.append(e)

            logging.info(f"✅ Returning {len(unique_entries)} URLs with metadata.")
            return unique_entries

    except httpx.RequestError as e:
        logging.error(f"❌ An error occurred during the request: {e}")
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
