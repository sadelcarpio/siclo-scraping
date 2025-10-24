import re

import openai
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page
from selectolax.parser import HTMLParser

from sitemap_utils import get_filtered_sitemap_urls
from src.llm import categorize_urls_with_llm, extract_structured_data, merge_gym_data_with_llm

pages_to_scrape = {
    # "bioritmo": "https://www.bioritmo.com.pe/",
    # "limayoga": "https://limayoga.com/",
    "nasceyoga": "https://www.nasceyoga.com/",
    "zendayoga": "https://www.zendayoga.com/",
    "matmax": "https://matmax.world/",
    "anjali": "https://anjali.pe/",
    "purepilatesperu": "https://www.purepilatesperu.com/",
    "balancestudio": "https://balancestudio.pe/",
    "curvaestudio": "https://curvaestudio.com/", # no robots.txt
    "fitstudioperu": "https://fitstudioperu.com/",
    "funcionalstudio": "https://www.funcionalstudio.pe/",
    "pilatesesencia": "https://pilatesesencia.com/",
    "twopilatesstudio": "https://twopilatesstudio.wixsite.com/twopilatesstudio", # no sitemap in robots.txt
    "iliveko": "https://iliveko.com/", # no sitemap in robots.txt
    "raise": "https://raise.pe/",
    "shadow": "https://shadow.pe/", #  no sitemap in robots.txt
    "elevatestudio": "https://elevatestudio.my.canva.site/", # no robots.txt
    "boost-studio": "https://www.boost-studio.com/"
}


def should_skip_frame(frame):
    skip_domains = ["stripe.com", "facebook.com", "google.com", "analytics", "wixapps"]
    return any(domain in frame.url for domain in skip_domains)


def scroll_until_iframes(page: Page, max_scrolls: int = 30, scroll_step: int = 1000, stable_checks: int = 3):
    """
    Hace scroll progresivo hasta que los iframes dejan de aumentar.
    Retorna el número final de iframes encontrados.
    """
    last_count = 0
    stable_counter = 0

    for i in range(max_scrolls):
        iframes_count = len(page.query_selector_all("iframe"))
        print(f"     🔎 Scroll {i+1}/{max_scrolls}: found {iframes_count} iframes")

        if iframes_count == last_count:
            stable_counter += 1
        else:
            stable_counter = 0
            last_count = iframes_count

        if stable_counter >= stable_checks and iframes_count > 0:
            print(f"     ✅ Iframes stabilized at {iframes_count} after {i+1} scrolls")
            break

        page.mouse.wheel(0, scroll_step)
        page.wait_for_timeout(1000)
    return last_count


def flatten_nested_divs_regex(html: str) -> str:
    """
    Colapsa wrappers <div><div>...</div></div> hasta dejar solo <div>...</div>.
    Funciona de forma iterativa y es segura para fragments.
    """
    if not html:
        return html

    prev = None
    out = html

    # Paso 1: normalizar un poco espacios entre etiquetas
    out = re.sub(r'>\s+<', '><', out)

    # Iteramos hasta convergencia
    while prev != out:
        prev = out
        # 1) Colapsar aperturas: <div ...><div ...> -> <div>
        out = re.sub(r'<div\b[^>]*>\s*<div\b[^>]*>', '<div>', out, flags=re.IGNORECASE)

        # 2) Colapsar cierres: </div></div> -> </div>
        out = re.sub(r'</div>\s*</div>', '</div>', out, flags=re.IGNORECASE)

    # Opcional: limpiar repetidos de espacios y newlines
    out = re.sub(r'\s+', ' ', out).strip()

    return out


def prune_html_for_llm(html_content: str, keywords: list[str] = None) -> tuple[str, list[str]]:
    """
    Limpia HTML y extrae contenido relevante y tablas legibles para un LLM.
    Devuelve (html_limpio, tablas_en_texto)
    """
    tree = HTMLParser(html_content)

    # 1️⃣ Intentar aislar contenedor principal
    main_container = None
    if tree.body:
        main_container = tree.body.css_first('main')
    if not main_container and tree.body:
        main_container = tree.body.css_first('[role="main"]')

    # fallback: buscar div o section con palabras clave
    if not main_container and keywords and tree.body:
        best_candidate, max_score = None, 0
        for node in tree.body.css('div, section'):
            score = sum(node.text(deep=True).lower().count(kw) for kw in keywords)
            if score > max_score:
                max_score, best_candidate = score, node
        if max_score > 0:
            main_container = best_candidate

    root_node = main_container if main_container else tree.body
    if not root_node:
        return "", []

    # 2️⃣ Remover ruido visual / irrelevante
    noise_tags = ['script', 'style', 'svg', 'nav', 'footer', 'header']
    for tag in noise_tags:
        for node in root_node.css(tag):
            node.decompose()

    # 3️⃣ Extraer tablas como texto legible
    table_texts = []
    for table in root_node.css('table'):
        rows = []
        for tr in table.css('tr'):
            cells = [td.text(strip=True) for td in tr.css('th, td')]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            table_texts.append("\n".join(rows))
        table.decompose()  # eliminar tabla del HTML principal (para no duplicar)

    # 4️⃣ Eliminar todos los atributos de los nodos
    for node in root_node.css("*"):
        if node.attributes:
            node.attributes.clear()

    # 5️⃣ Obtener HTML limpio
    html_clean = root_node.html

    # 6️⃣ Compactar espacios y saltos de línea
    html_clean = re.sub(r'\n+', '\n', html_clean)
    html_clean = re.sub(r'\s+', ' ', html_clean)
    html_clean = html_clean.strip()

    return html_clean, table_texts



def _get_item_key(item: dict, category: str) -> tuple | None:
    # ... (código de la respuesta anterior)
    if category == "ubicaciones":
        key_parts = (item.get("distrito"), item.get("direccion_completa"))
        return key_parts if all(key_parts) else None
    elif category == "precios":
        key_parts = (item.get("descripcion_plan"), item.get("valor"), item.get("recurrencia"))
        return key_parts if all(key_parts) else None
    elif category == "horarios":
        key_parts = (item.get("sede"), item.get("nombre_clase"), item.get("dia_semana"), item.get("hora_inicio"))
        return key_parts if all(key_parts) else None
    elif category == "disciplinas":
        key_parts = (item.get("nombre"),)
        return key_parts if all(key_parts) else None
    return None

def _merge_items(existing_item: dict, new_item: dict) -> dict:
    # ... (código de la respuesta anterior)
    score_existing = sum(1 for v in existing_item.values() if v)
    score_new = sum(1 for v in new_item.values() if v)
    if score_new > score_existing:
        return new_item
    elif score_new == score_existing:
        if len(new_item.get("content_para_busqueda", "")) > len(existing_item.get("content_para_busqueda", "")):
            return new_item
    return existing_item


def scrape_single_url(client: openai.OpenAI, page: Page, url: str, url_type: str, gym_name: str):
    """
    Raspa una URL y cualquier iframe relevante que contenga, fusionando los resultados.
    """
    print(f"  -> Scraping URL principal: {url}")

    chunks_data = {}

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=180000)
        scroll_until_iframes(page)

        # 3. Procesar los iframes relevantes
        for frame in page.frames:
            # Heurística para decidir si un iframe es interesante
            if should_skip_frame(frame):
                continue
            print(f"     Found relevant iframe. Scraping: {frame.url}")
            try:
                try:
                    page.goto(frame.url, wait_until="networkidle", timeout=45000)
                except Exception:
                    page.goto(frame.url, wait_until="domcontentloaded", timeout=45000)
                frame_html = page.content()
                pruned_frame_html, tables = prune_html_for_llm(frame_html)

                if pruned_frame_html.strip():
                    print(f"     Extracting from iframe content...")
                    iframe_data = extract_structured_data(client, frame.url, "iframe_content", pruned_frame_html,
                                                          gym_name, tables)
                    # Fusionar datos del iframe
                    if iframe_data:
                        chunks_data[frame.url] = iframe_data
            except Exception as e:
                print(f"     ❌ Failed to scrape iframe {frame.url}: {e}")

        # Convertir los diccionarios acumulados de nuevo a listas
        return chunks_data

    except Exception as e:
        print(f"     ❌ Failed to scrape main URL {url}: {e}")
        return {"ubicaciones": [], "precios": [], "horarios": [], "disciplinas": []}


def main():
    load_dotenv()
    client = openai.Client()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for gym_name, site_url in pages_to_scrape.items():
            print(f"Scraping {site_url}")
            urls_to_scrape = get_filtered_sitemap_urls(site_url)
            print(urls_to_scrape)
            filtered_urls = categorize_urls_with_llm(urls_to_scrape, client)
            filtered_urls["homepage"] = [site_url]
            print(filtered_urls)
            chunked_data = {}
            for page_type, sub_urls in filtered_urls.items():
                page = browser.new_page()
                try:
                    for sub_url in sub_urls:
                        extracted_data = scrape_single_url(client, page, sub_url, page_type, gym_name)
                        chunked_data = chunked_data | extracted_data
                except Exception as e:
                    print(e)
                finally:
                    page.close()
            print(chunked_data)
            merged_gym_data = merge_gym_data_with_llm(gym_name, chunked_data, client)
        browser.close()
    print("Scraping complete.")


if __name__ == "__main__":
    main()
