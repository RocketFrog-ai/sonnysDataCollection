import requests
import re
from urllib.parse import urljoin, urlparse
from type_car_wash.config import JINA_API_KEY

def scrape_with_jina(url: str) -> str:
    """
    Scrapes the text content of a single website URL using Jina AI's Reader API.
    """
    jina_url = f"https://r.jina.ai/{url}"
    headers = {}
    
    if JINA_API_KEY:
        headers['Authorization'] = f'Bearer {JINA_API_KEY}'
    
    print(f"[*] Scraping {url} via Jina AI...")
    try:
        # Added a timeout to prevent hanging indefinitely
        response = requests.get(jina_url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"[-] Error scraping {url}: {e}")
        return ""

def extract_internal_links(base_url: str, markdown_text: str) -> set[str]:
    """
    Extracts links from markdown text that belong to the same domain,
    specifically prioritizing pages that might have pricing or service info.
    """
    base_domain = urlparse(base_url).netloc
    
    # Regex to find markdown links: [text](url)
    link_pattern = re.compile(r'\[.*?\]\((.*?)\)')
    found_links = link_pattern.findall(markdown_text)
    
    internal_links: set[str] = set()
    for link in found_links:
        # Ignore anchor tags or javascript links
        if link.startswith('#') or link.startswith('javascript:'):
            continue
            
        # Resolve relative URLs
        full_url = urljoin(base_url, link)
        parsed_full = urlparse(full_url)
        
        url_path = parsed_full.path.lower()
        
        # Only keep links from the same domain
        if parsed_full.netloc == base_domain:
            # Skip media and document files
            if any(url_path.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.pdf', '.mp4', '.svg', '.webp']):
                continue
                
            # We specifically want to prioritize pages that likely have packages
            keywords = ['service', 'wash', 'price', 'pricing', 'package', 'detail', 'menu']
            if any(kw in url_path for kw in keywords):
                # Clean URL (remove fragments and query params for uniqueness if desired, 
                # but keep query params just in case they are important for routing)
                clean_url = f"{parsed_full.scheme}://{parsed_full.netloc}{parsed_full.path}"
                internal_links.add(clean_url)
                
    return internal_links

def scrape_site(start_url: str, max_pages: int = 3) -> str:
    """
    Scrapes the starting URL, then finds internal links related to services/pricing
    and scrapes a few of those as well to get a complete picture.
    """
    print(f"\n[*] Starting contextual scraping for: {start_url}")
    
    visited = set()
    all_text = []
    
    # Scrape the homepage first
    home_text = scrape_with_jina(start_url)
    if not home_text:
        return ""
        
    all_text.append(f"--- Homepage ({start_url}) ---\n" + home_text)
    visited.add(start_url)
    
    # Find service/pricing links
    links_to_visit = extract_internal_links(start_url, home_text)
    
    # Sort them to prioritize 'wash-services', 'pricing', etc.
    # Just grab a few to avoid scraping too much
    links_to_visit = list(links_to_visit)[:max_pages - 1]
    
    for link in links_to_visit:
        if link not in visited:
            page_text = scrape_with_jina(link)
            if page_text:
                all_text.append(f"\n--- Subpage ({link}) ---\n" + page_text)
            visited.add(link)
            
    return "\n\n".join(all_text)

