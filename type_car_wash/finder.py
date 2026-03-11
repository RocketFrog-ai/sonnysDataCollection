import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, unquote
from typing import Optional
import re
import time

# ----------------------------
# CONFIG
# ----------------------------

BLOCKED_DOMAINS = [
    "yelp.com",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "mapquest.com",
    "tripadvisor.com",
    "yellowpages.com",
    "bbb.org",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


# ----------------------------
# UTILS
# ----------------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_real_url(duckduckgo_url: str) -> str:
    """
    DuckDuckGo wraps URLs. This extracts the actual link.
    """
    parsed = urlparse(duckduckgo_url)
    if "duckduckgo.com" in parsed.netloc:
        params = parsed.query
        match = re.search(r'uddg=(.*?)&', params)
        if match:
            return unquote(match.group(1))
    return duckduckgo_url


def is_blocked_domain(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(blocked in domain for blocked in BLOCKED_DOMAINS)


def domain_score(url: str, company_name: str) -> int:
    """
    Score domain similarity (0-50)
    """
    domain = urlparse(url).netloc.lower()
    company_words = clean_text(company_name).split()

    score = 0
    for word in company_words:
        if len(word) > 2 and word in domain:  # Only score words longer than 2 chars to avoid 'car' matching everywhere
            score += 15

    return min(score, 50)


def content_score(url: str, company_name: str) -> int:
    """
    Score homepage content similarity (0-50)
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=8)
        text = clean_text(response.text)

        company_words = clean_text(company_name).split()
        score = 0

        for word in company_words:
            if len(word) > 2 and word in text:
                score += 10

        if "car wash" in text or "auto wash" in text:
            score += 10

        return min(score, 50)

    except:
        return 0


# ----------------------------
# SEARCH
# ----------------------------

def search_duckduckgo(query: str) -> list[str]:
    url = "https://html.duckduckgo.com/html/"
    response = requests.post(url, data={"q": query}, headers=HEADERS)

    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    for result in soup.find_all("a", class_="result__a", href=True):
        real_url = extract_real_url(result["href"])
        if real_url and real_url.startswith("http"):
            links.append(real_url)

    return links


# ----------------------------
# MAIN FINDER
# ----------------------------

def find_official_website(company_name: str) -> tuple[Optional[str], int]:
    print(f"[*] Searching DuckDuckGo for: '{company_name}'...")
    query = f"{company_name} official website"
    results = search_duckduckgo(query)
    
    if not results:
        print("[-] DuckDuckGo returned no results or blocked the request.")
        return None, 0

    best_match = None
    best_score = 0

    print("[*] Evaluating potential domain matches...")
    for link in results[:8]:

        if is_blocked_domain(link):
            continue

        d_score = domain_score(link, company_name)
        c_score = content_score(link, company_name)

        total_score = d_score + c_score

        if total_score > best_score:
            best_score = total_score
            best_match = link

        time.sleep(1)  # polite delay

    if best_score >= 40:  # confidence threshold
        print(f"[+] Found official website: {best_match} (Confidence: {best_score}/100)")
        return best_match, best_score

    print(f"[-] Could not confidently determine official website. Best guess: {best_match} (Score: {best_score}/100)")
    return None, best_score
