import os
import re
import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
import streamlit as st
import anthropic
from dotenv import load_dotenv

load_dotenv()

CACHE_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "news_cache.json")


def save_cache(cache: dict):
    """Save cache to disk, converting datetime to ISO string."""
    serializable = {}
    for source, articles in cache.items():
        serializable[source] = [
            {**a, "date": a["date"].isoformat()} for a in articles
        ]
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def load_cache() -> tuple[dict, str]:
    """Load cache from disk. Returns (cache_dict, status_message)."""
    if not os.path.exists(CACHE_FILE):
        return {}, f"快取檔案不存在（{CACHE_FILE}）"
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cache = {}
        for source, articles in raw.items():
            cache[source] = [
                {**a, "date": datetime.fromisoformat(a["date"])} for a in articles
            ]
        total = sum(len(v) for v in cache.values())
        return cache, f"從快取讀取 {len(cache)} 個來源，共 {total} 篇"
    except Exception as e:
        return {}, f"快取讀取失敗：{e}"

# ─── Constants ────────────────────────────────────────────────────────────────

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
SUMMARY_BATCH_SIZE = 20

SOURCES = {
    "micromobility.io": "https://micromobility.io/news",
    "Electrek E-bikes": "https://electrek.co/guides/ebikes/",
    "Zag Daily": "https://zagdaily.com/feed/",
    "Bikerumor": "https://bikerumor.com/bike-types/e-bike-2/feed/",
    "Electric Bike Review": "https://electricbikereview.com/feed/",
    "Electric Bike Report": "https://electricbikereport.com/electric-bike-news/",
    "BikeRadar": "https://www.bikeradar.com/electric-bikes",
    "Bike-EU Germany": "https://www.bike-eu.com/germany",
    "Bike-EU Netherlands": "https://www.bike-eu.com/the-netherlands",
}

CATEGORIES = {
    "共享微移動 Shared Mobility": [
        "bikeshare", "bike share", "scooter share", "dott", "lime", "veo",
        "tier", "bird", "shared", "rental", "deploy", "fleet", "donkey republic",
        "urban sharing", "republic", "bikeshare 101",
    ],
    "電動自行車 E-bikes": [
        "e-bike", "ebike", "emtb", "e-mtb", "cargo bike", "cycling", "bicycle",
        "bike sales", "pedal", "canyon", "limebike", "lectric", "radio flyer",
        "off-road", "off-roading", "revamps", "all-mountain", "gravel",
    ],
    "Media Review": [
        "review", "test ride", "hands-on", "first look", "we rode", "tested",
        "riding", "i rode", "i took", "i bought", "covers all", "first ride",
    ],
    "商業與投資 Business & Investment": [
        "funding", "round", "revenue", "sales", "acquisition", "closes",
        "raises", "investment", "profit", "ipo", "valuation", "hits €", "hits $",
        "down 7%", "up 13%", "200m", "$200",
    ],
    "政策與法規 Policy & Regulation": [
        "regulation", "law", "ban", "government", "policy", "bill", "congress",
        "states", "regulating", "permit", "speed limit", "federal",
    ],
    "技術與產品 Tech & Product": [
        "battery", "charging", "launch", "new model", "technology",
        "software", "platform", "waymo", "crash", "data", "wireless",
        "lighter", "carbon frame", "motor", "drive unit", "tease", "reveal",
    ],
    "產業動態 Industry News": [],  # catch-all
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def categorize(title: str) -> str:
    t = title.lower()
    for cat, keywords in CATEGORIES.items():
        if keywords and any(re.search(r'\b' + re.escape(kw) + r'\b', t) for kw in keywords):
            return cat
    return "產業動態 Industry News"


def parse_date(text: str):
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


def make_article(title, date_obj, url, source, prefetched_text=""):
    return {
        "title": title,
        "date": date_obj,
        "url": url,
        "source": source,
        "category": categorize(title),
        "summary_en": "",
        "summary_zh": "",
        "prefetched_text": prefetched_text,
    }


# ─── Scrapers ─────────────────────────────────────────────────────────────────

def scrape_micromobility(days: int = 7) -> list:
    """micromobility.io/news — Webflow CMS structure."""
    base = "https://micromobility.io"
    cutoff = datetime.now() - timedelta(days=days)
    articles = []
    seen = set()
    page = 1

    while True:
        url = f"{base}/news" if page == 1 else f"{base}/news?f4254343_page={page}"
        try:
            resp = requests.get(url, timeout=15, headers=HEADERS)
            resp.raise_for_status()
        except Exception as e:
            st.error(f"[micromobility.io] 抓取失敗：{e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.find_all("div", class_="news-item")
        if not items:
            break

        stop = False
        found = False
        for item in items:
            title_tag = item.find("a", class_="news-link")
            date_tag = item.find("div", class_="news-date")
            if not title_tag or not date_tag:
                continue

            title = title_tag.get_text(strip=True)
            href = title_tag.get("href", "")
            if not title or not href or href in seen:
                continue
            seen.add(href)

            date_obj = parse_date(date_tag.get_text(strip=True))
            if not date_obj:
                continue
            if date_obj < cutoff:
                stop = True
                break

            articles.append(make_article(title, date_obj, base + href, "micromobility.io"))
            found = True

        if stop or not found:
            break
        if not soup.find("a", href=re.compile(r"f4254343_page=\d+")):
            break
        page += 1

    return articles


def scrape_electrek(days: int = 7) -> list:
    """electrek.co/guides/ebikes/ — WordPress structure, date in URL."""
    base_url = "https://electrek.co/guides/ebikes/"
    cutoff = datetime.now() - timedelta(days=days)
    articles = []
    seen = set()
    page = 1

    while True:
        url = base_url if page == 1 else f"{base_url}page/{page}/"
        try:
            resp = requests.get(url, timeout=15, headers=HEADERS)
            resp.raise_for_status()
        except Exception as e:
            st.error(f"[Electrek] 抓取失敗：{e}")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # Titles are in <h2 class="h1"><a href="...">Title</a></h2>
        headings = soup.find_all("h2", class_="h1")
        if not headings:
            break

        stop = False
        found = False
        for h2 in headings:
            a_tag = h2.find("a", href=re.compile(r"electrek\.co/20\d\d/\d\d/\d\d/"))
            if not a_tag:
                continue

            href = a_tag.get("href", "")
            if href in seen:
                continue
            seen.add(href)

            title = a_tag.get_text(strip=True)
            if not title:
                continue

            # Extract date from URL: /2026/04/07/
            m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", href)
            if not m:
                continue
            date_obj = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))

            if date_obj < cutoff:
                stop = True
                break

            articles.append(make_article(title, date_obj, href, "Electrek E-bikes"))
            found = True

        if stop or not found:
            break
        page += 1

    return articles


def scrape_rss(feed_url: str, source_name: str, days: int = 7,
               extra_headers: dict = None) -> list:
    """Generic RSS scraper — works for any WordPress-style RSS feed."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    articles = []

    req_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if extra_headers:
        req_headers.update(extra_headers)

    try:
        resp = requests.get(feed_url, timeout=15, headers=req_headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
    except Exception as e:
        st.error(f"[{source_name}] 抓取失敗：{e}")
        return articles

    for item in soup.find_all("item"):
        title_tag = item.find("title")
        link_tag  = item.find("link")
        pub_tag   = item.find("pubDate")
        content_tag = item.find("encoded")

        if not title_tag or not link_tag or not pub_tag:
            continue

        title   = title_tag.get_text(strip=True)
        url     = link_tag.get_text(strip=True)
        pub_str = pub_tag.get_text(strip=True)

        try:
            date_obj = parsedate_to_datetime(pub_str)
        except Exception:
            continue

        if date_obj < cutoff:
            break

        plain_text = ""
        if content_tag:
            raw_html = content_tag.get_text()
            plain_text = BeautifulSoup(raw_html, "html.parser").get_text(separator="\n", strip=True)[:2000]

        articles.append(make_article(title, date_obj.replace(tzinfo=None), url, source_name, plain_text))

    return articles


def scrape_zagdaily(days: int = 7) -> list:
    return scrape_rss("https://zagdaily.com/feed/", "Zag Daily", days,
                      extra_headers={"Referer": "https://zagdaily.com/"})


def scrape_electricbikereview(days: int = 7) -> list:
    """electricbikereview.com — main page reviews + RSS news."""
    articles = []
    today = datetime.now()
    cutoff = today - timedelta(days=days)
    seen_urls: set[str] = set()

    # Part 1: main page reviews (dates unavailable, use today as placeholder)
    try:
        resp = requests.get("https://electricbikereview.com/", timeout=15, headers=HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for item in soup.find_all("div", class_="review-item-wrapper"):
            h2 = item.find("h2", class_="title")
            if not h2:
                continue
            a = h2.find("a")
            if not a:
                continue
            title = a.get_text(strip=True)
            url = a.get("href", "")
            if title and url and url not in seen_urls:
                seen_urls.add(url)
                articles.append(make_article(title, today, url, "Electric Bike Review"))
    except Exception as e:
        st.error(f"[Electric Bike Review] 主頁抓取失敗：{e}")

    # Part 2: RSS news with date filter
    try:
        rss = requests.get("https://electricbikereview.com/feed/", timeout=15, headers=HEADERS)
        rss.raise_for_status()
        soup2 = BeautifulSoup(rss.content, "xml")
        for item in soup2.find_all("item"):
            title_tag = item.find("title")
            link_tag  = item.find("link")
            pub_tag   = item.find("pubDate")
            content_tag = item.find("encoded")
            if not title_tag or not link_tag:
                continue
            title = title_tag.get_text(strip=True)
            url   = link_tag.get_text(strip=True)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                date_obj = parsedate_to_datetime(pub_tag.get_text(strip=True)).replace(tzinfo=None) if pub_tag else today
            except Exception:
                date_obj = today
            if date_obj < cutoff:
                break
            plain_text = ""
            if content_tag:
                plain_text = BeautifulSoup(content_tag.get_text(), "html.parser").get_text(separator="\n", strip=True)[:2000]
            articles.append(make_article(title, date_obj, url, "Electric Bike Review", plain_text))
    except Exception as e:
        st.error(f"[Electric Bike Review] RSS抓取失敗：{e}")

    return articles


def scrape_electricbikereport(days: int = 7) -> list:
    """electricbikereport.com — WordPress RSS feed."""
    return scrape_rss("https://electricbikereport.com/feed/", "Electric Bike Report", days)


def scrape_bikeradar(days: int = 7) -> list:
    """bikeradar.com/electric-bikes — Purple SPA with embedded JSON state."""
    base = "https://www.bikeradar.com"
    cutoff = datetime.now() - timedelta(days=days)
    articles = []

    try:
        resp = requests.get(f"{base}/electric-bikes", timeout=20, headers=HEADERS)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"[BikeRadar] 抓取失敗：{e}")
        return articles

    # Article data is embedded in a <script> block as JSON (Purple platform)
    data = None
    for script_text in re.findall(r"<script[^>]*>(.*?)</script>", resp.text, re.DOTALL):
        if "PURPLE_CONTENT_CACHE" in script_text and "PURPLE_API_CACHE" in script_text:
            try:
                data = json.loads(script_text)
                break
            except Exception:
                continue

    if not data:
        st.error("[BikeRadar] 無法解析頁面資料（JSON 結構可能已變更）")
        return articles

    content_cache = data.get("PURPLE_CONTENT_CACHE", {})
    api_cache = data.get("PURPLE_API_CACHE", {})

    # Find the feed key that holds ordered article UUIDs
    article_ids = []
    for key, val in api_cache.items():
        if "getContents" in key and val.get("nodes"):
            article_ids = val["nodes"]
            break

    for aid in article_ids:
        article = content_cache.get(aid)
        if not article:
            continue

        title = article.get("name", "").strip()
        slug = article.get("properties", {}).get("slug", "")
        pub_ms = article.get("publicationDate", 0)

        if not title or not slug or not pub_ms:
            continue

        date_obj = datetime.fromtimestamp(pub_ms / 1000)
        if date_obj < cutoff:
            continue

        description = article.get("description", "")
        articles.append(make_article(title, date_obj, f"{base}/{slug}", "BikeRadar", description))

    return articles


def scrape_bikeeu(url: str, source_name: str, days: int = 7) -> list:
    """bike-eu.com — Vue SSR, date/title/description all in card text."""
    base = "https://www.bike-eu.com"
    cutoff = datetime.now() - timedelta(days=days)
    articles = []
    seen = set()

    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"[{source_name}] 抓取失敗：{e}")
        return articles

    soup = BeautifulSoup(resp.text, "html.parser")
    cards = soup.find_all("a", href=re.compile(r"^/\d+/"))

    for card in cards:
        href = card.get("href", "")
        if href in seen:
            continue
        seen.add(href)

        # Text order: Category | Date | Title | Description
        parts = [t.strip() for t in card.get_text(separator="|").split("|") if t.strip()]
        if len(parts) < 3:
            continue

        # Find date (format: "31 Mar 26")
        date_obj = None
        date_idx = None
        for i, part in enumerate(parts):
            try:
                date_obj = datetime.strptime(part, "%d %b %y")
                date_idx = i
                break
            except ValueError:
                continue

        if not date_obj or date_idx is None:
            continue

        if date_obj < cutoff:
            break

        # Title is right after date, description after title
        title = parts[date_idx + 1] if date_idx + 1 < len(parts) else ""
        description = parts[date_idx + 2] if date_idx + 2 < len(parts) else ""

        if not title:
            continue

        article_url = base + href
        articles.append(make_article(title, date_obj, article_url, source_name, description))

    return articles


# ─── Article text fetcher ─────────────────────────────────────────────────────

def fetch_article_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup.find_all(["nav", "footer", "aside", "script", "style", "figure"]):
            tag.decompose()

        # Try specific article body selectors first
        for selector in [".post-content", ".entry-content", ".article-content",
                         ".article__content", ".post__content", "article > div", "article"]:
            body = soup.select_one(selector)
            if body:
                # Only grab <p> tags to avoid nav/sidebar noise
                paras = [p.get_text(strip=True) for p in body.find_all("p") if len(p.get_text(strip=True)) > 30]
                text = "\n".join(paras)
                if len(text) > 100:
                    return text[:2000]

        # Final fallback: all paragraphs on page
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
        return "\n".join(paragraphs)[:2000]
    except Exception:
        return ""


def fetch_article_text_full(url: str) -> str:
    """Same as fetch_article_text but with a higher character limit for full reading."""
    try:
        resp = requests.get(url, timeout=15, headers=HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup.find_all(["nav", "footer", "aside", "script", "style", "figure"]):
            tag.decompose()

        for selector in [".post-content", ".entry-content", ".article-content",
                         ".article__content", ".post__content", "article > div", "article"]:
            body = soup.select_one(selector)
            if body:
                paras = [p.get_text(strip=True) for p in body.find_all("p") if len(p.get_text(strip=True)) > 30]
                text = "\n\n".join(paras)
                if len(text) > 100:
                    return text[:6000]

        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
        return "\n\n".join(paragraphs)[:6000]
    except Exception:
        return ""


def fetch_and_translate(url: str, title: str) -> tuple[str, str]:
    """Fetch full article and return (english_text, chinese_translation)."""
    en_text = fetch_article_text_full(url)
    if not en_text:
        return "(無法取得全文)", ""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return en_text, "(未設定 ANTHROPIC_API_KEY)"

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "你是一位專業的科技與交通產業記者，母語為台灣繁體中文。\n"
        "請將以下英文文章翻譯成台灣繁體中文，要求：\n"
        "1. 語感自然，符合台灣讀者的閱讀習慣，避免逐字直譯\n"
        "2. 使用台灣慣用詞彙（例如：腳踏車、機車、手機、軟體、硬體、網路）\n"
        "3. 專有名詞、品牌名、人名保留英文原文\n"
        "4. 數字與單位照原文，金額幣別保留（如 $200M、€500K）\n"
        "5. 保留原文段落結構，只輸出繁體中文翻譯，不重複原文\n\n"
        f"標題：{title}\n\n"
        f"原文：\n{en_text}"
    )
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        return en_text, resp.content[0].text.strip()
    except Exception as e:
        return en_text, f"(翻譯失敗：{e})"


def fetch_all_texts_parallel(articles: list, max_workers: int = 6) -> dict:
    results = {}
    # Pre-fill articles that already have content from RSS
    to_fetch = []
    for a in articles:
        if a.get("prefetched_text"):
            results[a["url"]] = a["prefetched_text"]
        else:
            to_fetch.append(a)

    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(fetch_article_text, a["url"]): a["url"] for a in to_fetch}
            for future in as_completed(future_to_url):
                results[future_to_url[future]] = future.result()
    return results


# ─── Claude API ───────────────────────────────────────────────────────────────

def _call_claude_batch(client, batch: list, texts: dict) -> None:
    """Send one batch of articles to Claude and fill in summary_en / summary_zh in-place."""
    article_blocks = [
        f"<<<ARTICLE_{i+1}>>>\n標題：{a['title']}\n內容：{texts.get(a['url'], '') or '(無法取得內容)'}"
        for i, a in enumerate(batch)
    ]
    prompt = (
        "你是一位專業的科技與交通產業記者，母語為台灣繁體中文。\n"
        "請為以下每篇文章各生成一組摘要：英文摘要（2–3句）與台灣繁體中文摘要（2–3句）。\n"
        "中文摘要須語感自然、符合台灣讀者習慣，避免逐字直譯，使用台灣慣用詞彙。\n\n"
        "輸出格式嚴格如下，每篇之間空一行，分隔符必須完整保留：\n"
        "<<<ARTICLE_1>>>\nEN: <英文摘要>\nZH: <台灣繁體中文摘要>\n\n"
        "<<<ARTICLE_2>>>\nEN: <英文摘要>\nZH: <台灣繁體中文摘要>\n\n"
        "（只輸出分隔符和摘要，不要其他說明文字）\n\n"
        "文章列表：\n\n" + "\n\n".join(article_blocks)
    )
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=350 * len(batch),
        messages=[{"role": "user", "content": prompt}],
    )
    result = resp.content[0].text.strip()
    blocks = re.split(r"<<<ARTICLE_(\d+)>>>", result)
    i = 1
    while i < len(blocks) - 1:
        idx = int(blocks[i]) - 1
        content = blocks[i + 1]
        en_m = re.search(r"EN:\s*(.+)", content)
        zh_m = re.search(r"ZH:\s*(.+)", content)
        if 0 <= idx < len(batch):
            batch[idx]["summary_en"] = en_m.group(1).strip() if en_m else ""
            batch[idx]["summary_zh"] = zh_m.group(1).strip() if zh_m else ""
        i += 2


def generate_summaries(articles: list, existing_by_url: dict | None = None) -> list:
    # Restore existing summaries first to avoid unnecessary API calls
    if existing_by_url:
        for a in articles:
            cached = existing_by_url.get(a["url"])
            if cached and cached.get("summary_en"):
                a["summary_en"] = cached["summary_en"]
                a["summary_zh"] = cached["summary_zh"]

    to_summarize = [a for a in articles if not a.get("summary_en")]
    if not to_summarize:
        return articles

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.warning("未設定 ANTHROPIC_API_KEY，跳過摘要生成。")
        return articles

    client = anthropic.Anthropic(api_key=api_key)

    with st.spinner(f"同時抓取 {len(to_summarize)} 篇新文章內容..."):
        texts = fetch_all_texts_parallel(to_summarize)

    total_batches = (len(to_summarize) + SUMMARY_BATCH_SIZE - 1) // SUMMARY_BATCH_SIZE
    for batch_idx in range(total_batches):
        batch = to_summarize[batch_idx * SUMMARY_BATCH_SIZE:(batch_idx + 1) * SUMMARY_BATCH_SIZE]
        label = f"Claude 生成摘要中（第 {batch_idx + 1}/{total_batches} 批）..." if total_batches > 1 else "Claude 生成摘要中..."
        with st.spinner(label):
            try:
                _call_claude_batch(client, batch, texts)
            except Exception as e:
                st.warning(f"第 {batch_idx + 1} 批摘要生成失敗：{e}")

    return articles


# ─── Dashboard ────────────────────────────────────────────────────────────────

SCRAPER_MAP = {
    "micromobility.io": scrape_micromobility,
    "Electrek E-bikes": scrape_electrek,
    "Zag Daily": scrape_zagdaily,
    "Bikerumor": lambda days: scrape_rss("https://bikerumor.com/bike-types/e-bike-2/feed/", "Bikerumor", days,
                                         extra_headers={"User-Agent": "FeedFetcher-Google; (+http://www.google.com/feedfetcher.html)"}),
    "Electric Bike Review": scrape_electricbikereview,
    "Electric Bike Report": scrape_electricbikereport,
    "BikeRadar": scrape_bikeradar,
    "Bike-EU Germany": lambda days: scrape_bikeeu("https://www.bike-eu.com/germany", "Bike-EU Germany", days),
    "Bike-EU Netherlands": lambda days: scrape_bikeeu("https://www.bike-eu.com/the-netherlands", "Bike-EU Netherlands", days),
}


def render_articles(articles: list, key_prefix: str = ""):
    if not articles:
        st.info("此分類沒有文章。")
        return

    grouped: dict[str, list] = {}
    for a in articles:
        grouped.setdefault(a["category"], []).append(a)

    for cat, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x["date"], reverse=True)
        st.subheader(f"{cat}　`{len(items_sorted)} 篇`")

        for item in items_sorted:
            with st.container(border=True):
                title_col, meta_col = st.columns([6, 1])
                with title_col:
                    st.markdown(f"**{item['title']}**")
                with meta_col:
                    st.caption(item["date"].strftime("%Y-%m-%d"))
                    st.caption(f"🔗 {item['source']}")

                en_col, zh_col = st.columns(2)
                with en_col:
                    st.caption("🇺🇸 English Summary")
                    st.write(item["summary_en"] or "—")
                with zh_col:
                    st.caption("🇹🇼 中文摘要")
                    st.write(item["summary_zh"] or "—")

                cache_key = f"fulltext_{item['url']}"
                show_key = f"show_{item['url']}"
                is_open = st.session_state.get(show_key, False)
                btn_label = "🔼 收起全文" if is_open else "📖 展開中英對照全文"
                col_link, col_btn = st.columns([4, 1])
                with col_link:
                    st.markdown(f"[原文連結 →]({item['url']})")
                with col_btn:
                    if st.button(btn_label, key=f"btn_{key_prefix}_{item['url']}"):
                        st.session_state[show_key] = not is_open

                if st.session_state.get(show_key, False):
                    if cache_key not in st.session_state:
                        with st.spinner("正在抓取並翻譯全文..."):
                            st.session_state[cache_key] = fetch_and_translate(item["url"], item["title"])
                    en_text, zh_text = st.session_state[cache_key]
                    col_en, col_zh = st.columns(2)
                    with col_en:
                        st.caption("🇺🇸 English")
                        st.markdown(en_text)
                    with col_zh:
                        st.caption("🇹🇼 繁體中文")
                        st.markdown(zh_text)

        st.write("")


def main():
    st.set_page_config(
        page_title="Micromobility News Dashboard",
        layout="wide",
        page_icon="🛴",
    )
    st.title("🛴 Micromobility News Dashboard")

    FREE_SOURCES  = ["micromobility.io", "Electrek E-bikes", "Zag Daily", "Bikerumor", "Electric Bike Review", "Electric Bike Report", "BikeRadar"]
    PAID_SOURCES  = []  # Bike-EU Germany / Bike-EU Netherlands hidden but kept in SCRAPER_MAP as reference

    # Sidebar
    with st.sidebar:
        st.header("⚙️ 設定")
        days = st.slider("抓取天數", min_value=1, max_value=14, value=7, step=1)
        st.divider()
        selected_free = st.multiselect(
            "新聞來源",
            options=FREE_SOURCES,
            default=FREE_SOURCES,
        )
        st.divider()
        selected_paid = st.multiselect(
            "🔒 Paid Media",
            options=PAID_SOURCES,
            default=PAID_SOURCES,
        )
        st.divider()
        all_cats = list(CATEGORIES.keys())
        selected_cats = st.multiselect("文章分類篩選", options=all_cats, default=all_cats)
        st.divider()
        sources_to_refresh = st.multiselect(
            "指定重新抓取來源",
            options=list(SOURCES.keys()),
            default=list(SOURCES.keys()),
            help="只勾選的來源會重新抓取，其他保留快取",
        )
        refresh = st.button("🔄 重新抓取 & 更新", use_container_width=True)
        if "cache_status" in st.session_state:
            st.caption(st.session_state["cache_status"])

    selected_sources = selected_free + selected_paid

    # Fetch data — persistent disk cache, only re-fetch selected sources
    if "article_cache" not in st.session_state:
        loaded, status_msg = load_cache()
        st.session_state["article_cache"] = loaded
        st.session_state["cache_status"] = status_msg

    cache = st.session_state["article_cache"]

    if not cache:
        to_refresh = list(SOURCES.keys())
    elif refresh:
        to_refresh = sources_to_refresh
    else:
        to_refresh = []

    if to_refresh:
        for source in to_refresh:
            with st.spinner(f"抓取 {source}..."):
                arts = SCRAPER_MAP[source](days)
                if arts:
                    existing_by_url = {a["url"]: a for a in cache.get(source, [])}
                    cache[source] = generate_summaries(arts, existing_by_url)

        st.session_state["article_cache"] = cache
        save_cache(cache)
        st.session_state["fetched_at"] = datetime.now()
    elif "fetched_at" not in st.session_state and cache:
        st.session_state["fetched_at"] = datetime.fromtimestamp(
            os.path.getmtime(CACHE_FILE)
        ) if os.path.exists(CACHE_FILE) else datetime.now()

    all_articles = [a for arts in cache.values() for a in arts]

    if not all_articles:
        st.warning("未找到符合條件的文章。")
        return

    fetched_at = st.session_state.get("fetched_at")

    # Filter
    articles = [
        a for a in all_articles
        if a["source"] in selected_sources and a["category"] in selected_cats
    ]

    # Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("文章總數", len(articles))
    col2.metric("涵蓋分類", len({a["category"] for a in articles}))
    col3.metric("最新文章", articles[0]["date"].strftime("%m/%d") if articles else "—")
    col4.metric("資料更新時間", fetched_at.strftime("%H:%M") if fetched_at else "—")

    st.divider()

    # Tabs by source
    source_tabs = st.tabs(["全部"] + selected_sources)

    with source_tabs[0]:
        render_articles(articles, key_prefix="all")

    for i, source in enumerate(selected_sources):
        with source_tabs[i + 1]:
            render_articles([a for a in articles if a["source"] == source], key_prefix=source)


if __name__ == "__main__":
    main()
