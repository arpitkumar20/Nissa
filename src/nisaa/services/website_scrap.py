# import re
# import time
# import chardet
# import requests
# from tqdm import tqdm
# from typing import List, Dict
# from datetime import datetime
# from bs4 import BeautifulSoup
# from urllib.parse import urlparse, urljoin
# from langchain_core.documents import Document
# from langchain_community.document_loaders import RecursiveUrlLoader

# from src.nisaa.helpers.logger import logger

# try:
#     from playwright.sync_api import sync_playwright
# except ImportError:
#     sync_playwright = None


# class SimpleHTMLTransformer:
#     """A tiny HTML -> text transformer that mimics a "transform_documents" API.
#     Keeps metadata and returns Document objects with cleaned page_content.
#     """

#     def transform_documents(self, documents: List[Document]) -> List[Document]:
#         out = []
#         for doc in documents:
#             try:
#                 soup = BeautifulSoup(doc.page_content or "", "html.parser")
#                 for s in soup(["script", "style", "noscript"]):
#                     s.extract()
#                 text = soup.get_text(separator=" ")
#                 text = " ".join(text.split())
#                 new_meta = dict(doc.metadata or {})
#                 out.append(Document(page_content=text, metadata=new_meta))
#             except Exception as e:
#                 logger.debug(f"Transformer failed for a document: {e}")
#         return out

# class WebsiteIngester:
#     """Handles website scraping and document conversion with progress tracking

#     Major fixes applied:
#     - Replaced langchain SitemapLoader (which can use asyncio) with a synchronous
#       sitemap parser to avoid "coroutine ... was never awaited" and "asyncio.run()"
#       issues inside a running event loop.
#     - Added missing attributes: html_transformer and stats initialization.
#     - Made sitemap fetching robust to encodings using chardet and supports sitemap index.
#     - Implemented a small, robust HTML-to-text transformer to replace html_transformer
#       dependency.
#     - Improved error handling and logging.
#     """

#     def __init__(self, company_namespace: str, proxies: Dict[str, str] = None):
#         self.company_namespace = company_namespace
#         self.proxies = proxies or {}
#         self.session = requests.Session()
#         if proxies:
#             self.session.proxies.update(proxies)

#         self.html_transformer = SimpleHTMLTransformer()
#         self.stats = {
#             "total_pages": 0,
#             "total_websites": 0,
#             "total_documents": 0,
#             "failed_websites": 0,
#         }

#     def get_website_info(self, url: str) -> Dict[str, str]:
#         """Extract metadata from website homepage"""
#         try:
#             resp = self.session.get(url, timeout=15)
#             resp.raise_for_status()
#             soup = BeautifulSoup(resp.text, "html.parser")

#             title = soup.title.string.strip() if soup.title and soup.title.string else "No Title"

#             desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find(
#                 "meta", attrs={"property": "og:description"}
#             )
#             description = (
#                 desc_tag["content"].strip()
#                 if desc_tag and desc_tag.get("content")
#                 else ""
#             )

#             keywords_tag = soup.find("meta", attrs={"name": "keywords"})
#             keywords = (
#                 keywords_tag["content"].strip()
#                 if keywords_tag and keywords_tag.get("content")
#                 else ""
#             )

#             heading = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""

#             if not description and soup.find("p"):
#                 description = soup.find("p").get_text(strip=True)[:200]

#             return {
#                 "title": title,
#                 "description": description,
#                 "keywords": keywords,
#                 "main_heading": heading,
#             }
#         except Exception as e:
#             logger.warning(f"Failed to extract website info from {url}: {e}")
#             return {
#                 "title": "Unknown",
#                 "description": "",
#                 "keywords": "",
#                 "main_heading": "",
#             }

#     def _parse_sitemap_urls(self, sitemap_url: str, max_urls: int = 1000) -> List[str]:
#         """Synchronous sitemap parser that extracts all <loc> entries.

#         Supports sitemap index files as well.
#         """
#         urls: List[str] = []
#         try:
#             resp = self.session.get(sitemap_url, timeout=15)
#             resp.raise_for_status()

#             detected = chardet.detect(resp.content)
#             encoding = detected.get("encoding", "utf-8") or "utf-8"
#             try:
#                 text = resp.content.decode(encoding, errors="strict")
#             except Exception:
#                 text = resp.content.decode(encoding, errors="ignore")

#             # if this is a sitemap index, find nested sitemaps first
#             soup = BeautifulSoup(text, "xml")
#             # prefer <loc> under <sitemap> (index) or under <url>
#             sitemap_tags = soup.find_all("sitemap")
#             if sitemap_tags:
#                 for s in sitemap_tags:
#                     loc = s.find("loc")
#                     if loc and loc.text:
#                         try:
#                             nested = self._parse_sitemap_urls(loc.text.strip(), max_urls - len(urls))
#                             urls.extend(nested)
#                             if len(urls) >= max_urls:
#                                 return urls[:max_urls]
#                         except Exception:
#                             continue

#             url_tags = soup.find_all("url")
#             for ut in url_tags:
#                 loc = ut.find("loc")
#                 if loc and loc.text:
#                     urls.append(loc.text.strip())
#                     if len(urls) >= max_urls:
#                         break

#             # fallback: if no url tags found, attempt to regex-locate hrefs
#             if not urls:
#                 for m in re.finditer(r"<loc>\s*(https?://[^<\s]+)\s*</loc>", text, re.I):
#                     urls.append(m.group(1))
#                     if len(urls) >= max_urls:
#                         break

#             return urls
#         except Exception as e:
#             logger.debug(f"Sitemap loading failed: {e}")
#             return []

#     def load_sitemap_with_encoding(self, sitemap_url: str) -> List[Document]:
#         """Load sitemap (synchronously) and return Documents with raw HTML fetched.

#         We fetch each URL in the sitemap and create a Document for it. This avoids
#         using async loaders inside a sync context.
#         """
#         documents: List[Document] = []
#         urls = self._parse_sitemap_urls(sitemap_url)
#         if not urls:
#             return []

#         for u in tqdm(urls, desc="Fetching sitemap pages", disable=False):
#             try:
#                 r = self.session.get(u, timeout=15)
#                 if r.status_code == 200 and r.text:
#                     documents.append(Document(page_content=r.text, metadata={"source": u}))
#             except Exception as e:
#                 logger.debug(f"Failed to fetch {u} from sitemap: {e}")

#         return documents

#     def extract_spa_links(self, url: str, max_pages: int = 200, pbar: tqdm = None) -> List[str]:
#         """Extract links from SPA using Playwright"""
#         if not sync_playwright:
#             logger.warning("Playwright not available for SPA crawling")
#             return []

#         visited, queue, all_links = set(), [url], set()
#         domain = urlparse(url).netloc

#         try:
#             with sync_playwright() as pw:
#                 browser = pw.chromium.launch(headless=True)
#                 context = browser.new_context()
#                 page = context.new_page()

#                 while queue and len(visited) < max_pages:
#                     current_url = queue.pop(0)
#                     if current_url in visited:
#                         continue

#                     try:
#                         page.goto(current_url, timeout=60000, wait_until="networkidle")
#                         time.sleep(1.2)
#                         html = page.content()
#                         visited.add(current_url)
#                         all_links.add(current_url)

#                         if pbar:
#                             pbar.set_postfix_str(f"{len(all_links)} pages found")

#                         soup = BeautifulSoup(html, "html.parser")
#                         for a in soup.select("a[href]"):
#                             href = a.get("href") or ""
#                             href = href.strip()
#                             if not href or href.startswith(("mailto:", "tel:", "javascript:")):
#                                 continue
#                             abs_url = href if href.startswith("http") else urljoin(current_url, href)
#                             if urlparse(abs_url).netloc != domain:
#                                 continue
#                             abs_url = abs_url.split("#")[0]
#                             if abs_url not in visited and abs_url not in queue:
#                                 queue.append(abs_url)
#                     except Exception as e:
#                         logger.debug(f"Error visiting {current_url}: {e}")
#                         continue

#                 browser.close()
#         except Exception as e:
#             logger.error(f"SPA crawling failed: {e}")

#         return sorted(all_links)

#     def is_spa_site(self, url: str) -> bool:
#         """Detect if website is a Single Page Application"""
#         try:
#             resp = self.session.get(url, timeout=10)
#             html = resp.text.lower()
#             spa_signals = [
#                 '<div id="root"',
#                 "window.__INITIAL_STATE__",
#                 "ng-app",
#                 "vue",
#                 "react",
#             ]
#             return any(signal in html for signal in spa_signals)
#         except Exception as e:
#             logger.debug(f"SPA detection failed: {e}")
#             return False

#     def scrape_website(self, url: str, pbar: tqdm = None) -> List[Document]:
#         """Scrape website using sitemap-first strategy with SPA fallback"""
#         documents: List[Document] = []

#         with tqdm(
#             total=100,
#             desc=f"{urlparse(url).netloc}",
#             leave=False,
#             bar_format="{l_bar}{bar}| {percentage:3.0f}%",
#         ) as progress:
#             # Get website metadata
#             progress.set_postfix_str("Fetching metadata...")
#             website_info = self.get_website_info(url)
#             progress.update(10)

#             # Strategy 1: Try sitemap
#             progress.set_postfix_str("Loading sitemap...")
#             sitemap_url = f"{url.rstrip('/')}/sitemap.xml"
#             documents = self.load_sitemap_with_encoding(sitemap_url)
#             progress.update(30)

#             if documents:
#                 logger.info(f"Loaded {len(documents)} pages from sitemap")
#                 progress.update(30)
#             else:
#                 # Strategy 2: Check if SPA
#                 progress.set_postfix_str("Checking if SPA...")
#                 if self.is_spa_site(url):
#                     logger.info("Detected SPA site, using dynamic crawling")
#                     progress.update(10)

#                     progress.set_postfix_str("Crawling SPA...")
#                     spa_urls = self.extract_spa_links(url, max_pages=200, pbar=progress)
#                     progress.update(20)

#                     if spa_urls:
#                         try:
#                             progress.set_postfix_str("Loading SPA pages (requests)...")
#                             # Fetch SPA URLs synchronously rather than relying on Selenium loader
#                             for u in tqdm(spa_urls, desc="Fetching SPA pages", disable=False):
#                                 try:
#                                     r = self.session.get(u, timeout=15)
#                                     if r.status_code == 200 and r.text:
#                                         documents.append(Document(page_content=r.text, metadata={"source": u}))
#                                 except Exception as e:
#                                     logger.debug(f"Failed to fetch SPA url {u}: {e}")

#                             logger.info(f"Loaded {len(documents)} pages from SPA")
#                             progress.update(20)
#                         except Exception as e:
#                             logger.warning(f"SPA fetch failed: {e}")
#                             progress.update(20)

#                 # Strategy 3: Recursive fallback
#                 if not documents:
#                     logger.info("Using recursive URL loader")
#                     progress.set_postfix_str("Recursive crawling...")
#                     try:

#                         loader = RecursiveUrlLoader(url=url, max_depth=2, prevent_outside=True)
#                         documents = loader.load()
#                         logger.info(f"Loaded {len(documents)} pages recursively")
#                         progress.update(40)
#                     except Exception as e:
#                         logger.error(f"Recursive loader failed: {e}")
#                         progress.update(40)

#             if documents:
#                 progress.set_postfix_str("Cleaning HTML...")
#                 cleaned_docs = self.html_transformer.transform_documents(documents)

#                 for doc in cleaned_docs:
#                     doc.page_content = " ".join(doc.page_content.split())
#                     doc.metadata.update(
#                         {
#                             "company_namespace": self.company_namespace,
#                             "source_type": "website",
#                             "website_url": url,
#                             "website_title": website_info["title"],
#                             "website_description": website_info["description"],
#                             "website_keywords": website_info["keywords"],
#                             "scraped_at": datetime.now().isoformat(),
#                         }
#                     )

#                 self.stats["total_pages"] += len(cleaned_docs)
#                 progress.update(30)
#                 progress.set_postfix_str(f"✓ {len(cleaned_docs)} pages")
#                 return cleaned_docs

#             progress.update(100)
#             progress.set_postfix_str("✗ No pages found")

#         return []

#     def ingest_multiple_websites(self, urls: List[str]) -> List[Document]:
#         """Ingest data from multiple websites with progress tracking"""
#         logger.info(f"Starting website ingestion for {len(urls)} URLs")

#         all_documents: List[Document] = []

#         with tqdm(
#             total=len(urls),
#             desc="Websites",
#             unit="site",
#             bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
#         ) as pbar:
#             for url in urls:
#                 try:
#                     docs = self.scrape_website(url, pbar)
#                     if docs:
#                         all_documents.extend(docs)
#                         self.stats["total_websites"] += 1
#                         self.stats["total_documents"] += len(docs)
#                     else:
#                         logger.warning(f"No documents scraped from {url}")
#                         self.stats["failed_websites"] += 1
#                 except Exception as e:
#                     logger.error(f"Failed to scrape {url}: {e}")
#                     self.stats["failed_websites"] += 1
#                 finally:
#                     pbar.update(1)

#         logger.info(
#             f"Website ingestion complete - {self.stats['total_documents']} documents from "
#             f"{self.stats['total_websites']} websites"
#         )

#         return all_documents



"""
Website scraping and ingestion - EXISTING LOGIC INTACT
"""
import time
import random
from typing import List, Dict
from datetime import datetime
from urllib.parse import urlparse, urljoin
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import chardet
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    SitemapLoader,
    RecursiveUrlLoader,
    SeleniumURLLoader,
)
from langchain_community.document_transformers import Html2TextTransformer
from tqdm import tqdm
from src.nisaa.helpers.logger import logger

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


def get_request_session(proxies: dict = None):
    """Return a requests session with retries, headers, and optional proxy"""
    session = requests.Session()

    retries = Retry(
        total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504, 429]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/118.0",
    ]
    session.headers.update({"User-Agent": random.choice(user_agents)})

    if proxies:
        session.proxies.update(proxies)

    return session


class WebsiteIngester:
    """Handles website scraping and document conversion with progress tracking"""

    def __init__(self, company_namespace: str, proxies: dict = None):
        self.company_namespace = company_namespace
        self.session = get_request_session(proxies=proxies)
        self.html_transformer = Html2TextTransformer()
        self.stats = {
            "total_websites": 0,
            "total_pages": 0,
            "total_documents": 0,
            "failed_websites": 0,
        }

    def get_website_info(self, url: str) -> Dict[str, str]:
        """Extract metadata from website homepage"""
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            title = soup.title.string.strip() if soup.title else "No Title"

            desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find(
                "meta", attrs={"property": "og:description"}
            )
            description = (
                desc_tag["content"].strip()
                if desc_tag and desc_tag.get("content")
                else ""
            )

            keywords_tag = soup.find("meta", attrs={"name": "keywords"})
            keywords = (
                keywords_tag["content"].strip()
                if keywords_tag and keywords_tag.get("content")
                else ""
            )

            heading = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""

            if not description and soup.find("p"):
                description = soup.find("p").get_text(strip=True)[:200]

            return {
                "title": title,
                "description": description,
                "keywords": keywords,
                "main_heading": heading,
            }
        except Exception as e:
            logger.warning(f"Failed to extract website info from {url}: {e}")
            return {
                "title": "Unknown",
                "description": "",
                "keywords": "",
                "main_heading": "",
            }

    def load_sitemap_with_encoding(self, sitemap_url: str) -> List[Document]:
        """Load sitemap with encoding detection"""
        try:
            resp = self.session.get(sitemap_url, timeout=15)
            resp.raise_for_status()

            detected = chardet.detect(resp.content)
            encoding = detected.get("encoding", "utf-8") or "utf-8"

            try:
                text = resp.content.decode(encoding, errors="strict")
            except UnicodeDecodeError:
                logger.warning(
                    f"Decoding with {encoding} failed, using errors='ignore'"
                )
                text = resp.content.decode(encoding, errors="ignore")

            sitemap_loader = SitemapLoader(web_path=sitemap_url)
            return sitemap_loader.load()
        except Exception as e:
            logger.debug(f"Sitemap loading failed: {e}")
            return []

    def extract_spa_links(
        self, url: str, max_pages: int = 200, pbar: tqdm = None
    ) -> List[str]:
        """Extract links from SPA using Playwright"""
        if not sync_playwright:
            logger.warning("Playwright not available for SPA crawling")
            return []

        visited, queue, all_links = set(), [url], set()
        domain = urlparse(url).netloc

        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                context = browser.new_context()
                page = context.new_page()

                while queue and len(visited) < max_pages:
                    current_url = queue.pop(0)
                    if current_url in visited:
                        continue

                    try:
                        page.goto(current_url, timeout=60000, wait_until="networkidle")
                        time.sleep(1.5)
                        html = page.content()
                        visited.add(current_url)
                        all_links.add(current_url)

                        if pbar:
                            pbar.set_postfix_str(f"{len(all_links)} pages found")

                        soup = BeautifulSoup(html, "html.parser")
                        for a in soup.select("a[href]"):
                            href = a["href"].strip()
                            if not href or href.startswith(
                                ("mailto:", "tel:", "javascript:")
                            ):
                                continue
                            abs_url = (
                                href
                                if href.startswith("http")
                                else urljoin(current_url, href)
                            )
                            if urlparse(abs_url).netloc != domain:
                                continue
                            abs_url = abs_url.split("#")[0]
                            if abs_url not in visited and abs_url not in queue:
                                queue.append(abs_url)
                    except Exception as e:
                        logger.debug(f"Error visiting {current_url}: {e}")
                        continue

                browser.close()
        except Exception as e:
            logger.error(f"SPA crawling failed: {e}")

        return sorted(all_links)

    def is_spa_site(self, url: str) -> bool:
        """Detect if website is a Single Page Application"""
        try:
            resp = self.session.get(url, timeout=10)
            html = resp.text.lower()
            spa_signals = [
                '<div id="root"',
                "window.__INITIAL_STATE__",
                "ng-app",
                "vue",
                "react",
            ]
            return any(signal in html for signal in spa_signals)
        except Exception as e:
            logger.debug(f"SPA detection failed: {e}")
            return False

    def scrape_website(self, url: str, pbar: tqdm = None) -> List[Document]:
        """Scrape website using sitemap-first strategy with SPA fallback"""
        documents = []

        with tqdm(
            total=100,
            desc=f"  🌐 {urlparse(url).netloc}",
            leave=False,
            bar_format="{l_bar}{bar}| {percentage:3.0f}%",
        ) as progress:
            progress.set_postfix_str("Fetching metadata...")
            website_info = self.get_website_info(url)
            progress.update(10)

            progress.set_postfix_str("Loading sitemap...")
            sitemap_url = f"{url.rstrip('/')}/sitemap.xml"
            documents = self.load_sitemap_with_encoding(sitemap_url)
            progress.update(30)

            if documents:
                logger.info(f"Loaded {len(documents)} pages from sitemap")
                progress.update(30)
            else:
                progress.set_postfix_str("Checking if SPA...")
                if self.is_spa_site(url):
                    logger.info("Detected SPA site, using dynamic crawling")
                    progress.update(10)

                    progress.set_postfix_str("Crawling SPA...")
                    spa_urls = self.extract_spa_links(url, max_pages=200, pbar=progress)
                    progress.update(20)

                    if spa_urls:
                        try:
                            progress.set_postfix_str("Loading pages...")
                            loader = SeleniumURLLoader(
                                urls=spa_urls,
                                headless=True,
                                browser="chrome",
                                continue_on_failure=True,
                            )
                            documents = loader.load()
                            logger.info(f"Loaded {len(documents)} pages from SPA")
                            progress.update(20)
                        except Exception as e:
                            logger.warning(f"Selenium loader failed: {e}")
                            progress.update(20)

                if not documents:
                    logger.info("Using recursive URL loader")
                    progress.set_postfix_str("Recursive crawling...")
                    try:
                        loader = RecursiveUrlLoader(
                            url=url, max_depth=2, prevent_outside=True
                        )
                        documents = loader.load()
                        logger.info(f"Loaded {len(documents)} pages recursively")
                        progress.update(40)
                    except Exception as e:
                        logger.error(f"Recursive loader failed: {e}")
                        progress.update(40)

            if documents:
                progress.set_postfix_str("Cleaning HTML...")
                cleaned_docs = self.html_transformer.transform_documents(documents)

                for doc in cleaned_docs:
                    doc.page_content = " ".join(doc.page_content.split())
                    doc.metadata.update(
                        {
                            "company_namespace": self.company_namespace,
                            "source_type": "website",
                            "website_url": url,
                            "website_title": website_info["title"],
                            "website_description": website_info["description"],
                            "website_keywords": website_info["keywords"],
                            "scraped_at": datetime.now().isoformat(),
                        }
                    )

                self.stats["total_pages"] += len(cleaned_docs)
                progress.update(30)
                progress.set_postfix_str(f"✓ {len(cleaned_docs)} pages")
                return cleaned_docs

            progress.update(100)
            progress.set_postfix_str("✗ No pages found")

        return []

    def ingest_multiple_websites(self, urls: List[str]) -> List[Document]:
        """Ingest data from multiple websites with progress tracking"""
        if not urls:
            return []
            
        logger.info(f"🌐 Starting website ingestion for {len(urls)} URLs")

        all_documents = []

        with tqdm(
            total=len(urls),
            desc="🌐 Websites",
            unit="site",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
        ) as pbar:
            for url in urls:
                try:
                    docs = self.scrape_website(url, pbar)
                    if docs:
                        all_documents.extend(docs)
                        self.stats["total_websites"] += 1
                        self.stats["total_documents"] += len(docs)
                    else:
                        logger.warning(f"No documents scraped from {url}")
                        self.stats["failed_websites"] += 1
                except Exception as e:
                    logger.error(f"Failed to scrape {url}: {e}")
                    self.stats["failed_websites"] += 1
                finally:
                    pbar.update(1)

        logger.info(
            f"✅ Website ingestion complete - {self.stats['total_documents']} documents from "
            f"{self.stats['total_websites']} websites"
        )

        return all_documents