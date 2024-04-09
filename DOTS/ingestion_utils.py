import os, glob, re, requests, validators
from typing import List, Dict, Tuple
import pandas as pd
import pytesseract
from bs4 import BeautifulSoup
#import logging

from urllib.parse import urlparse, urljoin

from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import undetected_chromedriver as uc

import graphistry
from graphistry import Plottable


# some light help on BeatifulSoup pages that come back crazy
def reduce_newlines(text: str, max_newlines: int = 1) -> str:
    # Split the text into lines
    lines = text.split('\n')
    # print(f'Reducing {len(lines)} newlines...')

    # Remove empty lines and reduce consecutive newlines
    reduced_lines = []
    prev_line_empty = False
    for line in lines:
        if line.strip():
            reduced_lines.append(line)
            prev_line_empty = False
        elif not prev_line_empty:
            reduced_lines.extend([''] * min(max_newlines, 1))
            prev_line_empty = True

    # Join the reduced lines with specified number of newlines
    reduced_text = ('\n' * max_newlines).join(reduced_lines)
    chunks = reduced_text.split('\n')
    # print(f"-- {len(chunks)} paragraphs")
    return reduced_text


def simple_scrape(url: str) -> str:
    print('-- Simple Scraping', url)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return reduce_newlines(soup.get_text())


def scrape_selenium_headless(url: str, browser: str = "Firefox", wait_time: int = 5, page_load_strategy: str = 'eager') -> str:
    driver = None
    if browser == "Chrome":
        options = ChromeOptions()
        options.add_argument("--headless")
        options.page_load_strategy = page_load_strategy
        webdriver_service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=webdriver_service, options=options)
    elif browser == "undetected_chrome":
        options = uc.ChromeOptions()
        driver = uc.Chrome(headless=True,use_subprocess=False)
    elif browser == "Firefox":
        options = FirefoxOptions()
        options.add_argument("--headless")
        options.page_load_strategy = 'eager'
        options.set_preference("javascript.enabled", JS)
        options.page_load_strategy = page_load_strategy
        driver = webdriver.Firefox(options=options)
    # else:
        # raise ValueError(f"Unsupported browser: {browser}, must be one of 'Chrome' or 'Firefox'")

    # html = ''
    for _ in range(3):
        # try:
        driver.set_page_load_timeout(wait_time+1)
        driver.get(url)
        WebDriverWait(driver, wait_time).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        see = reduce_newlines(soup.get_text())
        parsed = urlparse(see)
        if 'http'  in see:
            url = (parsed.geturl())
            url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
            url = re.findall(url_pattern, url)
            url = ''.join(url)
            # print(url)

        if 'http' not in see:
            break
    driver.quit()
    return see


def iter_pull(url, JS = True, depth = 3):
    options = FirefoxOptions()
    options.add_argument("--headless")
    options.page_load_strategy = 'eager'
    options.set_preference("javascript.enabled", JS)
    driver = webdriver.Firefox(options=options)
    for _ in range(depth):
        driver.set_page_load_timeout(5)
        driver.get(url)
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        see = reduce_newlines(soup.get_text())
        parsed = urlparse(see)
        if 'http'  in see:
            url = (parsed.geturl())
            url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
            url = re.findall(url_pattern, url)
            url = ''.join(url)
            # print(url)

        if 'http' not in see:
            break
    driver.quit()
    return see

def safe_iter_pull(url):
    try:
        return scrape_selenium_headless(url)
    except Exception as e:
        print(f"Error: {e}")
    try:
        return iter_pull(url, JS= False)
    except Exception as e:
        print(f"Error: {e}")
    try:
        return scrape_selenium_headless(url,browser='undetected_chrome')
    except Exception as e:
        print(f"Error: {e}")  # hopefully not MSN
        return []
        


class WebCrawler:
    def __init__(self, base_urls: List[str], exclude_patterns: List[str] = [], use_ocr: bool=False, 
                 headless: bool=False, wait_time: int=10, browser: str="Firefox", page_load_strategy: str = 'eager'):
        """
        Initialize the WebCrawler with the given parameters.

        Args:
            base_urls (List[str]): A list of URLs which the web crawler will not go beyond. 
                                   It will only scrape URLs that contain any of these base URLs.
            exclude_patterns (List[str], optional): A list of URL patterns to exclude from the scraping process. 
                                                     Any URL containing any of these patterns will not be scraped. 
                                                     Defaults to None.
            use_ocr (bool, optional): A flag indicating whether OCR should be used for web scraping.
                                      If true, Selenium and pytesseract will be used to scrape the web pages. 
                                      Defaults to False.
            headless (bool, optional): A flag indicating whether to use the headless mode of Selenium for web scraping.
                                       If true, Selenium's headless mode will be used. 
                                       Defaults to True.
            wait_time (int, optional): The maximum time to wait for a page to load when using Selenium. 
                                       Defaults to 10 seconds.

        Examples:
            crawler = WebCrawler(base_urls=["example.com"], use_ocr=False, exclude_patterns=["/blog", "/signup"])
            crawler.crawl_webpage("http://example.com/getting-started", depth=3)
            nodes, edges = crawler.get_results()
            g = crawler.to_graphistry()
        """
        self.set_base_urls(base_urls)
        self.exclude_patterns = exclude_patterns
        self.page_load_strategy = page_load_strategy
        self.use_ocr = use_ocr
        if use_ocr:
            raise NotImplementedError("OCR is not supported yet.")
        self.headless = headless
        if headless:
            raise NotImplementedError("Headless mode is not supported yet.")
        self.wait_time = wait_time
        self.browser = browser
        self.nodes_df = pd.DataFrame(columns=['url', 'text'])
        self.edges_df = pd.DataFrame(columns=['from_url', 'to_url'])
        self.visited_urls: set = set()
    
    def set_base_urls(self, base_urls: List[str]) -> None:
        self.base_urls = [urlparse(url)._replace(scheme='', fragment='', query='', params='').geturl().strip('/') for url in base_urls]

    def is_valid_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if any(pattern in parsed.path for pattern in self.exclude_patterns):
            return False
        return parsed.netloc in self.base_urls
     
    def extract_links(self, url: str) -> List[str]:
        content = requests.get(url).content
        soup = BeautifulSoup(content, 'html.parser')

        links = []
        for a in soup.find_all('a', href=True):
            link = a['href']
            absolute_link = urljoin(url, link)
            if self.is_valid_url(absolute_link) and absolute_link not in self.visited_urls:
                links.append(absolute_link)
        return links

    def scrape_link(self, url: str) -> str:
        if self.use_ocr:
            raise NotImplementedError("OCR is not supported yet.")
            #content = scrape_selenium_ocr(url, self.wait_time, self.browser)
        elif url.endswith('.pdf'):
            pdf_file_path = get_pdf_from_url(url)
            content = extract_text_from_pdf(pdf_file_path)
        elif url.endswith(('.jpg', '.jpeg', '.png')):
            filename = get_file_from_url(url, 'temp_image.jpg')
            content = pytesseract.image_to_string(Image.open(filename))
            print(f"OCR on image {filename} completed successfully.")
        elif self.headless:
            raise NotImplementedError("Headless mode is not supported yet.")
            #content = scrape_selenium_headless(url, self.browser, self.wait_time, self.page_load_strategy)
        else:
            content = simple_scrape(url)
        return content

    def crawl_webpage(self, url: str, depth: int = 3) -> None:
        if depth < 0:
            return
        
        if url not in self.visited_urls:
            print(f"Processing URL {url}")
            self.visited_urls.add(url)
            try:
                content = self.scrape_link(url)
                valid_links = self.extract_links(url)
                # Add current URL and its content to nodes_df
                self.nodes_df = self.nodes_df.append({'url': url, 'text': content}, ignore_index=True)  # type: ignore
                print(f"Found {len(valid_links)} valid links on {url}")
                for link in valid_links:
                    # Add edges to edges_df
                    self.edges_df = self.edges_df.append({'from_url': url, 'to_url': link}, ignore_index=True)  # type: ignore
                    self.crawl_webpage(link, depth=depth-1)  # recursive call for valid link
            except Exception as e:
                print(f"*****Failed to process URL {url}: {e}")

    def get_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.nodes_df, self.edges_df

    def to_graphistry(self, drop_duplicates: bool = True) -> Plottable:
        if drop_duplicates:
            print("Dropping duplicate nodes and edges...")
            nodes_df = self.nodes_df.drop_duplicates(subset=['url'])
            edges_df = self.edges_df.drop_duplicates(subset=['from_url', 'to_url'])
        else:
            nodes_df = self.nodes_df
            edges_df = self.edges_df
        return graphistry.edges(edges_df, source='from_url', destination='to_url').nodes(nodes_df, node='url')
    
    
    
