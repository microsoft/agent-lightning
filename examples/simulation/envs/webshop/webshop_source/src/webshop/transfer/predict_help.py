from bs4 import BeautifulSoup
from bs4.element import Comment
from enum import Enum
import re, time
from urllib.parse import urlencode

import json, requests, torch

class Page(Enum):
    DESC = "description"
    FEATURES = "features"
    ITEM_PAGE = "item_page"
    RESULTS = "results"
    REVIEWS = "reviews"
    SEARCH = "search"
    SUB_PAGE = "item_sub_page"

HEADER_ = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.64 Safari/537.36'
DEBUG_HTML = "temp.html"
NUM_PROD_LIMIT = 10

WEBSHOP_URL = "http://3.83.245.205:3000"
WEBSHOP_SESSION = "abc"


def parse_results_ebay(query, page_num=None, verbose=True):
    query_string = '+'.join(query.split())
    page_num = 1 if page_num is None else page_num
    url = f'https://www.ebay.com/sch/i.html?_nkw={query_string}&_pgn={page_num}'
    if verbose:
        print(f"Search Results URL: {url}")
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    soup = BeautifulSoup(webpage.text, 'html.parser')
    products = soup.select('.s-item__wrapper.clearfix')

    results = []
    for item in products[:NUM_PROD_LIMIT]:
        title = item.select_one('.s-item__title').text.strip()
        if "shop on ebay" in title.lower():
            # Skip "Shop on ebay" product title
            continue
        link = item.select_one('.s-item__link')['href']
        asin = link.split("?")[0][len("https://www.ebay.com/itm/"):]

        try:
            price = item.select_one('.s-item__price').text
            if "to" in price:
                prices = price.split(" to ")
                price = [p.strip("$") for p in prices]
        except:
            price = None
        
        results.append({
            "asin": asin,
            "Title": title,
            "Price": price
        })
    if verbose:
        print(f"Scraped {len(results)} products")
    return results


def parse_item_page_ebay(asin, verbose=True):
    product_dict = {}
    product_dict["asin"] = asin
    
    url = f"https://www.ebay.com/itm/{asin}"
    if verbose:
        print(f"Item Page URL: {url}")
    begin = time.time()
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    end = time.time()
    if verbose:
        print(f"Item page scraping took {end-begin} seconds")
    soup = BeautifulSoup(webpage.content, "html.parser")

    # Title
    try:
        product_dict["Title"] = soup.find('h1', {'class': 'x-item-title__mainTitle'}).text.strip()
    except:
        product_dict["Title"] = "N/A"

    # Price: Get price string, extract decimal numbers from string
    try:
        price_str = soup.find('div', {'class': 'mainPrice'}).text
        prices = re.findall('\d*\.?\d+', price_str)
        product_dict["Price"] = prices[0]
    except:
        product_dict["Price"] = "N/A"

     # Main Image
    try:
        img_div = soup.find('div', {'id': 'mainImgHldr'})
        img_link = img_div.find('img', {'id': 'icImg'})["src"]
        product_dict["MainImage"] = img_link
    except:
        product_dict["MainImage"] = ""
    
    # Rating
    try:
        rating = soup.find('span', {'class': 'reviews-star-rating'})["title"].split()[0]
    except:
        rating = None
    product_dict["Rating"] = rating

    # Options
    options, options_to_images = {}, {} # TODO: options_to_images possible?
    try:
        option_blocks = soup.findAll('select', {'class': 'msku-sel'})
        for block in option_blocks:
            name = block["name"].strip().strip(":")
            option_tags = block.findAll("option")
            opt_list = []
            for option_tag in option_tags:
                if "select" not in option_tag.text.lower():
                    # Do not include "- select -" (aka `not selected`) choice
                    opt_list.append(option_tag.text)
            options[name] = opt_list
    except:
        options = {}
    product_dict["options"], product_dict["option_to_image"] = options, options_to_images

    # Description
    desc = None
    try:
        # Ebay descriptions are shown in `iframe`s
        desc_link = soup.find('iframe', {'id': 'desc_ifr'})["src"]
        desc_webpage = requests.get(desc_link, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
        desc_soup = BeautifulSoup(desc_webpage.content, "html.parser")
        desc = ' '.join(desc_soup.text.split())
    except:
        desc = "N/A"
    product_dict["Description"] = desc

    # Features
    features = None
    try:
        features = soup.find('div', {'class': 'x-about-this-item'}).text
    except:
        features = "N/A"
    product_dict["BulletPoints"] = features

    return product_dict
    

def parse_results_ws(query, page_num=None, verbose=True):
    query_string = '+'.join(query.split())
    page_num = 1 if page_num is None else page_num
    url = (
        f'{WEBSHOP_URL}/search_results/{WEBSHOP_SESSION}/'
        f'{query_string}/{page_num}'
    )
    if verbose:
        print(f"Search Results URL: {url}")
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    soup = BeautifulSoup(webpage.content, 'html.parser')
    products = soup.findAll('div', {'class': 'list-group-item'})

    results = []
    for product in products:
        asin = product.find('a', {'class': 'product-link'})
        title = product.find('h4', {'class': 'product-title'})
        price = product.find('h5', {'class': 'product-price'})

        if "\n" in title:
            title = title.text.split("\n")[0].strip()
        else:
            title = title.text.strip().strip("\n")

        if "to" in price.text:
            # Parse if price presented as range
            prices = price.text.split(" to ")
            price = [float(p.strip().strip("\n$")) for p in prices]
        else:
            price = float(price.text.strip().strip("\n$"))

        results.append({
            "asin": asin.text,
            "Title": title,
            "Price": price
        })

    if verbose:
        print(f"Scraped {len(results)} products")
    return results


def parse_item_page_ws(asin, query, page_num, options, verbose=True):
    product_dict = {}
    product_dict["asin"] = asin

    query_string = '+'.join(query.split())
    options_string = json.dumps(options)
    url = (
        f'{WEBSHOP_URL}/item_page/{WEBSHOP_SESSION}/'
        f'{asin}/{query_string}/{page_num}/{options_string}'
    )
    if verbose:
        print(f"Item Page URL: {url}")
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    soup = BeautifulSoup(webpage.content, 'html.parser')

    # Title, Price, Rating, and MainImage
    product_dict["Title"] = soup.find('h2').text
    
    h4_headers = soup.findAll("h4")
    for header in h4_headers:
        text = header.text
        if "Price" in text:
            product_dict["Price"] = text.split(":")[1].strip().strip("$")
        elif "Rating" in text:
            product_dict["Rating"] = text.split(":")[1].strip()
    
    product_dict["MainImage"] = soup.find('img')['src']

    # Options
    options, options_to_image = {}, {}
    option_blocks = soup.findAll("div", {'class': 'radio-toolbar'})
    for block in option_blocks:
        name = block.find("input")["name"]
        labels = block.findAll("label")
        inputs = block.findAll("input")
        opt_list = []
        for label, input in zip(labels, inputs):
            opt = label.text
            opt_img_path = input["onclick"].split("href=")[1].strip('\';')
            opt_img_url = f'{WEBSHOP_URL}{opt_img_path}'

            opt_list.append(opt)
            options_to_image[opt] = opt_img_url
        options[name] = opt_list
    product_dict["options"] = options
    product_dict["option_to_image"] = options_to_image

    # Description
    url = (
        f'{WEBSHOP_URL}/item_sub_page/{WEBSHOP_SESSION}/'
        f'{asin}/{query_string}/{page_num}/Description/{options_string}'
    )
    if verbose:
        print(f"Item Description URL: {url}")
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    soup = BeautifulSoup(webpage.content, 'html.parser')
    product_dict["Description"] = soup.find(name="p", attrs={'class': 'product-info'}).text.strip()

    # Features
    url = (
        f'{WEBSHOP_URL}/item_sub_page/{WEBSHOP_SESSION}/'
        f'{asin}/{query_string}/{page_num}/Features/{options_string}'
    )
    if verbose:
        print(f"Item Features URL: {url}")
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    soup = BeautifulSoup(webpage.content, 'html.parser')
    bullets = soup.find(name="ul").findAll(name="li")
    product_dict["BulletPoints"] = '\n'.join([b.text.strip() for b in bullets])

    return product_dict


# Query -> Search Result ASINs
def parse_results_amz(query, page_num=None, verbose=True):
    url = 'https://www.amazon.com/s?k=' + query.replace(" ", "+")
    if page_num is not None:
        url += "&page=" + str(page_num)
    if verbose:
        print(f"Search Results URL: {url}")
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    soup = BeautifulSoup(webpage.content, 'html.parser')
    products = soup.findAll('div', {'data-component-type': 's-search-result'})
    if products is None:
        temp = open(DEBUG_HTML, "w")
        temp.write(str(soup))
        temp.close()
        raise Exception("Couldn't find search results page, outputted html for inspection")
    results = []

    for product in products[:NUM_PROD_LIMIT]:
        asin = product['data-asin']
        title = product.find("h2", {'class': "a-size-mini"})
        price_div = product.find("div", {'class': 's-price-instructions-style'})
        price = price_div.find("span", {'class': 'a-offscreen'})

        result = {
            'asin': asin,
            'Title': title.text.strip(),
            'Price': price.text.strip().strip("$")
        }
        results.append(result)
    if verbose:
        print("Scraped", len(results), "products")
    return results


# Scrape information of each product
def parse_item_page_amz(asin, verbose=True):
    product_dict = {}
    product_dict["asin"] = asin

    url = f"https://www.amazon.com/dp/{asin}"
    if verbose:
        print("Item Page URL:", url)
    begin = time.time()
    webpage = requests.get(url, headers={'User-Agent': HEADER_, 'Accept-Language': 'en-US, en;q=0.5'})
    end = time.time()
    if verbose:
        print(f"Item page scraping took {end-begin} seconds")
    soup = BeautifulSoup(webpage.content, "html.parser")

    # Title
    try:
        title = soup.find("span", attrs={"id": 'productTitle'})
        title = title.string.strip().replace(',', '')
    except AttributeError:
        title = "N/A"
    product_dict["Title"] = title
 
    # Price
    try:
        parent_price_span = soup.find(name="span", class_="apexPriceToPay")
        price_span = parent_price_span.find(name="span", class_="a-offscreen")
        price = float(price_span.getText().replace("$", ""))
    except AttributeError:
        price = "N/A"
    product_dict["Price"] = price

    # Rating
    try:
        rating = soup.find(name="span", attrs={"id": "acrPopover"})
        if rating is None:
            rating = "N/A"
        else:
            rating = rating.text
    except AttributeError:
        rating = "N/A"
    product_dict["Rating"] = rating.strip("\n").strip()
 
    # Features
    try:
        features = soup.find(name="div", attrs={"id": "feature-bullets"}).text
    except AttributeError:
        features = "N/A"
    product_dict["BulletPoints"] = features
    
    # Description
    try:
        desc_body = soup.find(name="div", attrs={"id": "productDescription_feature_div"})
        desc_div = desc_body.find(name="div", attrs={"id": "productDescription"})
        desc_ps = desc_div.findAll(name="p")
        desc = " ".join([p.text for p in desc_ps])
    except AttributeError:
        desc = "N/A"
    product_dict["Description"] = desc.strip()

    # Main Image
    try:
        imgtag = soup.find("img", {"id":"landingImage"})
        imageurl = dict(imgtag.attrs)["src"]
    except AttributeError:
        imageurl = ""
    product_dict["MainImage"] = imageurl

    # Options
    options, options_to_image = {}, {}
    try:
        option_body = soup.find(name='div', attrs={"id": "softlinesTwister_feature_div"})
        if option_body is None:
            option_body = soup.find(name='div', attrs={"id": "twister_feature_div"})
        option_blocks = option_body.findAll(name='ul')
        for block in option_blocks:
            name = json.loads(block["data-a-button-group"])["name"]
            # Options
            opt_list = []
            for li in block.findAll("li"):
                img = li.find(name="img")
                if img is not None:
                    opt = img["alt"].strip()
                    opt_img = img["src"]
                    if len(opt) > 0:
                        options_to_image[opt] = opt_img
                else:
                    opt = li.text.strip()
                if len(opt) > 0:
                    opt_list.append(opt)
            options[name.replace("_name", "").replace("twister_", "")] = opt_list
    except AttributeError:
        options = {}
    product_dict["options"], product_dict["option_to_image"] = options, options_to_image
    return product_dict


# Get text observation from html
# TODO[john-b-yang]: Similar to web_agent_site/envs/...text_env.py func def, merge?
def convert_html_to_text(html, simple=False, clicked_options=None, visited_asins=None):
    def tag_visible(element):
        ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
        return (
            element.parent.name not in ignore and not isinstance(element, Comment)
        )
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    if simple:
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        observation = ''
        for t in visible_texts:
            if t == '\n': continue
            if t.parent.name == 'button':  # button
                processed_t = f'[button] {t} [button]'
            elif t.parent.name == 'label':  # options
                if f'{t}' in clicked_options:
                    processed_t = f'  [clicked button] {t} [clicked button]'
                    observation = f'You have clicked {t}.\n' + observation
                else:
                    processed_t = f'  [button] {t} [button]'
            elif t.parent.get('class') == ["product-link"]: # asins
                if f'{t}' in visited_asins:
                    processed_t = f'\n[clicked button] {t} [clicked button]'
                else:
                    processed_t = f'\n[button] {t} [button]'
            else: # regular, unclickable text
                processed_t =  str(t)
            observation += processed_t + '\n'
        return observation


# Get action from dict of values retrieved from html
def convert_dict_to_actions(page_type, products=None, asin=None, page_num=None) -> dict:
    info = {"valid": []}
    if page_type == Page.RESULTS:
        info["valid"] = ['click[back to search]']
        if products is None or page_num is None:
            print(page_num)
            print(products)
            raise Exception('Provide `products`, `page_num` to get `results` valid actions')
        # Decide whether to add `next >` as clickable based on # of search results
        if len(products) > 10:
            info["valid"].append('click[next >]')
        # Add `< prev` as clickable if not first page of search results
        if page_num > 1:
            info["valid"].append('click[< prev]')
        for product in products:
            info["valid"].append("click[item - " + product["Title"] + "]")
    if page_type == Page.ITEM_PAGE:
        if products is None or asin is None:
            raise Exception('Provide `products` and `asin` to get `item_page` valid actions')
        info["valid"] = ['click[back to search]', 'click[< prev]', 'click[description]',\
            'click[features]', 'click[buy now]'] # To do: reviews
        if "options" in products[asin]:
            for key, values in products[asin]["options"].items():
                for value in values:
                    info["valid"].append("click[" + value + "]")
    if page_type == Page.SUB_PAGE:
        info["valid"] = ['click[back to search]', 'click[< prev]']
    info['image_feat'] = torch.zeros(512)
    return info