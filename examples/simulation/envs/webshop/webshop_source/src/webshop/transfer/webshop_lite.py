import os

from flask import render_template_string, Flask
from .predict_help import Page

app=Flask(__name__)
app.debug=True

SESSION_ID = "ABC"
TEMPLATE_DIR = "../web_agent_site/templates/"
KEYWORDS = ["placeholder (not needed)"] # To Do: Does this matter?
QUERY = ""
product_map = {}

def read_html_template(path):
    with open(path) as f:
        template = f.read()
    return template

@app.route('/', methods=['GET', 'POST'])
def index(session_id, **kwargs):
    print("Hello world")

@app.route('/', methods=['GET', 'POST'])
def search_results(data):
    path = os.path.join(TEMPLATE_DIR, 'results_page.html')
    html = render_template_string(
        read_html_template(path=path),
        session_id=SESSION_ID,
        products=data,
        keywords=KEYWORDS,
        page=1,
        total=len(data),
        instruction_text=QUERY,
    )
    return html

@app.route('/', methods=['GET', 'POST'])
def item_page(session_id, asin, keywords, page, options):
    path = os.path.join(TEMPLATE_DIR, 'item_page.html')
    html = render_template_string(
        read_html_template(path=path),
        session_id=session_id,
        product_info=product_map[asin],
        keywords=keywords,
        page=page,
        asin=asin,
        options=options,
        instruction_text=QUERY
    )
    return html

@app.route('/', methods=['GET', 'POST'])
def item_sub_page(session_id, asin, keywords, page, sub_page, options):
    path = os.path.join(TEMPLATE_DIR, sub_page.value.lower() + "_page.html")
    html = render_template_string(
        read_html_template(path),
        session_id=session_id, 
        product_info=product_map[asin],
        keywords=keywords,
        page=page,
        asin=asin,
        options=options,
        instruction_text=QUERY
    )
    return html

@app.route('/', methods=['GET', 'POST'])
def done(asin, options, session_id, **kwargs):
    path = os.path.join(TEMPLATE_DIR, 'done_page.html')
    html = render_template_string(
        read_html_template(path),
        session_id=session_id,
        reward=1,
        asin=asin,
        options=product_map[asin]["options"],
        reward_info=kwargs.get('reward_info'),
        goal_attrs=kwargs.get('goal_attrs'),
        purchased_attrs=kwargs.get('purchased_attrs'),
        goal=kwargs.get('goal'),
        mturk_code=kwargs.get('mturk_code'),
        query=kwargs.get('query'),
        category=kwargs.get('category'),
        product_category=kwargs.get('product_category'),
    )
    return html
    
# Project Dictionary Information onto Fake Amazon
def dict_to_fake_html(data, page_type, asin=None, sub_page_type=None, options=None, prod_map={}, query=""):
    global QUERY, product_map
    QUERY = query
    product_map = prod_map
    with app.app_context(), app.test_request_context():
        if page_type == Page.RESULTS:
            return search_results(data)
        if page_type == Page.ITEM_PAGE:
            return item_page(SESSION_ID, asin, KEYWORDS, 1, options)
        if page_type == Page.SUB_PAGE:
            if sub_page_type is not None:
                return item_sub_page(SESSION_ID, asin, KEYWORDS, 1, sub_page_type, options)
            else:
                raise Exception("Sub page of type", sub_page_type, "unrecognized")