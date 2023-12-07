import hydra
import json, re
import regex
import yaml
import os, shutil
import pandas as pd
from omegaconf import DictConfig
from math import floor, ceil
from bs4 import BeautifulSoup
from urllib.parse import unquote
from time import sleep
from pathlib import Path
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_paths_real_sample(n=100, paths_real_file='wikispeedia/wikispeedia_paths-and-graph/paths_real_processed.tsv', experiment=None) -> list[dict]:
    """
    Gets a sample of unique paths from the paths_real_processed.tsv file.
    Note: right now returns only the src and tgt pages.
    """
    assert experiment in ['uniform_spl', 'human_test_set', None], f"Experiment {experiment} not supported."

    paths_real_df = pd.read_csv(paths_real_file, sep='\t')
    paths_real_sample_df = []
    if experiment == 'uniform_spl':
        spl_vals = [2,3,4,5,6]
        n_per_spl = int(n/len(spl_vals))
        for spl_val in spl_vals:
            df = paths_real_df.groupby(['src', 'tgt', 'spl']).agg({"shpl": "min", "shepl": "min", 'user_path_rating': 'mean'}).reset_index()
            paths_real_sample_df.append(df[df["spl"] == spl_val].sample(n=n_per_spl, random_state=42))
        paths_real_sample_df = pd.concat(paths_real_sample_df)
    elif experiment == 'human_test_set':
        # sample uniformly from games where is_path_finished==1 count >=4
        # paths_real_sample_df = paths_real_df[paths_real_df["is_path_finished"] == 1].groupby(["src", "tgt"]).agg({"is_path_finished": "count", "spl": "min", "mhepl": "min"}).query("is_path_finished >= 4").sample(n=n, random_state=42)
        df = paths_real_df[paths_real_df["is_path_finished"] == 1].groupby(["src", "tgt"]).agg({"is_path_finished": "count", "spl": "min", "mhepl": "min"}).query("is_path_finished >= 4").reset_index()
        paths_real_sample_df = df.sample(n=n, random_state=42)
    else:
        paths_real_sample_df = paths_real_df.groupby(['src', 'tgt']).agg({"shpl": "min", "shepl": "min"}).reset_index().sample(n=n, random_state=42)
    
    paths_real_sample = []
    for i, row in paths_real_sample_df.iterrows():
        paths_real_sample.append({"src": row["src"], "tgt": row["tgt"]})
    return paths_real_sample

def effective_path(path_str: str) -> str:
    """
    Creates an effective path, removing backtracking.
    Should be in format link1;link2;link3;...
    """
    path = path_str.split(";")
    effective_path = path.copy()
    # if there's an element '<' remove it and the previous element
    i=0
    while i < len(effective_path):
        if effective_path[i] == '<':
            effective_path.pop(i)
            effective_path.pop(i-1)
            i-=1
        else:
            i += 1
    return ";".join(effective_path)

def load_tsv_to_dict_list(file: str) -> list[dict]:
    """
    Loads a .tsv file into a list of dictionaries.
    """
    df = pd.read_csv(file, sep='\t')
    return df.to_dict('records')

def get_api_key(fname='secrets.json', provider='openai', key='dlab_key'):
    try:
        with open(fname) as f:
            keys = json.load(f)[provider]
            if key is not None:
                api_key = keys[key]
            else:
                api_key = list(keys.values())[0]
    except Exception as e:
        print(f'error: unable to load {provider} api key {key} from file {fname} - {e}')
        return None

    return api_key

def printv(msg, v=0, v_min=0, c=None, debug=False):
    # convenience print function
    if debug:
        c = 'yellow' if c is None else c
        v, v_min = 1, 0
        printc('\n\n>>>>>>>>>>>>>>>>>>>>>>START DEBUG\n\n', c='yellow')
    if (v > v_min) or debug:
        if c is not None:
            printc(msg, c=c)
        else:
            print(msg)
    if debug:
        printc('\n\nEND DEBUG<<<<<<<<<<<<<<<<<<<<<<<<\n\n', c='yellow')


def printc(x, c='r'):
    m1 = {'r': 'red', 'g': 'green', 'y': 'yellow', 'w': 'white',
          'b': 'blue', 'p': 'pink', 't': 'teal', 'gr': 'gray'}
    m2 = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'pink': '\033[95m',
        'teal': '\033[96m',
        'white': '\033[97m',
        'gray': '\033[90m'
    }
    reset_color = '\033[0m'
    print(f'{m2.get(m1.get(c, c), c)}{x}{reset_color}')

def load_links(path="wikispeedia/wikispeedia_paths-and-graph/links.tsv") -> dict:
    """
    Loads the links file into a dictionary.
    """
    flinks = open(path)
    links = {}
    for i, line in enumerate(flinks):
        line = line.strip()
        if '#' in line or line == '':
            continue
        line = line.split('\t')
        src = unquote(line[0]); tgt = unquote(line[1])
        if src not in links:
            links[src] = [tgt]
        else:
            links[src].append(tgt)
    return links

def wikify_article_name(name):
    return "_".join(name.strip().split(" "))
def un_wikify_article_name(name):
    return " ".join(name.strip().split("_"))

def get_super(x):
    """
    Returns the given script in superscript
    Notes
        - From a very brief search it appears Q doesn't have a superscript in unicode.
    """
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

def get_sub(x):
    """
    Returns the given script in subscript
    """
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)

def count_initial_tabs(sentence):
    """
    Counts amount of tabs at the beginning of a sentence
    """
    count = 0
    for char in sentence:
        if char == '\t':
            count += 1
        else:
            break 
    return count

def fix_page_encoding(page):
    return page.replace("%25", "%")

def href2page(href):
    """
    Returns the page name from a href.
    Examples
        - "A/B.htm" -> "B", where A can be anything
        - "../data/wpcd/wp/u/United_states.htm" -> "United_states"
    Exceptions
        - "#subsection" -> ""
    """
    pattern = r'[^/]+/([^/]+)\.htm'
    page = re.search(pattern, href)

    if page:
        word = page.group(1)
    elif href[0]=="#":
        word=""
    else:
        print("No such page found found : {} from {}".format(page, href))
    return fix_page_encoding(word)

def page2href(pages_path, page):
    """
    Given the folder where the pages are stored and the page name, returns the href.
    """
    initial = page[0].lower()
    href = pages_path + "{}/{}.htm".format(initial, page)
    return href

def get_links_old(input_text):
    """
    TO-DELETE : Just keeping here for a day or two in case we need it.
    Returns all link in a string. The links have to be formatted as [*link*]
    Inputs
        - input_text [str]: text to extract links from
    Outputs
        - matches [list(str)]: list of links
    """
    pattern = r'\[\*(.*?)\*\]'
    matches = re.findall(pattern, input_text)
    return matches

def get_links(html):
    """
    Given an html string, it returns all the links in it.
    """
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all('a')
    link_list = []
    pattern = r'\.\.\/\.\.\/wp\/./([^/]+)\.htm'
    for link in links:
        href = link.get('href')
        if href and re.match(pattern, href):
            link_list.append(re.findall(pattern, href)[0])
    return link_list

def format_path(path):
    """
    Formats a path from a string to a list
    Inputs
        - path [str]: path in string format
    Outputs
        - path [list(str)]: path in list format
    Example
        - "A;B;C" -> ["A", "B", "C"]
    """
    if isinstance(path, str):   
        path = path.split(";")
    return path

def extract_json_from_response(response):
    pattern = r"""\{(?:[^{}]|(?R))*\}"""
    matches = regex.finditer(pattern, response, regex.VERBOSE)

    jsons = [match.group() for matchNum, match in enumerate(matches, start=1)]
    if len(jsons) > 0:
        return jsons[0]
    else:
        return None

def get_field_from_json(json_str, field='Path'):
    if len(json_str)==0:
        return ""
    try:
        json_obj = json.loads(json_str)
        path = json_obj[field]
        return path
    except:
        return ""   
    
def waitForDownload(file_path, verbose=False):
    """
    Given a file path, this function waits until the file is downloaded
    """
    if verbose:
        print("Waiting for '{}' to be downloaded ...".format(file_path))
    while not os.path.exists(file_path):
        sleep(0.2)
    if verbose:
        print("File '{}' downloaded !".format(file_path))
        
def delete_file_if_exists(file_path):
    """
    Given a file path, this function deletes the file if it exists
    """
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")
    else:
        print(f"File '{file_path} ' does not exist.")

def getActiveParagraph(text, skipTop, skipBot):
    """
    Given a text coming from a paragraph, it returns the active text
    Inputs
        - text [str]: text from a paragraph
        - skipTop [float]: proportion of text to skip from the top
        - skipBot [float]: proportion of text to skip from the bottom
    Outputs
        - activeText [str]: active text
    """
    words = text.split(" ")
    n = len(words)
    activeText = " ".join(words[ceil(skipTop*n):len(words)-(floor(skipBot*n))])
    return activeText

def getActiveText(text, tag, skipTop, skipBot, active):
    """
    Given a text and the active proportions it returns the active text
    Inputs
        - text [str]: text
        - tag [str]: tag of the text (eg. "P", "H1", ...)
        - skipTop [float]: proportion of text to skip from the top
        - skipBot [float]: proportion of text to skip from the bottom
        - active [float]: active proportion of text to keep
    Outputs
        - activeText [str]: active text
    """
    if tag=="P":
        activeText = getActiveParagraph(text, skipTop, skipBot)
    elif active>=0.5:
        activeText = text
    else:
        activeText = ""
    return activeText

def formatPageURL(page):
    """
    Given just the name of the file it formats it into the URL so it can be used by the selenium driver.
    --> This might need to be updated as it covers all pages except for the ones in the "index" folder.
    """
    source = os.path.join(ROOT_DIR, "data/wpcd/wp/{}/{}.htm".format(page.lower()[0], page))
    return source

def retextMarkdownLinks(text) : 
    """
    Simplifies markdown links.
    eg. 
    [ European Union ](../../wp/e/European_Union.htm "European Union") -> [European Union)(European_Union)
    """

    # [ * ] -> [*]
    pattern = r'\[\s*(.*?)\s*\]'
    text = re.sub(pattern, r'[\1]', text)

    # (../../wp/p/Page.htm *) -> (../../wp/p/Page.htm)
    pattern = r'(\.htm)[^\)]*'
    text = re.sub(pattern, r'\1', text)

    # (../../wp/p/Page.htm) -> (Page)
    pattern = r'../../wp/./(.*?).htm'
    text = re.sub(pattern, r'\1', text)
    return text

def filter_links_from_text(text, valid_links):
    # Regular expression pattern to find all [content](link) occurrences
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'

    # Function to decide whether to keep the word or just the content
    def replace_func(match):
        content, word = match.groups()
        # Check if the content is empty or the word is not in the list
        if not content or word not in valid_links:
            return content
        else:
            return f"[{content}]({word})"

    # Use re.sub() to apply the replace_func to all matches
    reformatted_text = re.sub(pattern, replace_func, text)
    return reformatted_text

def clean_nav_text(text:str):
    #remove empty lines
    text = re.sub(r'\n\s*\n', '\n', text)

    # turn 2 or more spaces into tabs
    text = re.sub(r' {2,}', '\t', text)

    return text

def emptyFolder(path):
    """
    Given a directory path, it empties it. 
    """
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def getWikiNavFolder():
    """
    Returns the absolute path to the WikiNav folder.
    """
    current_dir = os.getcwd()
    root = current_dir.split("/WikiNav/")[0]
    wikinavFolder = os.path.join(root, "WikiNav")
    return wikinavFolder

def getHeadings(page_source):
    """
    Returns in a list the text of all headings in an html page.
    """
    soup = BeautifulSoup(page_source, 'html.parser')
    headings = soup.find_all(re.compile('^h[1-6]$'))
    headings = [h.text.strip() for h in headings]
    return headings

def getTextHTML(html_source):
    """
    Given the source code of an html document it returns just the text encoded in it.
    """
    soup = BeautifulSoup(html_source, 'html.parser')
    text = soup.text.strip()
    return text

def remove_items_from_list(l, items):
    """
    Removes items from a list.
    """
    l = [item for item in l if item not in items]
    return l