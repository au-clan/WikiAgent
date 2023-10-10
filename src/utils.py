import json, re
from urllib.parse import unquote
import pandas as pd
import regex

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

def load_links(file="data/wikispeedia/wikispeedia_paths-and-graph/links.tsv") -> dict:
    """
    Loads the links file into a dictionary.
    """
    flinks = open(file)
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
    page = href[11:-4]
    return fix_page_encoding(page)

def page2href(page):
    initial = page[0].lower()
    href = "../data/wpcd/wp/{}/{}.htm".format(initial, page)
    return href

def get_links(input_text):
    pattern = r'\[\*(.*?)\*\]'
    matches = re.findall(pattern, input_text)
    return matches

def format_path(path):
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