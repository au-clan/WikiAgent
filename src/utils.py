import json
from urllib.parse import unquote
import pandas as pd

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