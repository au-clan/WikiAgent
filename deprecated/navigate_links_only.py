import sys 
sys.path.append("src/")
from WikiAgent.deprecated.prompt_utils import wikify_article_name
import pandas as pd
from pprint import pprint
from utils import load_links
from typing import List

        
def check_path(src, tgt, generated_path, links_file="data/wikispeedia/wikispeedia_paths-and-graph/links.tsv", check_clicks=True):
    """
    Checks if the generated path is valid.
    Path is valid if:
    - The source is the same as the source given
    - Each click is valid
    - The target is the same as the target given
    
    Path format: link1;link2;link3;...
    """
    links = load_links(links_file)
    gen_path_eff = effective_path(generated_path)
    gen_path_list = []
    for node in gen_path_eff.split(";"):
        gen_path_list.append(wikify_article_name(node))
    if gen_path_list[0] != src:
        return gen_path_list, 0, f"Generated source {gen_path_list[0]} not in the list of Wikispeedia articles"
    if check_clicks:
        for i in range(len(gen_path_list)-1):
            cur = gen_path_list[i]; nxt = gen_path_list[i+1]
            if nxt not in links[cur]:
                return gen_path_list, 0, f"Generated click from {cur} to {nxt} is incorrect. Correct path prefix {';'.join(gen_path_list[:i+1])}"
    if gen_path_list[-1] == tgt:
        return gen_path_list, 1, "Sucess"
    else:
        return gen_path_list, 0, "Valid path, but couldn't reach the target"

def get_path_demonstrations(path_demonstrations_file='data/wikispeedia/wikispeedia_paths-and-graph/path_demonstrations.tsv') -> List[dict]:
    """
    Gets the paths from the path_demonstrations.tsv file. Only keeps the effective path.
    """
    path_demonstrations_df = pd.read_csv(path_demonstrations_file, sep='\t')
    path_demonstrations = []
    for i, row in path_demonstrations_df.iterrows():
        path_demonstrations.append({"src": row["src"], "tgt": row["tgt"], "output": effective_path(row["output"])})
    return path_demonstrations


if __name__ == "__main__":
    path_demonstrations = get_path_demonstrations()
    print(path_demonstrations[-1]["output"])
    print(effective_path(path_demonstrations[-1]["output"]))
    