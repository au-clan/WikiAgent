import re

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
    href = "data/wpcd/wp/{}/{}.htm".format(initial, page)
    return href

def get_links(input_text):
    pattern = r'\[\*(.*?)\*\]'
    matches = re.findall(pattern, input_text)
    return matches

def format_path(path):
    if isinstance(path, str):   
        path = path.split(";")
    return path
    

