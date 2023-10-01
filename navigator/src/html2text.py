#from bs4 import BeautifulSoup
from bs4 import Comment
import re

from src.utils import get_super, get_sub, href2page

def clean_row(row):
    """
    Removes all new lines in row[:-1]
    """
    clean_row = [i for i in row.contents[:-1] if i != "\n"]
    clean_row.append(row.contents[-1])
    return clean_row

def clean_div(div):
    """
    Removes all new lines in row[:-1]
    """
    clean_row = [i for i in div.contents if i != "\n"]
    if div.attrs.get("class") and ("NavFrame" in div.attrs.get("class")):
        clean_row.append(div.contents[-1])
    return clean_row

def clean_string(string):
    clean_text = re.sub("del_lnk", "", string)
    return clean_text

def clean_p(p):
    pattern = r'(?<!\n)\n(?!\\n)'

    clean_text = re.sub(pattern, '\n\t', p)
    return clean_text

def clean_italic(italic):
    clean_text = italic.text
    return clean_text

def clean_tt(tt):
    clean_text = tt.text
    return clean_text

def clean_em(em):
    clean_text = em.text
    return clean_text

def clean_underlined(underlined):
    clean_text = underlined.text
    return clean_text

def clean_super(super):
    clean_text = get_super(super.text)
    return clean_text

def clean_sub(sub):
    clean_text = get_sub(sub.text)
    return clean_text

def clean_link(link):
    if "href" in link.attrs:
        clean_text = link.text + "[*" + href2page(link.attrs["href"]) + "*]"
    else:
        clean_text=""
    return clean_text

def clean_script(script_text):
    return ""

def clean_heading(heading):
    clean_text = heading.text
    return clean_text

def clean_break(break_):
    clean_text = "\n" 
    return clean_text

def post_process_text(text):
    out = re.sub(" +", " ", text)
    out = re.sub("\t\n", "\n", out)
    out = re.sub("\n{3,}", "\n\n", out)
    out = re.sub("\t+", "\t", out)
    out = re.sub(" \t", " ", out)
    out = re.sub("\t ", "\t", out)
    return out

def clean_theader(theader):
    clean_text = theader.text
    return clean_text

def clean_strong(strong):
    clean_text = strong.text
    return clean_text

def clean_code(code):
    clean_text = code.text
    return clean_text

def clean_strike(strike):
    clean_text = strike.text
    return clean_text

def clean_var(var):
    clean_text = var.text
    return clean_text

def clean_caption(caption):
    clean_text = caption.text
    return clean_text

def clean_cite(cite):
    clean_text = cite.text
    return clean_text

def clean_img(img):
    clean_text = ""
    return clean_text

def config_list(current_list_ordered, current_list_number):
    if current_list_ordered:
        return "{}. ".format(current_list_number)
    else:
        return "*"


def clean(part, i, document, removed_parts = ["script", "map"], current_list_ordered = False, current_list_number=0):
    """
    i should be removed, used here to indicate part
    """
    clean_text = ""

    
    for element in [e for e in part if not isinstance(e, Comment)]:
        tag = element.name
        if tag==None:
            clean_text += clean_string(element)
        # Starting a list
        elif tag == "ul":
            clean_text += clean(element.contents, i, document)
        elif tag == "ol":
            clean_text += clean(element.contents, i, document, current_list_ordered = True)
        # Adding a new item to a list
        elif tag == "li":
            clean_text += "\n\t " + config_list(current_list_ordered, current_list_number) + clean(element.contents, i, document,current_list_ordered=current_list_ordered, current_list_number = current_list_number+1)
        # Creates a table
        elif tag == "table":
            clean_text += clean(clean_row(element), i, document)
        # Creates a table row
        elif tag == "tr":
            clean_text += clean(clean_row(element), i, document)
        # Sets the table row's header
        elif tag == "th":
            clean_text += "\t" + clean(element.contents, i, document)
        # Sets the table row's details
        elif tag == "td":
            clean_text += "\t" + clean(element.contents, i, document)
        #Definition list
        elif tag in ["dl", "dt"]:
            clean_text += "\n\t" + clean(element.contents,i, document)
        elif tag in ["dd"]:
            clean_text += "\n\t\t" + clean(element.contents,i, document)
        elif tag == "p":
            clean_text += "\n\t" + clean(element.contents, i, document)
        elif tag == "pre":
            clean_text += "\t" + clean(element.contents, i, document)
        elif tag == "div":
            clean_text += clean(clean_div(element), i, document)
        elif tag == "blockquote":
            clean_text += clean(element.contents, i, document)
        elif tag == "center":
            clean_text += clean(element.contents, i, document).replace("\n", "\n\t")
        elif tag == "b":
            clean_text += clean(element.contents, i, document)
        elif tag == "i":
            clean_text += clean(element.contents, i, document)
        elif tag == "big":
            clean_text += clean(element.contents, i, document)
        elif tag == "tt":
            clean_text += clean_tt(element)
        elif tag == "s":
            clean_text += clean_strike(element)
        elif tag == "em":
            clean_text += clean_em(element)
        elif tag == "u":
            clean_text += clean_underlined(element)
        elif tag == "font":
            clean_text += clean(element.contents, i, document)
        elif tag == "small":
            clean_text += clean(element.contents, i, document)
        elif tag == "strong":
            clean_text += clean_strong(element)
        elif tag == "caption":
            clean_text += clean_caption(element)
        elif tag == "cite":
            clean_text += clean_cite(element)
        elif tag == "code":
            clean_text += clean_code(element)
        elif tag == "var":
            clean_text += clean_var(element)
        elif tag == "sup":
            clean_text += clean_super(element)
        elif tag == "sub":
            clean_text += clean_sub(element)
        elif tag == "span":
            clean_text += clean(element.contents, i, document)
        elif tag == "a":
            clean_text += " " + clean_link(element)
        elif tag == "br":
            clean_text += clean_break(element) 
        elif tag == "hr":
            clean_text += "_______________________________________________\n"
        elif tag == "img":
            clean_text += clean_img(element)
        elif tag in removed_parts:
            clean_text += ""
        # The next case only happens for 1 article so I kinda "shortcuted" it
        elif tag in ["head", "html", "body"]:
            clean_text += "\n\n" + clean(element.contents, i, document)
        elif tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            clean_text += "\n"+clean_heading(element)
        else :
            print("Problem : Document {} in part {}".format(document, i))
            print("ELEMENT : {}".format(element))
            print("TAG : {}".format(tag))
            print("TYPE : {}".format(type(element)))
            print("-----")
    preocessed_text = post_process_text(clean_text)
    return preocessed_text

def create_body_parts(body):
    """
    To delete (probably)
    """
    parts = []
    temp = body.contents
    while True:
        element = temp[:-1]
        parts.append(element)
        next_element = temp[-1]
        try:
            temp = next_element.contents
        except AttributeError:
            break
    return parts