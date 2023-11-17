import  subprocess, sys
from tqdm import tqdm
from bs4 import BeautifulSoup, Tag, NavigableString, Comment


def clone(el):
    """
    Clones an htlm element.
    """
    if isinstance(el, NavigableString):
        return type(el)(el)

    copy = Tag(None, el.builder, el.name, el.namespace, el.nsprefix)
    copy.attrs = dict(el.attrs)
    for attr in ('can_be_empty_element', 'hidden'):
        setattr(copy, attr, getattr(el, attr))
    for child in el.contents:
        copy.append(clone(child))
    return copy

def docCaptionsComments(pathIn, pathOut):
    """
    Removes comments and changes class of captions so they appear in Readability.
    """
    # Load file
    with open(pathIn, "r", encoding='utf-8', errors='ignore') as f:
        doc = BeautifulSoup(f, "html.parser")

    # Remove comments
    comments = doc.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()
    
    # Change class of  captions so they appear in Readability
    thumbcaptions = doc.find_all("div", {"class": "thumbcaption"})
    for thumbcaption in thumbcaptions:
        thumbcaption["class"] = ["internal"]
    
    # Save file
    with open(pathOut, 'w', encoding='utf-8') as file:
        file.write(doc.prettify())

def decompose_nfs(pathIn, pathOut):
    """
    Decomposes navigation frames so they appear in Readability.
    """
    with open(pathIn, "r", encoding='utf-8', errors='ignore') as f:
        doc = BeautifulSoup(f, "html.parser")

    nfs = doc.findAll("div", {"class": "NavFrame"})
    while len(nfs) > 0:
        contents = [nf.find("div", {"class": "NavContent"}) for nf in nfs]
        contents = [c.contents for c in contents if c is not None]
        navheads = [nf.find("div", {"class": "NavHead"}) for nf in nfs]
        navheads = [h.contents for h in navheads if h is not None]
        if len(contents) == 0:
            print(pathIn)
            break
        for head, content, nf in zip(navheads, contents, nfs):
            for i in content:
                j = clone(i)
                nf.insert_after(j)
            h = clone(head[0])
            nf.insert_after(h)
        for nf in nfs:
            nf.decompose()
        nfs = doc.findAll("div", {"class": "NavFrame"})
    with open(pathOut, 'w', encoding='utf-8') as file:
            file.write(doc.prettify())
            
def del_siteSub(pathIn, pathOut):
    """
    Removes siteSub element.
    """
    with open(pathIn, "r", encoding='utf-8', errors='ignore') as f:
        doc = BeautifulSoup(f, "html.parser")
    element_to_delete = doc.find('h3', id='siteSub')
    if element_to_delete:
        element_to_delete.extract()
    with open(pathOut, 'w', encoding='utf-8') as file:
            file.write(doc.prettify())

def unwrapTable(pathIn, pathOut):
    """
    Unwraps tables in case they are in a div element. Needs to be done so we can
    get the elements row by row.
    """
    with open(pathIn, "r", encoding='utf-8', errors='ignore') as f:
        doc = BeautifulSoup(f, "html.parser")
    bodyContent = doc.find("div", {"id":"bodyContent"})
    if bodyContent is None:
        print(pathIn)
        return
    divs = bodyContent.find_all("div")
    for div in divs:
        for child in div.children:
            if child.name == "table":
                div.insert_after(child)
    with open(pathOut, 'w', encoding='utf-8') as file:
            file.write(doc.prettify())

def removeEnlarge(pathIn, pathOut):
    """
    Removes Enlarge images.
    """
    # Load file
    with open(pathIn, "r", encoding='utf-8', errors='ignore') as f:
        doc = BeautifulSoup(f, "html.parser")

    img_elements_to_remove = doc.find_all('img', alt="Enlarge")
    for img_element in img_elements_to_remove:
        img_element.extract()
    
    # Save file
    with open(pathOut, 'w', encoding='utf-8') as file:
        file.write(doc.prettify())

def formatHTML(pathIn, pathOut):
    """
    Given a path to a HTML file, it reads it modifies it and saves it in pathOut.
    Modifications:
        1. Removes comments and changes class of captions so they appear in Readability.
        2. Decomposes navigation frames so they appear in Readability.
        3. Adds buttons and correct structure to the HTML file.
        4. Removes siteSub element.
        5. Unwraps tables in case they are in a div element.
        6. Removes Enlarge images.

    """

    #Thumbcaptions
    docCaptionsComments(pathIn, pathOut)
    
    #Navigation Frames
    decompose_nfs(pathOut, pathOut)

    #Readability
    command = ['node', '../create_new_htm.js', '--sourceFile', pathOut, '--destinationFolder', pathOut]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)        
    
    # Define the HTML content to add at the beginning and end
    header = """<html>
    <head></head>
    <body>
        <button id="goDown" onclick="goDown()">DOWN</button>
        <button id="goUp" onclick="goUp()">UP</button>
        <button id="get" onclick="downloadActiveElements()">GET</button>
        <button id="test" onclick="test()">TEST</button>
    """
    footer = """
    <script type="text/javascript" src = "../../js/buttons.js"  ></script>
    </body>
    </html>"""

    # Add buttons
    with open(pathOut, 'r') as file:
        original_content = file.read()
    modified_content = header + original_content + footer
    with open(pathOut, 'w') as file:
        file.write(modified_content)

    # Remove siteSub
    del_siteSub(pathOut, pathOut)

    # Unwrap tables
    unwrapTable(pathOut, pathOut)

    # Remove Enlarge
    removeEnlarge(pathOut, pathOut)