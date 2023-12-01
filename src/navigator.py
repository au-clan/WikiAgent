import os, html2text
from bs4 import BeautifulSoup
import pandas as pd
import urllib.parse
import re
from utils import printc
from transformers import AutoTokenizer
from selenium.webdriver.common.by import By
from urllib.parse import unquote,quote
from typing import List, Dict, Tuple

# Custom 
from html2text_ import clean, clean_text
from utils import get_links, waitForDownload, delete_file_if_exists, getActiveText
from utils import page2href, get_links, formatPageURL, retextMarkdownLinks
from utils import getWikiNavFolder, emptyFolder, getHeadings, getTextHTML
from formatHTML import formatHTML

class NavigatorHTML:
    """
    First implementation of the HTML Navigator.
    Moves from page to page by just transforming html to text.
    """
    def __init__(self, pages_path, model_name="meta-llama/Llama-2-7b-chat-hf", token_window=300, overlap=0):
        self.pages_path = pages_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.current_page = None
        self.text = None # Whole document text
        self.links = None # Links existing in the whole document
        self.tokens = None # Whole document encodings
        self.window = {"start":0, "end":token_window} # Token indices for current window
        self.token_window = token_window # Size of the token window
        self.overlap = overlap # Size of the overlap when you move the window
        self.window_text = None # Text for the current window
        self.window_encodings = None # Encodings for the current window
        self.window_links = None # Links existing in the current window
        self.headings = None # Headings in the current page

        self.html2text_handler = html2text.HTML2Text() # Handler for html2text
        self.html2text_handler.body_width = 0 # Set body width to 0 to avoid line breaks


    def get_scroll_position(self)->tuple:
        """
        Returns the relative position for the middle of the window in the document (in %).
        """
        # total number of scrolls 
        total_scrolls = len(self.tokens)//self.token_window
        current_scroll = self.window["end"]//self.token_window
        return current_scroll, total_scrolls

    def extract_headings(self, doc):
        """
        Returns all headings of the current page.
        Each heading is a dict {"text": stripped_heading_text, "raw_text": raw_text, "tag": heading_tag}.
        """
        return [{"text": heading.text.strip(), "tag": heading.name} for heading in doc.find_all(["h2", "h3", "h4", "h5", "h6"])]
    
    def get_headings(self):
        return [heading["text"] for heading in self.headings]
    
    def set_curr_doc(self, current_page, window_start=0):
        self.current_page = current_page
        with open(page2href(self.pages_path, current_page), "r", encoding='utf-8') as f:
                doc = BeautifulSoup(f, "html.parser")

        self.headings = self.extract_headings(doc)

        body = doc.find("div", {"id" : "bodyContent"})
        if body == None:
           print("No body found !")
        else:
            self.text = self.html2text_handler.handle(str(body))
            self.reformat_links_in_text()
            self.clean_text()
            self.links = self.extract_links_from_text(self.text)
            self.tokens = self.get_tokens()
            print([(token[0], self.tokenizer.decode(token[0], skip_special_tokens=True)) for token in self.tokens])
            self.window = {"start":window_start, "end":window_start+self.token_window} 
            self.update_window()

    def reformat_links_in_text(self):
        """
        - Turns links from [ text ]( url.link.htm "name" ) to [text](link)
        - Removes all non Wikipedia links (mostly links to images) from the current window text.
        """
        # remove image links that are markdown formatted, for example: ![text](../../images/50/5099.jpg.htm "Enlarge")
        image_link_pattern = r"!\[[^\]]*\]\([^)]*\)"
        self.text = re.sub(image_link_pattern, "", self.text)        

        # turn links from [ text ]( url.link.htm "name" ) to <a href="link">text</a>
        #example: [ fossil record ](../../wp/f/Fossil_record.htm "Fossil record") -> <a href="Fossil_record">fossil record</a>
        pattern = r"\[\s*([^]\s]+(?:\s+[^]\s]+)*)\s*\]\([^)]*/([^)]+)\.htm[^)]*\)"

        def replace_with_html(match):
            text = match.group(1)
            url = unquote(unquote(match.group(2)))
            return f"<a href=\"{url}\">{text}</a>"

        self.text = re.sub(pattern, replace_with_html, self.text)

        # remove empty links
        empty_link_pattern = r"\[\s*\]\([^)]*\)"
        self.text = re.sub(empty_link_pattern, "", self.text)


    def clean_text(self):
        # turn 2 newlines or more into 1 newline
        self.text = re.sub(r"\n{2,}", "\n", self.text)

        # turn 2 spaces or more into \t
        self.text = re.sub(r" {2,}", "\t", self.text)

    def get_tokens(self) -> List[tuple]:
        """
        Breaks the text into a list of strings, where each string stops at a link. This is so that links aren't split when scrolling.
        Links are formatted like this: <a href="link">text</a>
        Output: [(token, type), (token, type), ...]
        """
        # Regex pattern to match <a href="link">text</a>
        pattern = r'<a href="[^"]+">[^<]+</a>'


        # Find all matches and save to a list. We want a list of all <a href="link">text</a> in the text.
        matches = re.findall(pattern, self.text)
        matches = [{"text":match.strip(), "type":"link"} for match in matches]
        print("matches: ", matches)


        # split the text into chunks, where each chunk stops at a link
        chunks = re.split(pattern, self.text)
        chunks = [{"text":chunk.strip(), "type":"text"} for chunk in chunks]

        # interleave the chunks with the links
        chunks = [chunk for pair in zip(chunks, matches) for chunk in pair]

        # remove empty chunks
        chunks = [chunk for chunk in chunks if chunk['text'] != ""]

        # tokenize the text chunks
        for chunk in chunks:
            chunk["tokens"] = self.tokenizer.encode(chunk["text"], add_special_tokens=False)

        # now turn this into a list of tuples, where each tuple is one single token and its type
        chunks = [(token, chunk["type"]) for chunk in chunks for token in chunk["tokens"]]
        return chunks


    def extract_links_from_text(self, text: str) -> List[str]:
        """
        Extracts the links from the current window text. Links are formatted like this: [text](link).
        """
        # links are formatted like this: <a href="link">text</a>
        link_pattern = r"<a href=\"([^\"]+)\">[^<]+</a>"
        links = re.findall(link_pattern, text)

        # remove bad links
        links = [link for link in links if ".jpg" not in link and ".png" not in link and ".gif" not in link]
        return links

    def update_window(self):
        """
        Updae the window text, encodings and links based on the current window.
        """
        # check if the last token is a link. If it is, extend the window to include the whole link
        if self.tokens[self.window["end"]-1][1] == "link":
            # find the index of the next text token
            next_text_token = next((i for i, token in enumerate(self.tokens[self.window["end"]:]) if token[1] == "text"), None)
            if next_text_token != None:
                self.window["end"] += next_text_token

        # get the encodings for the current window
        self.window_encodings = [token[0] for token in self.tokens[self.window["start"]:self.window["end"]]]
        self.window_text  = self.tokenizer.decode(self.window_encodings, skip_special_tokens=True)
        self.window_links = self.extract_links_from_text(self.window_text)
    
    def move_down(self):
        """
        Moves the window down.
        """
        self.window["start"] = self.window["end"] - int(self.token_window * self.overlap)
        self.window["end"] += self.token_window - int(self.token_window * self.overlap)
        if self.window["end"] >= len(self.tokens):
            self.window["start"] = len(self.tokens)-self.token_window
            self.window["end"] = len(self.tokens)
        self.update_window()
    
    def move_up(self):
        """
        Moves the window up.
        """
        self.window["start"] -= self.token_window - int(self.token_window * self.overlap)
        self.window["end"] -= self.token_window - int(self.token_window * self.overlap)
        if self.window["start"] < 0:
            self.window["start"] = 0
            self.window["end"] = self.token_window
        self.update_window()
    
    def check_target(self, target_page, verbose=False):
        """
        If the target page is in the current window, open page and return True.
        """
        if target_page in self.window_links:
            if verbose:
                print("Moving from {} to {}".format(self.current_page, target_page))
            self.set_curr_doc(current_page=target_page)
            return True
        else:
            return False
    
    def move_heading(self, heading_text: str):
        """
        Moves the window to the heading if it exists.
        """
        # heading = (heading_text, heading_tag), heading_tag = h2, h3, h4, h5, h6
        # h2 starts with ##\t, h3 with ###\t, etc.
        markdown_prefix = {
        "h2": "##\t",
        "h3": "###\t",
        "h4": "####\t",
        "h5": "#####\t",
        "h6": "######\t"
        }
        # find heading with the stripped text == heading_text
        heading = next((heading for heading in self.headings if heading["text"] == heading_text), None)
        heading_string = "\n"+markdown_prefix[heading['tag']] + heading['text']

        #find the sequence of tokens in self.tokens that corresponds to heading_string
        heading_encodings = self.tokenizer.encode(heading_string, add_special_tokens=False)[1:]

        # Search for the first occurrence of the heading_encodings in self.tokens
        for i in range(len(self.tokens) - len(heading_encodings) + 1):
            # if self.tokens[i:i+len(heading_encodings)] == heading_encodings:
            #     # Assuming moving to the heading involves setting an index
            #     self.window["start"] = i  # Or any other relevant action
            #     self.window["end"] = i + self.token_window
            # update this to work with the new tokens format
            if [token[0] for token in self.tokens[i:i+len(heading_encodings)]] == heading_encodings:
                self.window["start"] = i
                self.window["end"] = i + self.token_window
                self.update_window()
                return True

        # Update the window
        self.update_window()
                
class WebNavigatorHTML:
    """
    Second implementation of the HTML Navigator.
    Opens the page in a browser and moves by interacting with the page.
    """
    def __init__(self, driver, pages, downloads_folder, agents=None, markdown=True, emptyActivePages=True, onFly = False):
        self.driver = driver
        self.onFly = onFly
        self.agentTabs = {}
        if agents is None:
            agents = list(range(len(pages)))
        
        # In case html pages should be formatted on the fly
        if self.onFly:
            # Folder where active pages are stored
            self.activePagesPath = getWikiNavFolder() + "/data/wpcd/wp/activePages/" 
            # If the folder doesn't exist already create it
            if not os.path.exists(self.activePagesPath):
                os.makedirs(self.activePagesPath)
            # If asked empty active pages folder from previous navigators
            if emptyActivePages:
                emptyFolder(self.activePagesPath) 

        self.initTabs(agents, pages)
        
        file_name = "active_elements.json" # Name of the file downloaded - Defined already by JS
        self.file_path = os.path.join(downloads_folder, file_name)
        self.df = None
        self.active_updated= True
        if markdown:
            self.parser = html2text.HTML2Text()
            self.parser.ignore_links = False
            self.parser.images_to_alt = True
            self.parser.body_width = 0
        else:
            self.parser = None

    def initTabs(self, agents, pages):
        original_tab = self.driver.current_window_handle
        for agent, page in zip(agents, pages):
            self.driver.switch_to.new_window("tab")
            self.agentTabs[agent] = self.driver.current_window_handle
            if self.onFly:
                originalPagePath = formatPageURL(page)
                formattedPagePath = self.activePagesPath + page + ".htm"
                formatHTML(originalPagePath, formattedPagePath)
                self.driver.get("file://" + formattedPagePath)
            else:
                self.driver.get("file://"+formatPageURL(page))
        
        self.driver.switch_to.window(original_tab)
        self.driver.close()
        self.driver.switch_to.window(list(self.agentTabs.values())[0])
    
    def switchTab(self, agent):
        """
        Switches to the tab corresponding to the agent.
        """
        assert agent in self.agentTabs.keys(), "Agent {} not found !".format(agent)
        self.driver.switch_to.window(self.agentTabs[agent])
        self.active_updated=True

    
    def movePage(self, page):
        """
        Given the name of a page, it firstly translates it to the corresponding link and then
        clicks on it, if it's in the current page.
        """
        link = "../../wp/" + page[0].lower() +"/" +page + ".htm"
        try :
            element = self.driver.find_element(By.CSS_SELECTOR, "a[href='{}']".format(link))
            element.click()
            if self.onFly:
                pageSource = self.driver.page_source
                with open(self.activePagesPath+page+".htm", 'w') as file:
                    file.write(pageSource)
                formatHTML(self.activePagesPath+page+".htm", self.activePagesPath+page+".htm")
                self.driver.get("file://" + self.activePagesPath+page+".htm")
            self.active_updated= True
        except Exception as e:
            print(e)
            print("Link for '{}' [{}] not found in the page !".format(page, link))

    def moveUp(self):
        """
        Move the active window up
        """
        up_button = self.driver.find_element(By.ID, "goUp")
        up_button.click()
        self.active_updated = True

    def moveDown(self):
        """
        Move the active window down.
        """
        down_button = self.driver.find_element(By.ID, "goDown")
        down_button.click()
        self.active_updated = True

    def moveBack(self):
        """
        Move back once in the browser history.
        """
        self.driver.back()        
        self.active_updated = True

    def moveForward(self):
        """
        Move forward once in the browser history.
        """
        self.driver.forward()
        self.active_updated = True

    def downloadActive(self):
        """
        Clicks on the "get" button to download the active elements.
        """
        get_button = self.driver.find_element(By.ID, "get")
        get_button.click()

    def quit(self):
        """
        Quits the navigator.
        """
        self.driver.quit()
        
    ###---Main---###
    def updateActive(self, verbose = False):
        """
        Main function to update the active elements.
            1. Download active elements (json file).
            2. Load elements to df.
            3. Get text from elements.
            4. Compute active text from each element.
            5. Compute active links from each active element.
        """
        # If a file exists it has to be deleted
        delete_file_if_exists(file_path = self.file_path)
        # Download active elements
        self.downloadActive()
        # Wait for elements to be downloaded
        waitForDownload(self.file_path)
        # Load elemnts to df
        df = pd.read_json(self.file_path)
        # Active area
        df["active"] = df.apply(lambda x: 1-x["skipTop"]-x["skipBot"], axis=1)
        # html -> text
        if self.parser:
            df["text"] = df["elements"].apply(lambda x : self.parser.handle(x))
            df["text"] = df["text"].apply(retextMarkdownLinks)
        else:
            df["text"] = df["elements"].apply(clean_text)
        df["activeText"] = df.apply(lambda x : getActiveText(x["text"], x["tag"], x["skipTop"], x["skipBot"], x["active"]), axis=1)
        df["activeLinks"] = df["elements"].apply(get_links)
        self.df = df
        self.active_updated = False
        if verbose:
            self.getText()
    
    def getText(self):
        """
        Get the text from the active elements.
        """
        if self.active_updated:
            self.updateActive()
        active_text = ("").join(self.df["activeText"])
        return active_text
    
    def getHTML(self):
        """
        Get the html from the active elements.
        """
        if self.active_updated:
            self.updateActive()
        html = ("\n").join(self.df["elements"])
        return html
    
    def getLinks(self):
        """
        Get the links from the active elements.
        """
        if self.active_updated:
            self.updateActive()
        links = self.df["activeLinks"].explode().dropna().tolist()
        return links
    
    def getScrollPosition(self):
        """
        Get the scroll position for the active elements.
        """
        if self.active_updated:
            self.updateActive()
        currScroll = self.df.iloc[0]["currScroll"]
        maxScroll = self.df.iloc[0]["maxScroll"]
        return currScroll, maxScroll
    
    def getHeadings(self):
        """
        Returns all headings of the current page.
        """
        page_source = self.driver.page_source
        headings = getHeadings(page_source)
        return headings
    
    def moveHeading(self, heading):
        # In case nav just initialised or moved to a new page
        if self.active_updated:
            self.updateActive()
        # Go to the top of the page and once there update active content.
        maxScroll = self.df.iloc[0]["maxScroll"]
        for i in range(maxScroll):
            self.moveUp()
        self.updateActive()

        scrolls = 0
        headingNotFound = True
        while headingNotFound and scrolls<=maxScroll+1:
            test_df = self.df[self.df.tag.str.startswith("H")].copy()
            test_df["headings"] = test_df["elements"].apply(getTextHTML)

            corr_element = test_df[test_df.headings == heading]
            if len(corr_element)>0:
                index = corr_element.index[0]
                self.df = self.df[self.df.index>=index]
                headingNotFound=False
            else:
                self.moveDown()
                self.updateActive()
                scrolls +=1
        return not headingNotFound
        