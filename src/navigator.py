import tiktoken, os, html2text
from bs4 import BeautifulSoup
import pandas as pd
import urllib.parse

from selenium.webdriver.common.by import By

# Custom 
from html2text_ import clean, clean_text
from utils import get_links, waitForDownload, delete_file_if_exists, getActiveText
from utils import page2href, get_links, formatPageURL, retextMarkdownLinks


class NavigatorHTML:
    """
    First implementation of the HTML Navigator.
    Moves from page to page by just transforming html to text.
    """
    def __init__(self, pages_path, encoding_model="cl100k_base", token_window=300, overlap=0.1):
        self.pages_path = pages_path
        self.encoder = tiktoken.get_encoding(encoding_model)
        self.current_page = None
        self.text = None # Whole document text
        self.encodings = None # Whole document encodings
        self.window = {"start":0, "end":token_window} # Token indices for current window
        self.token_window = token_window # Size of the token window
        self.overlap = overlap # Size of the overlap when you move the window
        self.window_text = None # Text for the current window
        self.window_encodings = None # Encodings for the current window
        self.window_links = None # Links existing in the current window
        
    def get_position(self):
        """
        Returns the relative position for the middle of the window in the document (in %).
        """
        position = self.window["end"]- self.token_window/2
        relative_position = round(position / len(self.encodings) * 100, 2)
        return relative_position

    def set_curr_doc(self, current_page):
        self.current_page = current_page
        with open(page2href(self.pages_path, current_page), "r", encoding='unicode_escape') as f:
                doc = BeautifulSoup(f, "html.parser")
        body = doc.find("div", {"id" : "bodyContent"})
        if body == None:
           print("Can't decode SOS Children’s village documents")
        else:
            self.text = clean(body,0, current_page)
            self.encodings = self.encoder.encode(self.text)
            self.window = {"start":0, "end":self.token_window} 
            self.update_window()

    def update_window(self):
        """
        Updae the window text, encodings and links based on the current window.
        """
        self.window_encodings = self.encodings[self.window["start"] : self.window["end"]]
        self.window_text  = self.encoder.decode(self.window_encodings)
        self.window_links = get_links(self.window_text)
    
    def move_down(self):
        """
        Moves the window down.
        """
        self.window["start"] += self.token_window - int(self.token_window * self.overlap)
        self.window["end"] += self.token_window - int(self.token_window * self.overlap)
        if self.window["end"] >= len(self.encodings):
            self.window["start"] = len(self.encodings)-self.token_window
            self.window["end"] = len(self.encodings)
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
                
                
class WebNavigatorHTML:
    """
    Second implementation of the HTML Navigator.
    Opens the page in a browser and moves by interacting with the page.
    """
    def __init__(self, driver, pages, downloads_folder, agents=None, markdown=False):
        self.driver = driver
        self.agentTabs = {}
        if agents is None:
            agents = list(range(len(pages)))
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
            self.driver.get(formatPageURL(page))
        
        self.driver.switch_to.window(original_tab)
        self.driver.close()
        self.driver.switch_to.window(self.agentTabs[0])
    
    def switchTab(self, agent):
        """
        Switches to the tab corresponding to the agent.
        """
        assert agent in self.agentTabs.keys(), "Agent {} not found !".format(agent)
        self.driver.switch_to.window(self.agentTabs[agent])

    
    def movePage(self, page):
        """
        Given the name of a page, it firstly translates it to the corresponding link and then
        clicks on it, if it's in the current page.
        """
        link = "../../wp/" + page[0].lower() +"/" +page + ".htm"
        try :
            element = self.driver.find_element(By.CSS_SELECTOR, "a[href='{}']".format(link))
            element.click()
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
        df["activeText"] = df.apply(lambda x : getActiveText(x["text"], x["tag"], x["skipTop"], x
        ["skipBot"], x["active"]), axis=1)
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
        