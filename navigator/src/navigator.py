import tiktoken
from bs4 import BeautifulSoup

# Custom 
from src.html2text import clean
from src.utils import page2href, get_links


class NavigatorHTML:
    def __init__(self, encoding_model="cl100k_base", token_window=300, overlap=0.1):
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
        

    def set_curr_doc(self, current_page):
        self.current_page = current_page
        with open(page2href(current_page), "r", encoding='unicode_escape') as f:
                doc = BeautifulSoup(f, "html.parser")
        body = doc.find("div", {"id" : "bodyContent"})
        if body == None:
           print("Can't decode SOS Children’s village documents")
        else:
            self.text = clean(body,0, current_page)
            # Saving in a constant place for debugging purposes
            with open("output.text", "w") as file:
                file.write(self.text)
            self.encodings = self.encoder.encode(self.text)
            self.window = {"start":0, "end":self.token_window} 
            self.update_window()
    
    def move_down(self):
        self.window["start"] += self.token_window - int(self.token_window * self.overlap)
        self.window["end"] += self.token_window - int(self.token_window * self.overlap)
        if self.window["end"] >= len(self.encodings):
            self.window["start"] = len(self.encodings)-self.token_window
            self.window["end"] = len(self.encodings)
        self.update_window()
    
    def move_up(self):
        self.window["start"] -= self.token_window - int(self.token_window * self.overlap)
        self.window["end"] -= self.token_window - int(self.token_window * self.overlap)
        if self.window["start"] < 0:
            self.window["start"] = 0
            self.window["end"] = self.token_window
        self.update_window()
    
    def update_window(self):
        self.window_encodings = self.encodings[self.window["start"] : self.window["end"]]
        self.window_text  = self.encoder.decode(self.window_encodings)
        self.window_links = get_links(self.window_text)

    def check_target(self, target_page, verbose=False):
        if target_page in self.window_links:
            if verbose:
                print("Moving from {} to {}".format(self.current_page, target_page))
            self.set_curr_doc(current_page=target_page)
            return True
        else:
            return False