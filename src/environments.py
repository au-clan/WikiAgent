import attr
from attr import define, field
import sys
from typing import List
sys.path.append("src/")
from utils import load_links, wikify_article_name, clean_nav_text, printc
from navigator import WebNavigatorHTML
from urllib.parse import quote, unquote
import re



# every time the current page is updated, we should update the environment

@define
class Environment:
    """
    Virtual environment class. Defines the main interface for the environment.
    """
    current_page = field(factory=str)
    content = field(default=None)
    page_history = field(factory=list) #this is basically the effective history
    full_page_history = field(factory=list) #this includes backtracks
    content_history = field(factory=list)
    links = field(factory=list)

    # may or may not be used by subclasses
    links_file = field(default="wikispeedia/wikispeedia_paths-and-graph/links.tsv")
    
    def to_dict(self):
        # Create a dictionary manually
        attributes = {}
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                attributes[attr] = getattr(self, attr)
        return attributes
    
    def _set_links(self):
        """
        This depends on the type of environment
        """
        raise NotImplementedError("Subclasses must implement the set_links method.")
    
    def _set_content(self):
        """
        This depends on the type of environment
        """
        raise NotImplementedError("Subclasses must implement the set_content method.")
        

        
@define
class WikiEnvironment(Environment):
    """
    Still a virtual class with more specific implementations.
    """

    def __attrs_post_init__(self):
        self.current_page = wikify_article_name(self.current_page)
        self.page_history.append(self.current_page)
        self.full_page_history.append(self.current_page)
        self._set_content()
        self.content_history.append(self.content)
        self._set_links()

    def _set_current_page(self, new_page: str):
        self.current_page = wikify_article_name(new_page)

    def change_page(self, new_page: str, links_seen_on_current_page: List[str] = None, **kwargs):
        wikified_new_page = wikify_article_name(new_page)
        assert len(wikified_new_page) > 0, "Page name can't be empty. Choose a different action."
        if not links_seen_on_current_page:
            assert wikified_new_page in self.links, f"Link {wikified_new_page} not on current page. Try a different page."
        else:
            assert wikified_new_page in links_seen_on_current_page, f"Link {wikified_new_page} not on current page. Try a different page."
            
        self.page_history.append(wikified_new_page)
        self.full_page_history.append(wikified_new_page)
        self.content_history.append(self.content)
        self._set_current_page(wikified_new_page)
        self._set_content()
        self._set_links()
    
    def remove_pages_from_links(self, pages: List[str]):
        """
        Removes pages from the candidate links.
        """
        self.links = [link for link in self.links if link not in pages]
    
    def back(self):
        """
        Return to previous page.
        """
        if len(self.page_history) > 1:
            self.full_page_history.append("<")
            self.page_history.pop()
            self.content_history.pop()
            self._set_current_page(self.page_history[-1])
            self._set_content()
            self._set_links()
        else:
            raise Exception("Cannot go back any further. Choose a different action.")

    
@define
class LinkWikiEnviroment(WikiEnvironment):
    """
    Each page is only represented by the links it has.
    """
    def _set_links(self):
        """
        This depends on the type of environment
        """
        all_links = load_links(self.links_file)
        assert self.current_page in all_links, f"Invalid page {self.current_page}."
        self.links = all_links[self.current_page]

    def _set_content(self):
        pass

@define
class WebWikiEnvironment(WikiEnvironment):

    nav = field(factory=WebNavigatorHTML)
    position = field(factory=tuple)
    html = field(default=False)

    def __attrs_post_init__(self):
        self._set_position()
        super().__attrs_post_init__()

    def _set_current_page(self, new_page: str):
        new_page_wikified= wikify_article_name(new_page)
        self.nav.movePage(quote(quote(new_page_wikified)))
        self.current_page = new_page_wikified
    
    def _set_position(self):
        self.position = self.nav.getScrollPosition()

    def _set_links(self):
        self.links = [unquote(unquote(link)) for link in self.nav.getLinks()]
    
    def _set_content(self):
        if self.html:
            ### remove html comments
            pattern = r"<!--.*?-->\n"
            ### remove blocks of 3 or more spaces
            pattern += r"| {3,}"
            self.content = re.sub(pattern, "", self.nav.getHTML())
            # self.content = self.nav.getHTML()
        else:
            ### dirty fix for invalid links
            self.content = clean_nav_text(self.nav.getText())
            # pattern = r"\[\*(index|subject\.[^\]]*)\*\]"
            # self.content = unquote(self.nav.getText())
            # self.content = re.sub(pattern, "", self.content)

    def move(self, direction: str):
        """
        Moves the window down or up.
        """
        if direction.strip().lower() == "down":
            self.nav.moveDown()
        elif direction.strip().lower() == "up":
            self.nav.moveUp()
        else:
            raise Exception("Invalid direction. Choose either 'up' or 'down'.")
        self._set_content()
        self._set_links()
        self._set_position()
        self.content_history.append(self.content)

    def back(self):
        """
        Return to previous page.
        """
        if len(self.page_history) > 1:
            self.full_page_history.append("<")
            self.page_history.pop()
            self.content_history.pop()
            self.nav.moveBack()
            self.current_page = self.page_history[-1]
            self._set_content()
            self._set_links()
            self._set_position()
        else:
            raise Exception("Cannot go back any further. Choose a different action.")
    
    def change_env(self, agent_id):
        """
        Switch to the environment (tab) of the specified agent.
        """
        self.nav.switchTab(agent_id)
        self._set_content()
        self._set_links()
        self._set_position()
        # self.current_page : Not sure yet how to handle this.