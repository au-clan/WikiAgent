# -*- coding: utf-8 -*-

import attr
from attr import define, field
import sys
from typing import List
sys.path.append("src/")
from utils import load_links, wikify_article_name, clean_nav_text, printc, filter_links_from_text, remove_items_from_list
from navigator import WebNavigatorHTML, NavigatorHTML
from urllib.parse import quote, unquote
import re



# every time the current page is updated, we should update the environment

@define
class Environment:
    """
    Virtual environment class. Defines the main interface for the environment.
    """
    current_page: str = field(default=None)
    content = field(default=None)
    page_history = field(factory=list) #this is basically the effective history
    full_page_history = field(factory=list) #this includes backtracks
    content_history = field(factory=list)
    links: List[str] = field(default=None)

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

    def get_current_page(self):
        return self.current_page
    
    def get_clickable_links(self):
        return self.links

    def get_links(self):
        """Returns all links in current article"""
        return self.links
    
    def _set_current_page(self, new_page: str):
        self.current_page = wikify_article_name(new_page)
        self._set_content()
        self._set_links()

    def change_page(self, new_page: str, links_seen_on_current_page: List[str] = None, **kwargs):
        wikified_new_page = wikify_article_name(new_page)
        assert len(wikified_new_page) > 0, "Error: Page name can't be empty. Choose a different action."
        if not links_seen_on_current_page:
            assert wikified_new_page in self.links, f"Error: Link {wikified_new_page} not on current page. Try a different page."
        else:
            assert wikified_new_page in links_seen_on_current_page, f"Error: Link {wikified_new_page} not on current page. Try a different page."
            
        self.page_history.append(wikified_new_page)
        self.full_page_history.append(wikified_new_page)
        self.content_history.append(self.content)
        self._set_current_page(wikified_new_page)
    
    def back(self):
        """
        Return to previous page.
        """
        if len(self.page_history) > 1:
            self.full_page_history.append("<")
            self.page_history.pop()
            self.content_history.pop()
            self._set_current_page(self.page_history[-1])
        else:
            raise Exception("Error: Cannot go back any further. Choose a different action.")

    
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
        assert self.current_page in all_links, f"Error: Invalid page {self.current_page}."
        self.links = all_links[self.current_page]

    def _set_content(self):
        pass

@define 
class TextWikiEnvironment(WikiEnvironment):
    """
    This one is a bit different from the other environments, it mostly uses the navigator to get stuff, and keeps histories.
    """
    use_dfs = field(default=True)
    nav = field(factory=NavigatorHTML)
    window_position_page: dict = field(factory=dict) # {page: last_window}
    links_seen_on_page: dict = field(default={}) # {page: [links]}
    visited_headings_on_page: dict = field(default={}) # {page: [headings]}
    visited_links: List[str] = field(factory=list) # links that have been visited

    def __attrs_post_init__(self):
        self.page_history.append(self.nav.current_page)
        self.full_page_history.append(self.nav.current_page)
        self.content_history.append(self.nav.window_text)
        self.window_position_page[self.nav.current_page] = self.nav.window
        self.links_seen_on_page[self.nav.current_page] = self.nav.window_links
        self.visited_headings_on_page[self.nav.current_page] = []
        self.visited_links.append(self.nav.current_page)

    def get_scroll_position(self):
        """ Its a tuple of (current_scroll, total_scrolls)"""
        return self.nav.get_scroll_position()
    
    def update_history_dict(self, dict_to_update: dict, key: str, value):
        if key not in dict_to_update:
            if isinstance(value, list):
                dict_to_update[key] = value
            else:
                dict_to_update[key] = [value]
        else:
            # if value is a list, extend the list
            if isinstance(value, list):
                dict_to_update[key].extend(value)
            else:
                dict_to_update[key].append(value)
        dict_to_update[key] = list(set(dict_to_update[key]))

    def get_current_page(self):
        return self.nav.current_page
    
    def get_links(self):
        """Returns all links in current article"""
        return self.nav.links
    
    def get_clickable_links(self):
        if self.use_dfs:
            return remove_items_from_list(self.links_seen_on_page[self.nav.current_page], self.visited_links)
        else:
            return self.nav.window_links
    
    def get_content(self):
        if self.use_dfs:
            return filter_links_from_text(self.nav.window_text, valid_links=self.links_seen_on_page[self.nav.current_page])
        else:
            return self.nav.window_text

    def get_headings(self):
        if self.use_dfs:
            headings = remove_items_from_list(self.nav.get_headings(), self.visited_headings_on_page[self.nav.current_page])
            return headings
        else:
            return self.nav.window_headings

    def move(self, direction: str):
        """
        Move up or down on the page, or to a heading.
        """
        assert direction in ["up", "down"] or direction in self.get_headings(), "Invalid direction. Choose either 'up' or 'down' or specific heading."
        
        if direction == "down":
            self.nav.move_down()
        elif direction == "up":
            self.nav.move_up()
        else: # move to heading
            self.nav.move_heading(direction)
            # update visited headings
            self.update_history_dict(self.visited_headings_on_page, self.nav.current_page, direction)
        
        # update links seen
        self.update_history_dict(self.links_seen_on_page, self.nav.current_page, self.nav.window_links)
        # update window position
        self.window_position_page[self.nav.current_page] = self.nav.window
        # update content history
        self.content_history.append(self.get_content())

    def change_page(self, new_page: str):
        """
        Change the current page.
        """
        new_page_wikified = wikify_article_name(new_page)
        assert len(new_page_wikified) > 0, "Page name can't be empty. Choose a different action."
        assert new_page_wikified in self.links_seen_on_page[self.nav.current_page], f"Link {new_page_wikified} not on current page. Try a different page."
        
        # save window position
        self.window_position_page[self.nav.current_page] = self.nav.window

        self.nav.set_curr_doc(new_page_wikified)

        # update window position
        self.window_position_page[self.nav.current_page] = self.nav.window
        # update links seen
        self.update_history_dict(self.links_seen_on_page, self.nav.current_page, self.nav.window_links)
        # update page history
        self.page_history.append(new_page_wikified)
        self.full_page_history.append(new_page_wikified)
        self.content_history.append(self.get_content())
        # update visited headings
        self.visited_headings_on_page[self.nav.current_page] = [] if self.nav.current_page not in self.visited_headings_on_page else self.visited_headings_on_page[self.nav.current_page]
        # update visited links
        self.visited_links.append(new_page_wikified)
    
    def back(self):
        """
        Return to previous page.
        """
        if len(self.page_history) > 1:
            self.full_page_history.append("<")
            self.page_history.pop()
            self.nav.set_curr_doc(self.page_history[-1], self.window_position_page[self.page_history[-1]]['start'])
        else:
            raise Exception("Error: Cannot go back any further. Choose a different action.")


@define
class WebWikiEnvironment(WikiEnvironment):

    nav = field(factory=WebNavigatorHTML)
    position = field(factory=tuple)
    html = field(default=False)
    headings = field(factory=list)

    def __attrs_post_init__(self):
        self._set_position()
        self._set_headings()
        super().__attrs_post_init__()

    def _set_current_page(self, new_page: str):
        new_page_wikified= wikify_article_name(new_page)
        self.nav.movePage(quote(quote(new_page_wikified)))
        self.current_page = new_page_wikified
        self._set_position()
        self._set_content()
        self._set_links()
        self._set_headings()
    
    def _set_position(self):
        self.position = self.nav.getScrollPosition()

    def _set_headings(self):
        self.headings = self.nav.getHeadings()

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
            self.content = clean_nav_text(self.nav.getText())

    def move(self, direction: str):
        """
        Moves the window down or up.
        """
        if direction.strip().lower() == "down":
            self.nav.moveDown()
        elif direction.strip().lower() == "up":
            self.nav.moveUp()
        elif direction in self.headings:
            self.nav.moveHeading(direction)
        else:
            raise Exception("Invalid direction. Choose either 'up' or 'down' or specific heading.")
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
            #self.content_history.pop()
            self.nav.moveBack()
            self._set_current_page(self.page_history[-1])
        else:
            raise Exception("Error: Cannot go back any further. Choose a different action.")
    
    def change_env(self, agent_id):
        """
        Switch to the environment (tab) of the specified agent.
        """
        self.nav.switchTab(agent_id)
        self._set_content()
        self._set_links()
        self._set_position()
        # self.current_page : Not sure yet how to handle this.