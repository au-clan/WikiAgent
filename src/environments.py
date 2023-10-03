import attr
from attr import define, field
import sys
sys.path.append("src/")
from utils import load_links, wikify_article_name


# every time the current page is updated, we should update the environment

@define
class Environment:
    """
    Virtual environment class. Defines the main interface for the environment.
    """
    current_page = field(factory=str)
    content = field(default=None)
    page_history = field(factory=list)
    content_history = field(factory=list)
    links = field(factory=list)
    
    
    def __str__(self):
        return self.content
    
    def to_dict(self):
        # Create a dictionary manually
        attributes = {}
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                attributes[attr] = getattr(self, attr)
        return attributes
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    
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
        self._set_content()
        self.content_history.append(self.content)
        self._set_links()

    def _set_current_page(self, new_page: str):
        self.current_page = wikify_article_name(new_page)

    def change_page(self, new_page: str, **kwargs):
        wikified_new_page = wikify_article_name(new_page)

        assert wikified_new_page in self.links, f"Invalid click {wikified_new_page}."

        self.page_history.append(new_page)
        self.content_history.append(self.content)
        self._set_current_page(wikified_new_page)
        self._set_content()
        self._set_links()
    
@define
class LinkWikiEnviroment(WikiEnvironment):
    """
    Each page is only represented by the links it has.
    """
    links_file = field(default="data/wikispeedia/wikispeedia_paths-and-graph/links.tsv")

    def _set_links(self):
        """
        This depends on the type of environment
        """
        all_links = load_links(self.links_file)
        assert self.current_page in all_links, f"Invalid page {self.current_page}."
        self.links = all_links[self.current_page]

    def _set_content(self):
        pass

