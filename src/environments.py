import attr
from attr import define, field

# every time the current page is updated, we should update the environment

@define
class Environment:
    current_page = field(factory=str)
    content = field()
    page_history = field(factory=list)
    content_history = field(factory=list)

    def __str__(self):
        return self.content
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def update(self, **kwargs):
        raise NotImplementedError("Subclasses must implement the update method.")

        


@define
class LinkOnlyEnviroment(Environment):
    """
    Here each page is only represented by the links it has.
    """
    def update(self, **kwargs):
        pass

