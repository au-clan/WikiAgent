import attr
from attr import define, field
import sys
sys.path.append("src/")

@define
class Agent:
    """
    Virtual agent class. Defines the main interface for the agent.
    """
    prompt_template = field(factory=dict)

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
    
    