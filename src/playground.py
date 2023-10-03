# File to play around with the code

from environments import *
from pprint import pprint

def main():
    env = LinkWikiEnviroment("Andy Warhol")
    pprint(env.to_dict())
    env.change_page("Bob Dylan")
    pprint(env.to_dict())

if __name__ == "__main__":
    main()