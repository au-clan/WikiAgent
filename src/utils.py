import json

def get_api_key(fname='secrets.json', provider='openai', key='dlab_key'):
    try:
        with open(fname) as f:
            keys = json.load(f)[provider]
            if key is not None:
                api_key = keys[key]
            else:
                api_key = list(keys.values())[0]
    except Exception as e:
        print(f'error: unable to load {provider} api key {key} from file {fname} - {e}')
        return None

    return api_key
