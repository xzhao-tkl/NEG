import os
import json
import pickle

def load_pickle(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Pickle file - {filename} is not found")
    
    with open(filename, "rb") as fp:
        return pickle.load(fp)

def dump_pickle(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def load_json(filename, create_if_nonexist=False):
    if not os.path.exists(filename):
        if create_if_nonexist:
            dump_json({}, filename)
            return {}
        else:
            raise FileNotFoundError(f"{filename} is not found")
    
    with open(filename, 'r') as fp:
        return json.load(fp)

def dump_json(obj, filename, pretty=False):
    with open(filename, 'w') as fp:
        if pretty:
            json.dump(obj, fp, indent=2)
        else:
            json.dump(obj, fp)