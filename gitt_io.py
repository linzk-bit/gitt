from json import dumps, load


def load_json(file: str, encoding='utf-8') -> dict:
    with open(file, "r", encoding=encoding) as f:
        return load(f)
    
def save_json(file: str, dict: dict, encoding='utf-8') -> None:
    json_data = dumps(dict, indent=4)
    with open(file, "w", encoding=encoding) as f:
        f.write(json_data)

