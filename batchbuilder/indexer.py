def id_provider(ii) -> str:
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    alpha = ii%26
    numeric = ii // 26
    return f"{alphabet[alpha]}_{numeric}"

class Indexer:
    def __init__(self):
        self._index = -1

    def index(self):
        self._index += 1
        return self._index
