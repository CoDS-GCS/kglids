class Label:

    def __init__(self, text: str, lan: str):
        self.text = text
        self.lan = lan

    def get_text(self) -> str:
        return self.text

    def get_lan(self) -> str:
        return self.lan

    def __repr__(self):
        return "\"" + self.text + "\"" + "@" + self.lan

    def __str__(self):
        return str(self.__repr__())
