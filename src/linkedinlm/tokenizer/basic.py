from collections.abc import Sequence
class BasicTokenizer:


    def __init__(self):
        pass




    def encode(self, text:str) -> list[int]:
        """
        encode converts text into UTF-8 byte IDs for now
        """
        a = text.encode("utf-8")
        return list(a)

    def decode(self, token_ids: Sequence[int]) -> str:
        """
        decode converts UTF-8 byte IDs back into text for now
        """
        a = bytes(token_ids)
        return a.decode("utf-8")