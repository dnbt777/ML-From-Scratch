


class Embeddings():
    def __init__(self, tokens, dimensions):
        self.tokens = tokens
        self.embeddings = self.create_embeddings(tokens, dimensions=dimensions)
    

    def create_starting_embeddings(tokens, dimensions=100):