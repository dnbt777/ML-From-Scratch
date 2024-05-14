import numpy as np
from scipy.special import softmax # yes I could implement this myself but itd prob be inefficient

class Transformer():
    def __init__(self):
        # parameters
        self.tokens = "01"
        self.tokens = dict(zip(self.tokens, range(len(self.tokens))))
        self.vocab_size = 2
        self.model_size = 512 # dims for each encoding, d_input
        self.query_size = 128


        self.MLP = MLP(input_layer_size=self.model_size)
        self.Wq = np.random.rand(self.model_size, self.query_size)
        self.Wk = np.random.rand(self.model_size, self.query_size)
        self.Wv = np.random.rand(self.model_size, self.model_size) # for k and q of embeddings
        self.base_embeddings = np.random.rand(self.vocab_size, self.model_size)


    def get_base_embeddings_from(self, input_tokens):
        base_embeddings = np.array([self.base_embeddings[self.tokens[token]] for token in input_tokens])
        return base_embeddings



    def run(self, tokens):
        # Encoding layer
        tokens = "asodjsaojdas"
        starting_embeddings = self.base_embeddings


    def self_attention(self, embeddings):
    # method: attention(self, embeddings)
#         get embeddings for tokens (input)
#         positional encode the embeddings
        pass
#         do attention table
        Q = np.matmul(self.Wq, embeddings)
        K = np.matmul(self.Wk, embeddings)
        attention_pattern = np.dot(Q, np.transpose(K)) / np.sqrt(self.model_size)
        attention_pattern = softmax(attention_pattern)

        # attention
        attention = np.dot(attention_pattern, self.Wv)

        # residual
        updated_embeddings = attention + embeddings

        # normalize
        return np.linalg.norm(updated_embeddings)


    def feed_forward(self, embeddings):
        pass
    #     method: feed_forward(self, embeddings)
#         feed embeddings through MLP
#         residue stuff and normalize

    def encode_layer(self, embeddings):
        pass
#         embeddings = get_base_embeddings_from(tokens)
#         for sublayer in range(sublayers):
#             embeddings = attention(embeddings)
#             embeddings = feed_forward(embeddings)
#         return embeddings




class MLP():
    def __init__(self, input_layer_size):
        self.biases_settings = [input_layer_size, input_layer_size, input_layer_size] # size of each embedding - first layer is input layer - set up layers here
        self.biases = np.array([np.random.rand(input_layer_size) for input_layer_size in self.biases_settings]) # make layers

        self.weights_settings = [n*m for n, m in zip(self.layer_settings[1:], self.layer_settings)] # set up connections num of connections between each one
        self.weights = np.array([np.random.rand(connection_count) for connection_count in self.connections_settings]) # make connections, initialize as random



    def feed_forward(self, input_layer):
        ReLU = np.vectorize(lambda x: x if x > 0 else 0)
        # h = RELU(w matmul input_layer)
        neurons = input_layer
        for w in self.weights:
            h = ReLU(np.transpose(w), neurons)
            neurons = h


        





# transformer class for the whole thing
#     hold all weights and biases: mlp, Wq, Wk, V, MLP network, original embeddings
#     (intiialize as random)
#     Wu: unembedding matrix
#         same size as encoding matrix - this is a function (matrix) that maps a final vector to a probability distribution of possible tokens (hence why it has one row for each token)

#     # Encoder stuff
#     method: get original embeddings(tokens):
#         self explanatory, return embeddings

#     method: attention(self, embeddings)
#         get embeddings for tokens
#         positional encode the embeddings
#         do attention table
#         multiply w V
#         softmax 
#         residue
#         normalize 
#         return output
    
#     method: feed_forward(self, embeddings)
#         feed embeddings through MLP
#         residue stuff and normalize
    
#     method: encoder_layer(self, tokens, sublayers=6)
#         embeddings = get_original_embeddings(tokens)
#         for sublayer in range(sublayers):
#             embeddings = attention(embeddings)
#             embeddings = feed_forward(embeddings)
#         return embeddings


#     # Decoder stuff
#     method: masked attention
#         regular attention, BUT for each token, turn the keys for the tokens that follow it into -inf before softmax so they become 0.
    
#     # do masked attention

#     # do attention

#     # do feed forward


#     # finally unencode
#     add positional encoding to the output embeddings
    
#     unembedding matrix x embeddings[-1] => probability distribution for tokens