import torch
"""
creating the cbow model
"""
class CBOW(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size, context_size):
        super(CBOW, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.context_size = context_size
        # embedding layer
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        # linear layer
        self.linear_layer = torch.nn.Linear(self.embedding_dim, self.vocab_size)

    # inputs should be a list of context
    def forward(self, inputs):
        # print("inputs ")
        # print(inputs)
        embeddings = self.embedding_layer(inputs)
        # get the sum of the context words embeddings
        sum_layer = sum(embeddings)
        # use linear layer to get the target
        target = self.linear_layer(sum_layer)
        return target
        # outputs = []
        # for i in inputs:
        #     embeddings = self.embedding_layer(i)
        #     # get the sum of the context words embeddings
        #     sum_layer = sum(embeddings)
        #     # use linear layer to get the target
        #     target = self.linear_layer(sum_layer)
        #     outputs.append(target)
        # return outputs
