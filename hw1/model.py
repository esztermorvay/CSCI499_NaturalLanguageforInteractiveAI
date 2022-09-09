# IMPLEMENT YOUR MODEL CLASS HERE
import torch


class TargetActionIdet(torch.nn.Module):
    def __init__(
            self,
            device,
            vocab_size,
            input_len,
            n_actions,
            n_targets,
            embedding_dim
    ):
        super(TargetActionIdet, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.input_len = input_len
        self.n_actions = n_actions
        self.n_targets = n_targets


        # embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # maxpool layer
        self.maxpool = torch.nn.MaxPool2d((input_len, 1), ceil_mode=True)

        # linear layer
        self.fc = torch.nn.Linear(embedding_dim, 2)

        self.lstm = torch.nn.LSTM(embedding_dim,2)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        embeds = self.embedding(x)
        maxpooled_embeds = self.maxpool(embeds)
        lstm, (n, m )= self.lstm(maxpooled_embeds.squeeze(1))
        out = self.fc(lstm.squeeze(1)).squeeze(1)  # squeeze out the singleton length dimension that we maxpool'd over
        # actions = []
        # targets = []
        # for a,t in out:
        #     actions.append(a.item())
        #     targets.append(t.item())
        # return torch.tensor(actions), torch.tensor(targets)
        return out[:, 0], out[:, 1]
