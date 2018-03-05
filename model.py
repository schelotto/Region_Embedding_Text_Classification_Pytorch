import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from overrides import overrides
from torch.nn.init import xavier_uniform

class ContextWordEmb(nn.Module):
    def __init__(self, args) -> None:
        super(ContextWordEmb, self).__init__()
        self.gpu = args.cuda
        self._vocab_size = args.vocab_size
        self._embed_size = args.embed_size
        self._context = args.context
        self._attn = args.attn
        self._n_class = args.n_class
        self._w_pad = args.vocab.stoi['<pad>']
        self._c_pad = self._w_pad

        # Model
        self.embedding = nn.ModuleList(
            [nn.Embedding(self._vocab_size, self._embed_size, padding_idx=self._c_pad) for _ in range(2 * self._context + 1)]
        )
        for i, emb_layer in enumerate(self.embedding):
            self.add_module('context_embed_%d' % i, emb_layer)

        # Attention
        self.attn_linear = nn.Linear(self._embed_size, self._embed_size)
        self.context = nn.Parameter(self._embed_size, self._attn)
        xavier_uniform(self.context)

        self.classifier = nn.Sequential(
            nn.SELU(),
            nn.AlphaDropout(0.5),
            nn.Linear(self._embed_size * self._attn, self._n_class)
        )

    @overrides
    def forward(self, x: torch.LongTensor):
        [batch_size, sent_len] = x.size()
        padding_ = Variable(torch.LongTensor([self._c_pad] * self._context))
        x = torch.cat((padding_, x, padding_), dim=1)
        if self.gpu:
            x = x.cuda()

        embedding = torch.stack([
            embed(x[:, i:(i + sent_len)]) for i, embed in enumerate(self.embedding)], dim=3)
        multiple = []

        for i in range(2 * self._context + 1):
            multiple.append(embedding[:, :, :, self._context] * embedding[:, :, :, i])

        multiple = torch.sum(multiple, dim=3)

        context = [self.context.view(1, self._embed_size, self._attn) for _ in range(batch_size)]
        context = torch.cat(context, 0).contiguous()

        multi_rep = F.tanh(self.attn_linear(multiple))

        alpha = torch.bmm(multi_rep, context) # [batch, len, embed] x [batch, embed, attn]
        alpha = torch.softmax(alpha, 1) # [batch, len, attn]
        alpha = torch.transpose(alpha, 1, 2) # [batch, attn, len]

        multiple = torch.bmm(alpha, multiple).view(batch_size, -1) # [batch, attn x embed]
        return self.classifier(multiple)