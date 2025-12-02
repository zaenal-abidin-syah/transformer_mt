import torch
from torch import nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
  def __init__(self, src_vocab, tgt_vocab, d_model=512, nhead=8, N=6, d_ff=2048, dropout=.0):
    super(Transformer, self).__init__()
    self.d_model = d_model
    self.input_embedding = nn.Embedding(src_vocab, d_model)
    self.output_embedding = nn.Embedding(tgt_vocab, d_model)
    self.pos_encod = PositionalEncoding(d_model=d_model, dropout=dropout, batch_first=True)
    self.transformer = nn.Transformer(
      d_model=d_model, 
      nhead=nhead, 
      num_encoder_layers=N, 
      num_decoder_layers=N, 
      dim_feedforward=d_ff,
      dropout=dropout,
      batch_first=True
      )
    self.projection = nn.Linear(d_model, tgt_vocab)
    self.init_weights()
  def init_weights(self):
    initrange = 0.1
    nn.init.uniform_(self.input_embedding.weight, -initrange, initrange)
    nn.init.uniform_(self.output_embedding.weight, -initrange, initrange)
    
    # Init projection
    nn.init.zeros_(self.projection.bias)
    nn.init.uniform_(self.projection.weight, -initrange, initrange)

    # Optional: inisialisasi seluruh layer transformer
    for p in self.transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
  def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
    src_encoding = self.pos_encod(self.input_embedding(src) * math.sqrt(self.d_model))
    tgt_encoding = self.pos_encod(self.output_embedding(tgt) * math.sqrt(self.d_model))
    x = self.transformer(
        src=src_encoding,
        tgt=tgt_encoding,
        src_mask=src_mask,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_key_padding_mask,
        tgt_key_padding_mask=tgt_key_padding_mask
    )
    # self.output_embedding(tgt)
    return self.projection(x)



class PositionalEncoding(nn.Module):

  def __init__(self, d_model, dropout=0.1, batch_first=True, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.batch_first = batch_first
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if batch_first:
      pe = pe.unsqueeze(0)
    else:
      pe = pe.unsqueeze(1)
    # pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    if self.batch_first:
      x = x + self.pe[:,:x.size(1), :] # type: ignore
    else:
      x = x + self.pe[:x.size(0), :, :] # type: ignore
    return self.dropout(x)

# class TransformerModel(nn.Transformer):
#   """Container module with an encoder, a recurrent or transformer module, and a decoder."""

#   def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
#     super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers)
#     self.model_type = 'Transformer'
#     self.src_mask = None
#     self.pos_encoder = PositionalEncoding(ninp, dropout)

#     self.input_emb = nn.Embedding(ntoken, ninp)
#     self.ninp = ninp
#     self.decoder = nn.Linear(ninp, ntoken)

#     self.init_weights()

#   def _generate_square_subsequent_mask(self, sz):
#     return torch.log(torch.tril(torch.ones(sz,sz)))

#   def init_weights(self):
#     initrange = 0.1
#     nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
#     nn.init.zeros_(self.decoder.bias)
#     nn.init.uniform_(self.decoder.weight, -initrange, initrange)
#   def forward(self, src, has_mask=True):
#     if has_mask:
#       device = src.device
#       if self.src_mask is None or self.src_mask.size(0) != len(src):
#         mask = self._generate_square_subsequent_mask(len(src)).to(device)
#         self.src_mask = mask
#     else:
#       self.src_mask = None

#     src = self.input_emb(src) * math.sqrt(self.ninp)
#     src = self.pos_encoder(src)
#     output = self.encoder(src, mask=self.src_mask)
#     output = self.decoder(output)
#     return F.log_softmax(output, dim=-1)