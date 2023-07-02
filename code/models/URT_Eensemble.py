import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np



# Assuming you have 10 pre-trained AutoFormers in a list called "autoformers"
# and 10 separate models in a list called "models"


def feed_into_models(inputs, autoformers, models):
    outputs = []
    for autoformer in autoformers:
        outputs.append(autoformer(inputs))
    final_outputs = []
    for output, model in zip(outputs, models):
        final_outputs.append(model(output))
    return final_outputs



##################################################################### URT



# TODO: integrate the two functions into the following codes


def get_dotproduct_score(proto, cache, model):
  proto_emb   = model['linear_q'](proto)
  s_cache_emb = model['linear_k'](cache)
  raw_score   = F.cosine_similarity(proto_emb.unsqueeze(1), s_cache_emb.unsqueeze(0), dim=-1)
  return raw_score  


def get_mlp_score(proto, cache, model):
  n_proto, fea_dim = proto.shape
  n_cache, fea_dim = cache.shape
  raw_score = model['w']( model['nonlinear'](model['w1'](proto).view(n_proto, 1, fea_dim) + model['w2'](cache).view(1, n_cache, fea_dim) ) )
  return raw_score.squeeze(-1)


# this model does not need query, only key and value
class MultiHeadURT_value(nn.Module):
  def __init__(self, fea_dim, hid_dim, temp=1, n_head=1):
    super(MultiHeadURT_value, self).__init__()
    self.w1 = nn.Linear(fea_dim, hid_dim)
    self.w2 = nn.Linear(hid_dim, n_head)
    self.temp     = temp

  def forward(self, cat_proto):
    # cat_proto n_class*8*512
    n_class, n_extractors, fea_dim = cat_proto.shape
    raw_score = self.w2(self.w1(cat_proto)) # n_class*8*n_head 
    score   = F.softmax(self.temp * raw_score, dim=1)
    return score


class URTPropagation(nn.Module):

  def __init__(self, key_dim, query_dim, hid_dim, temp=1, att="cosine"):
    super(URTPropagation, self).__init__()
    self.linear_q = nn.Linear(query_dim, hid_dim, bias=True)
    self.linear_k = nn.Linear(key_dim, hid_dim, bias=True)
    #self.linear_v_w = nn.Parameter(torch.rand(8, key_dim, key_dim))
    self.linear_v_w = nn.Parameter( torch.eye(key_dim).unsqueeze(0).repeat(8,1,1)) 
    self.temp     = temp
    self.att      = att
    # how different the init is
    for m in self.modules():
      if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)

  def forward_transform(self, samples):
    bs, n_extractors, fea_dim = samples.shape
    '''
    if self.training:
      w_trans = torch.nn.functional.gumbel_softmax(self.linear_v_w, tau=10, hard=True)
    else:
      # y_soft = torch.softmax(self.linear_v_w, -1)
      # index = y_soft.max(-1, keepdim=True)[1]
      index = self.linear_v_w.max(-1, keepdim=True)[1]
      y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
      w_trans = y_hard
      # w_trans = y_hard - y_soft.detach() + y_soft
    '''
    w_trans = self.linear_v_w 
    # compute regularization
    regularization = w_trans @ torch.transpose(w_trans, 1, 2)
    samples = samples.view(bs, n_extractors, fea_dim, 1)
    w_trans = w_trans.view(1, 8, fea_dim, fea_dim)
    return torch.matmul(w_trans, samples).view(bs, n_extractors, fea_dim), (regularization**2).sum()

  def forward(self, cat_proto):
    # cat_proto n_class*8*512 
    # return: n_class*8
    n_class, n_extractors, fea_dim = cat_proto.shape
    q       = cat_proto.view(n_class, -1) # n_class * 8_512
    k       = cat_proto                   # n_class * 8 * 512
    q_emb   = self.linear_q(q)            # n_class * hid_dim
    k_emb   = self.linear_k(k)            # n_class * 8 * hid_dim  | 8 * hid_dim
    if self.att == "cosine":
      raw_score   = F.cosine_similarity(q_emb.view(n_class, 1, -1), k_emb.view(n_class, n_extractors, -1), dim=-1)
    elif self.att == "dotproduct":
      raw_score   = torch.sum( q_emb.view(n_class, 1, -1) * k_emb.view(n_class, n_extractors, -1), dim=-1 ) / (math.sqrt(fea_dim)) 
    else:
      raise ValueError('invalid att type : {:}'.format(self.att))
    score   = F.softmax(self.temp * raw_score, dim=1)

    return score