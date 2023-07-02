import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


from models import Informer, Autoformer



class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_models =10
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        # Encoder
        self.encoder=[]
        for i in range (0, self.num_models):
            self.encoder.append(Encoder(
                [
                    EncoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ) for l in range(configs.e_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model)
            ))


        # Decoder
        self.decoder=[]
        for i in range (0, self.num_models):
            self.decoder.append(Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            ))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        
        # enc
        eo = self.enc_embedding(x_enc, x_mark_enc)
        enc_out=[]
        attns=[]
        for i in range (0, self.num_models):
            e, a = self.encoder[i](eo, attn_mask=enc_self_mask)
            enc_out.append(e)
            attns.append(a)
        

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part=[]
        trend_part=[]
        for i in range (0, self.num_models):
            s, t  = self.decoder[i](dec_out, enc_out[i], x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
            seasonal_part.append(s)
            trend_part.append(t)


        # final
        #print(trend_part[0].shape)
        #print(seasonal_part[0].shape)
        # dec_out = trend_part + seasonal_part
        dec_out = []
        for i in range (0, self.num_models):
            dec_out.append(trend_part[i] + seasonal_part[i])

        dec_out = torch.stack(dec_out)
        # print(dec_out)
        # print(dec_out.shape)
        # dec_out = dec_out.mean(axis=1)
        dec_out = torch.mean(dec_out,axis=0)
        # print(dec_out.shape)
        # print("check attns")
        # print(attns[0])
        if self.output_attention:
            # return dec_out[:, -self.pred_len:, :], attns
            return dec_out[:, -self.pred_len:, :], attns[0]
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]






#################################################################URT###################################################


def get_dotproduct_score(dec_out_q, dec_out_k, model):
  dec_out_q = model['linear_q'](dec_out)
  dec_out_k = model['linear_k'](dec_out)
  raw_score   = F.cosine_similarity(dec_out_q .unsqueeze(1), dec_out_k.unsqueeze(0), dim=-1)
  return raw_score  

  print('raw_score',shape.raw_score)


def get_mlp_score(dec_out_q, dec_out_k, model):
  dec_out_q, fea_dim = dec_out_q.shape
  dec_out_k, fea_dim = dec_out_k.shape
  raw_score = model['w']( model['nonlinear'](model['w1'](dec_out_q).view(dec_out_q, 1, fea_dim) + model['w2'](dec_out_k).view(1, fea_dim) ) )
  return raw_score.squeeze(-1)


#def forward(self, cat_proto):
    # cat_proto n_class*8*512
 #   n_class, n_extractors, fea_dim = cat_proto.shape
 #   raw_score = self.w2(self.w1(cat_proto)) # n_class*8*n_head 
 #   score   = F.softmax(self.temp * raw_score, dim=1)
  #  return score


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




















































# class Model(nn.Module):
    # def __init__(self, configs):
    #     super(Model, self).__init__()
    #     self.num_autoformer = 10
    #     self.autoformer_array=[]
    #     for m in range(0,self.num_autoformer):
    #         aut = Model(configs).float()
    #         self.autoformer_array.append(Autoformer(configs))
    #     print("create autoformer array")
    #     print(self.autoformer_array)


    # def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
    #             enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
    #     return 0



# # Define your base models
# model1 = autoformer
# model2 = autoformer
# model3 = autoformer

# torch.nn.init.xavier_uniform(model1.weights)
# torch.nn.init.xavier_uniform(model2.weights)
# torch.nn.init.xavier_uniform(model3.weights)

# # Define the weights for each model
# #weights = torch.tensor([0.4, 0.3, 0.3], dtype=torch.float32)

# # Define an input
# x = dec_out


# # Compute the output of each model
# output1 = model1(x)
# output2 = model2(x)
# output3 = model3(x)

# # Concatenate the outputs
# outputs = torch.cat([output1, output2, output3], dim=1)

# # Take the weighted average of the outputs
# weighted_output = torch.sum(outputs * weights, dim=1)

# print(weighted_output)


# modelarray = []
# numModels = 10
# for m in range(0,numModels):
#   autoformer = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1))
#   modelarray.Concatenate(autoformer)

# for m in range(0,numModels):
#   torch.nn.init.xavier_uniform(modelarray[m].weight)

#   #modelarray[m].train()
