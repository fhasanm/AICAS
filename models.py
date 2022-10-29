import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# This is the implementation of the Multi-Head Attention Seq2Seq (MASS) network for neighboring trajectory prediction
# Inputs to the model: historical trajectory
# Outputs Future Trajectory
# It should be trained on the BLVD or NGSIM Dataset


# ---------------------------------layers-----------------------------------------

class pos_embedding_layer(nn.Module):

    def __init__(self, d_out, n_posidx=200):
        super(pos_embedding_layer, self).__init__()
        self.pos_emb = nn.Embedding(n_posidx, d_out)

    def forward(self, x):
        posidxs = torch.LongTensor([i for i in range(x.size(1))]).unsqueeze(0).to(x.device)
        pos_embeds = self.pos_emb(posidxs)

        return x + pos_embeds

#*
class MHAlayer(nn.Module):

    def __init__(self, n_heads, d_model, p=0.1):
        super(MHAlayer, self).__init__()
        self.n_heads = n_heads
        self.d_q = d_model / n_heads
        self.d_k = d_model / n_heads
        self.d_v = d_model / n_heads
        assert d_model % n_heads == 0, "d_model should be completely divisible by n_heads"

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)

        # Weight initialize as normal distribution
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_q)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_v)))

        self.fc = nn.Linear(d_model, d_model)
        # Weight initialize via a method described in "Understanding the difficulty of training
        # deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010)"
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input, mask=None):
        residual = input
        q = self.w_qs(input)
        k = self.w_ks(input)
        v = self.w_vs(input)
        #assert q.dim(0) == 3 and
        assert q.shape[2] == self.n_heads * self.d_q, "q must be of b*t_h*d_dim size"
        # # Change dimension of q,k,v to split the heads
        # q = torch.permute(q.reshape(q.shape[0], q.shape[1], self.n_heads, self.d_q), (2, 0, 1, 3))
        # k = torch.permute(k.reshape(k.shape[0], k.shape[1], self.n_heads, self.d_k), (2, 0, 1, 3))
        # v = torch.permute(v.reshape(v.shape[0], v.shape[1], self.n_heads, self.d_v), (2, 0, 1, 3))
        # attention = torch.einsum('hblk,hbtk->hblt', [q, k]) / np.sqrt(q.shape[-1])
        q = rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_heads)
        k = rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_heads)
        v = rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_heads)
        attention = torch.einsum('hblk,hbtk->hblt', [q, k]) / np.sqrt(q.shape[-1])
        if mask is not None:
            attention = attention.masked_fill(mask[None] == 0, -1e9)

        attention = torch.softmax(attention, 3)
        out = torch.einsum('hblt,hbtv->hblv', [attention, v])
        # Change the dimension of the output back from split-heads
        # out = torch.permute(out, (1, 2, 0, 3))
        # out = out.view(out.shape[0], out.shape[1], out.shape[2]*out.shape[3])
        out = rearrange(out, 'head b l v -> b l (head v)')
        out = self.fc(out)
        out = self.dropout(out)
        out = out + residual
        out = self.layer_norm(out)

        return out, attention


class FFlayer(nn.Module):

    def __init__(self, d_in, d_hidden, p=0.1):
        super(FFlayer, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        residual = x

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x


# Socio-Temporal Feature Extraction block
class STMHAlayer(nn.Module):

    def __init__(self, d_model, d_ffn, n_heads, p, time_attn_layer=True, soc_attn_layer=True):
        super(STMHAlayer, self).__init__()
        if time_attn_layer and soc_attn_layer:
            self.time_attn_layer = time_attn_layer
            self.soc_attn_layer = soc_attn_layer
            self.time_attention = MHAlayer(n_heads, d_model, p)
            self.soc_attention = MHAlayer(n_heads, d_model, p)  # soc -> social
        elif time_attn_layer:
            self.time_attn_layer = time_attn_layer
            self.time_attention = MHAlayer(n_heads, d_model, p)
        elif soc_attn_layer:
            self.soc_attention = soc_attn_layer
            self.soc_attention = MHAlayer(n_heads, d_model, p)

        self.ffn = FFlayer(d_model, d_ffn, p)

    def forward(self, input, batch_size, time_mask=None, soc_mask=None):

        # input: b*n th c

        time_attn_score = torch.zeros(input.shape[0], input.shape[1], input.shape[1])

        # Does all zeros attention affect
        out = input
        if self.time_attn_layer:
            out, time_attn_score = self.time_attention(input, time_mask)
        # Shape of time_attn_out = [b*n th c]
        # Change shape to [b*th n c] to put it into space_attention layer
        # assert time_attn_out.shape[0] % batch_size == 0, "batch_size * no of vehicle not equal to time_attn_out's first dim"
        # n_vehicles = int(time_attn_out.shape[0]/batch_size)
        # n_timesteps = int(time_attn_out.shape[1])
        # d = int(time_attn_out.shape[-1])
        # time_attn_out = torch.permute(time_attn_out.reshape(batch_size, n_vehicles, n_timesteps, d), (0, 2, 1, 3))
        # time_attn_out = time_attn_out.reshape(batch_size * n_timesteps, n_vehicles, d)
        # space_attn_out, space_attn = self.space_attention(time_attn_out, space_attn_mask)
        #
        # # Now change shape back to [b*n th c] to put it into ffn
        # space_attn_out = torch.permute(space_attn_out.reshape(batch_size, n_timesteps, n_vehicles, d), (0, 2, 1, 3))
        # space_attn_out = space_attn_out.reshape(batch_size * n_vehicles, n_timesteps, d)
        # out = self.ffn(space_attn_out)

        out = rearrange(out, '(bs no) sl hs -> (bs sl) no hs', bs=batch_size)
        soc_attn_score = torch.zeros(out.shape[0], out.shape[1], out.shape[1])
        if self.soc_attn_layer:
            out, soc_attn_score = self.soc_attention(out, soc_mask)
        out = rearrange(out, '(bs sl) no hs -> (bs no) sl hs', bs=batch_size)

        out = self.ffn(out)

        return out, time_attn_score, soc_attn_score


class RNNEncoder(nn.Module):
    def __init__(self, type, input_size, hidden_size, num_layers):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.lstm = nn.LSTM(input_size, hidden_size*30, num_layers, batch_first=True)
        if type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size * 30, num_layers, batch_first=True)
        elif type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size * 30, num_layers, batch_first=True)

    def forward(self, input):
        output, hidden = self.rnn(input)
        return output, hidden


class RNNDecoder(nn.Module):
    def __init__(self, type, hidden_size, out_size, num_layers, dropout=0.5):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_layers = num_layers
        if type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, out_size * 30, num_layers, batch_first=True)
        elif type == 'gru':
            self.rnn = nn.GRU(hidden_size, out_size * 30, num_layers, batch_first=True)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(out_size * 30, out_size)
        self.tanh = nn.Tanh()

    def forward(self, encoded_input, hidden):
        decoded_output, hidden = self.rnn(encoded_input, hidden)
        decoded_output = self.dropout(decoded_output)
        decoded_output = self.linear(decoded_output)
        decoded_output = self.tanh(decoded_output)
        return decoded_output, hidden


class STMHAblock(nn.Module):

    def __init__(self, in_size, n_layers, n_heads, d_model, d_ffn, n_posembeds=200, p=0.1, scale_emb=False,
                 time_attn=True, soc_attn=True):
        super(STMHAblock, self).__init__()
        self.embed = nn.Linear(in_size, d_model, bias=False)  # embedding does not require bias and activation
        self.pos_embed = pos_embedding_layer(d_model, n_posembeds)
        self.dropout = nn.Dropout(p)
        self.stmha_stack = nn.ModuleList(
            [STMHAlayer(d_model, d_ffn, n_heads, p, time_attn_layer=time_attn, soc_attn_layer=soc_attn) for _ in
             range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.time_attn = time_attn
        self.soc_attn = soc_attn

    def forward(self, input, time_mask, soc_mask, batch_size, return_attn_scores=False):

        out = self.embed(input)
        if self.scale_emb:
            out *= self.d_model ** 0.5
        out = self.layer_norm(self.dropout(self.pos_embed(out)))

        time_attn_scores = []
        soc_attn_scores = []
        for stmha in self.stmha_stack:
            out, time_attn_score, soc_attn_score = stmha(out, batch_size, time_mask=time_mask, soc_mask=soc_mask)
            time_attn_scores.append(time_attn_score)
            soc_attn_scores.append(soc_attn_score)

        if return_attn_scores:
            return out, time_attn_scores, soc_attn_scores

        return out

        # Seq2Seq block


#*d_k and d_v
class Seq2Seq(nn.Module):
    ##**
    def __init__(self, in_size, out_size, d_model=128, d_ffn=512,
                 n_st_layers=2, n_heads=4, dropout=0.1, n_posembeds=16, spatial_interact=True,
                 time_attn =True, soc_attn=True, future_attn=True, transformer_decoder=False, rnn_type='gru'):
        super().__init__()
        self.d_model = d_model
        self.input_encoder = STMHAblock(in_size=in_size, n_layers=n_st_layers, n_heads=n_heads,
                                        d_model=d_model, d_ffn=d_ffn, n_posembeds=n_posembeds, p=dropout, scale_emb=False, time_attn=time_attn, soc_attn=soc_attn)
        self.out_size = out_size
        self.rnn_type = rnn_type
        self.future_attn = future_attn
        self.transformer_decoder = transformer_decoder
        self.transformer_decoder_ff = nn.Linear(d_model, out_size*30)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        self.rnn_encoder = RNNEncoder(self.rnn_type, input_size=d_model, hidden_size=out_size, num_layers=2)
        self.rnn_decoder = RNNDecoder(self.rnn_type, hidden_size=out_size, out_size=out_size, num_layers=2,
                                      dropout=0.5)

        if self.future_attn:
            d_model = 2 * 60
            self.layer_norm = nn.LayerNorm(60)
            self.attention_interact = MHAlayer(n_heads=n_heads, d_model=d_model, p=dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input, soc_graph, pred_len, input_mask=None, tfr=0, teacher=None):

        batch_size, in_size, seq_len, num_v = input.size()
        input = rearrange(input, 'bs is sl no -> (bs no) sl is')

        if tfr > 0 and type(teacher) is not type(None):
            teacher = rearrange(teacher, 'bs is sl no -> (bs no) sl is')
            assert pred_len == teacher.size(-2)

        preds = torch.zeros((batch_size * num_v, pred_len, self.out_size), device=input.device)

        time_mask = None
        soc_mask = rearrange(soc_graph, 'b l m n -> (b l) m n').bool()  # (bs * sl, no, no)
        future_attn_mask = None
        if type(input_mask) is not type(None):
            # input_mask: (bs, 1, sl, no)
            time_mask = rearrange(input_mask, 'bs is sl no -> (bs no) sl is')  # (bs * no, sl, 1)
            len_s = time_mask.squeeze(-1).size(-1)
            sub_mask = (1 - torch.triu(
                torch.ones((1, len_s, len_s), device=time_mask.squeeze(-1).device), diagonal=1)).bool()
            time_mask = torch.einsum('bsi,bxi->bsx', time_mask, time_mask)
            time_mask = (sub_mask * time_mask).bool()

            if self.future_attn:
                future_attn_mask = rearrange(input_mask, 'bs is sl no -> bs no (sl is)')  # (bs, no, sl)
                future_attn_mask = future_attn_mask.sum(axis=-1, keepdim=True)  # (bs, no, 1)
                future_attn_mask = torch.einsum('boi,bui->bou', future_attn_mask,
                                               future_attn_mask).bool()  # (batch_size, num_object, num_object)

        input_encoder_out = self.input_encoder(input, time_mask, soc_mask, batch_size, return_attn_scores=False)  # enc_output: (bs * num_object, history_frames, hidden_size)

        dec_hidden = rearrange(input_encoder_out, 'bv t c -> t bv c')  # num_layers x bv x out_size*30
        dec_hidden_tup = dec_hidden.tensor_split(2, dim=0)
        dec_hidden1 = dec_hidden_tup[0].mean(dim=0, keepdim=True)
        dec_hidden2 = dec_hidden_tup[1].mean(dim=0,keepdim=True)
        dec_hidden = torch.cat((dec_hidden1, dec_hidden2), dim=0)
        dec_hidden = self.transformer_decoder_ff(dec_hidden)
        assert dec_hidden.shape == torch.Size([2, batch_size*num_v, self.out_size*30]), "dec_hidden shape incorrect"

        if not self.transformer_decoder:
            out, hidden = self.rnn_encoder(input_encoder_out)
        last_input = input[:, -1:, :2]
        dec_input = last_input

        for t in range(pred_len):
            if self.transformer_decoder:
                now_out, hidden = self.rnn_decoder(dec_input, dec_hidden)
            else:
                now_out, hidden = self.rnn_decoder(dec_input, hidden)

            preds[:, t:t + 1] = now_out
            teacher_force = np.random.random() < tfr
            last_input = (teacher[:, t:t + 1] if (type(teacher) is not type(
                None)) and teacher_force else now_out)
            dec_input = last_input
            if self.future_attn:
                hidden = self.future_attn_layer(hidden,  batch_size=batch_size, mask=future_attn_mask,)


        out = rearrange(preds, '(bs no) sl hs -> bs hs sl no', bs=batch_size)

        return out, None

    def future_attn_layer(self, hidden, batch_size, mask=None):
        shaped_hidden = rearrange(hidden, 'nl (b o) hs -> b o (nl hs)',
                                  b=batch_size)  # (batch_size, num_object, num_layers * hidden_size)
        interacted_hidden, _ = self.attention_interact(shaped_hidden, mask=mask)
        interacted_hidden = rearrange(interacted_hidden, 'b o (nl hs) -> nl (b o) hs',
                                      nl=self.rnn_decoder.num_layers).contiguous()
        interacted_hidden = self.dropout(interacted_hidden)
        hidden = self.layer_norm(hidden + interacted_hidden)
        return hidden



    #     super(Seq2Seq, self).__init__()
    #
    #     self.d_model = d_model
    #     self.num_layers = n_st_layers
    #
    #     self.input_encoder = STMHAblock(in_size=in_size, n_layers=self.num_layers_layers, n_heads=n_heads,
    #                                     d_model=d_model,
    #                                     d_ffn=d_ffn, n_posembeds=200, p=dropout, scale_emb=False)
    #
    #     self.interact_in_decoding = kwargs.get('interact_in_decoding', False)
    #     self.dropout = nn.Dropout(p=0.5)
    #     self.rnn_encoder = RNNEncoder(type=self.seq2seq_type, input_size=d_model, hidden_size=out_size,
    #                                          num_layers=2)
    #     self.rnn_decoder = RNNDecoder(type=self.seq2seq_type, hidden_size=out_size, out_size=out_size,
    #                                          num_layers=2)
    #
    #     if self.interact_in_decoding:
    #         self.d_model = 2 * 60
    #         self.layer_norm = nn.LayerNorm(60)
    #         self.attention_interact = layers.MHAlayer(n_heads=n_heads, d_model=self.d_model, p=dropout)
    #
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
    #
    # def forward(self, x, A, pred_len, input_mask=None, tfr=0, teacher_loc=None, **kwargs):
    #     # teacher, labelled output at the timestep
    #     # teacher_loc size: (b*n) t c
    #     # input_mask size:  b n t 1
    #     # x size: b c t n -> b n t c
    #     # A size: (b*t) n n
    #
    #     x = x.permute(0, 3, 2, 1)
    #     batch_size, n_vehicles, tot_timesteps, in_size = x.shape
    #     x = x.reshape(batch_size * n_vehicles, tot_timesteps, in_size)
    #
    #     velocity_out = torch.zeros((batch_size * n_vehicles, tot_timesteps, in_size))
    #
    #     # Prepare the masks
    #     temporal_mask = None
    #     dec_mask = None
    #     spatial_mask = A.bool()
    #
    #     if type(input_mask) is not type(None):
    #         mask = input_mask.reshape(batch_size * tot_timesteps, n_vehicles, 1)
    #         len_seq = mask.shape[-2]
    #         mask = torch.einsum("bso,bto->bst", input_mask, input_mask)
    #         subseq_mask = (1 - torch.triu(torch.ones((1, len_seq, len_seq)), diagonal=1)).bool()
    #         temporal_mask = (subseq_mask * mask).bool()
    #
    #         if self.interact_in_decoding:
    #             # change input_mask shape to (b, n, t)
    #             dec_mask = torch.permute(input_mask.reshape(batch_size, tot_timesteps, n_vehicles), (0, 2, 1))
    #             dec_mask = dec_mask.sum(dim=-1, keepdim=True)
    #             dec_mask = torch.einsum("bmt,bnt->bmn")
    #
    #     encoder_out, _ = self.input_encoder(x, temporal_mask, spatial_mask=spatial_mask, batch_size=batch_size)
    #     last_pos_vel = x[:, -1, 2]
    #
    #     # SEQ2SEQ
    #     s2sencoder_out, hidden = self.rnn_encoder(encoder_out)
    #     for t in range(pred_len):
    #         s2sdecoder_out, hidden = self.rnn_decoder(last_pos_vel, hidden)
    #         velocity_out[:, t + t + 1, :] = s2sdecoder_out  # (b*n, 1, 2)
    #         teacher_force = np.random.random() < tfr
    #         last_pos_vel = (teacher_loc[:, t:t + 1, :] if (type(teacher_loc) is not type(
    #             None)) and teacher_force else s2sdecoder_out)
    #         if self.interact_in_decoding:
    #             hidden = self.message_pass(hidden, batch_size=batch_size, n_vehicles=n_vehicles, mask=dec_mask)
    #
    #     # out_size=2
    #     # out shape: (b, n, pred_len, 2)
    #     out = velocity_out.reshape(batch_size, n_vehicles, tot_timesteps, 2)
    #
    #     return out
    #
    # def message_pass(self, hidden, batch_size, n_vehicles, mask=None):
    #     hidden_size = hidden.shape[-1]
    #     # change shape of hidden to be compatible for the
    #     # attenction layer: nl (b o) hs -> b o (nl hs)
    #
    #     residual = hidden
    #
    #     hidden = hidden.reshape(self.num_layers, batch_size, n_vehicles, hidden_size)
    #     hidden = torch.permute(hidden, (1, 2, 0, 3))
    #     hidden = hidden.reshape(batch_size, n_vehicles, self.num_layers * hidden_size)
    #
    #     interacted_hidden, _ = self.attention_interact(hidden, mask=mask)
    #
    #     # now change the shape back: b o (nl hs) -> nl (b o) hs
    #     interacted_hidden = interacted_hidden.reshape(batch_size, n_vehicles, self.num_layers, hidden)
    #     interacted_hidden = torch.permute(interacted_hidden, (2, 0, 1, 3))
    #     interacted_hidden = interacted_hidden.reshape(self.num_layers, batch_size * n_vehicles, hidden_size)
    #
    #     interacted_hidden = self.dropout(interacted_hidden)
    #
    #     attention_hidden = self.layer_norm(interacted_hidden + residual)
    #
    #     return attention_hidden
