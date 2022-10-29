

import einops
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

# This is the implementation of the Graph Concolution Network (GCN) for self-intention prediction
# Inputs to the model: self-speed, KLT to detect facial keypoints.
# Outputs Driver Intention
# It should be trained on the Brain4Cars Dataset


class HeteroGraph():
    """ The Heterogenous Graph Representation
    How to use:
        1. graph = HeteroGraph(types=3)
        2. A = graph.get_adjacency()
        3. A = code to modify A
        4. normalized_A = graph.normalize_adjacency(A)
    """

    def __init__(self, num_node=120, num_hetero_types=3, **kwargs):
        self.num_node = num_node
        self.num_hetero_types = num_hetero_types

    def get_adjacency_batch(self, relation_matrix):
        """
        relation_matrix is shape of (batch_size, num_obj, num_obj)
        relation_matrix[i,j,k] is the relation type between the j-th node and k-th node in the i-th example.
        """
        adjacency = (relation_matrix > 0).float()
        self.relation_matrix = relation_matrix
        return adjacency

    def normalize_adjacency_batch(self, A):
        Dl = A.sum(axis=1)
        Dl[Dl == 0] = float('inf')
        # Dn = torch.zeros_like(A, dtype=torch.float32)
        # batch_indices, obj_indices = torch.where(Dl > 0)
        # Dn[batch_indices, obj_indices, obj_indices] = Dl.view(-1) ** (-1)
        AD = A / Dl.unsqueeze(
            1)  # (batch_size, object_num, object_num) / (batch_size, 1, object_num) -> (batch_size, object_num, object_num)
        A_normalized = torch.zeros((A.size(0), self.num_hetero_types, self.num_node, self.num_node))
        for i, type_ in enumerate(range(1, self.num_hetero_types + 1)):
            A_normalized[:, i][self.relation_matrix == type_] = AD[self.relation_matrix == type_]

        return A_normalized

    def get_adjacency(self, relation_matrix):
        """
        relation_matrix[i,j] is the relation type between the i-th node and j-th node.
        """
        assert int(relation_matrix.max()) <= self.num_hetero_types and int(relation_matrix.min()) >= 0
        adjacency = (relation_matrix > 0).astype(float)
        self.relation_matrix = relation_matrix
        return adjacency

    def normalize_adjacency(self, A):
        Dl = np.sum(A, 0)
        Dn = np.zeros((self.num_node, self.num_node))
        for i in range(self.num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)

        A = np.zeros((self.num_hetero_types, self.num_node, self.num_node))
        for i, type_ in enumerate(range(1, self.num_hetero_types + 1)):
            A[i][self.relation_matrix == type_] = AD[self.relation_matrix == type_]

        return A


class Graph():
    """ The Graph Representation
    How to use:
        1. graph = Graph(max_hop=1)
        2. A = graph.get_adjacency()
        3. A = code to modify A
        4. normalized_A = graph.normalize_adjacency(A)
    """

    def __init__(self,
                 num_node=120,
                 max_hop=1,
                 **kwargs):
        self.max_hop = max_hop
        self.num_node = num_node

    def get_adjacency_batch(self, A):
        """
        A is shape of (batch_size, num_node, num_node)
        """
        transfer_mat = [torch.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = torch.stack(transfer_mat, dim=1) > 0  # (batch_size, max_hop+1, num_node, num_node)
        self.hop_dis = torch.zeros((A.size(0), self.num_node, self.num_node)) + np.inf
        for d in range(self.max_hop, -1, -1):
            self.hop_dis[arrive_mat[:, d]] = d  # when d = 0, all self-connections will be set to 0
        adjacency = (self.hop_dis <= self.max_hop).float()
        return adjacency

    def normalize_adjacency_batch(self, A):
        Dl = A.sum(axis=1)
        Dl[Dl == 0] = float('inf')
        # Dn = torch.zeros_like(A, dtype=torch.float32)
        # batch_indices, obj_indices = torch.where(Dl > 0)
        # Dn[batch_indices, obj_indices, obj_indices] = Dl.view(-1) ** (-1)
        AD = A / Dl.unsqueeze(
            1)  # (batch_size, object_num, object_num) / (batch_size, 1, object_num) -> (batch_size, object_num, object_num)
        A_normalized = torch.zeros((A.size(0), self.max_hop + 1, self.num_node, self.num_node))
        for i, type_ in enumerate(range(0, self.max_hop + 1)):
            A_normalized[:, i][self.hop_dis == type_] = AD[self.hop_dis == type_]

        return A_normalized

    def get_adjacency(self, A):
        # compute hop steps
        self.hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(self.max_hop, -1, -1):
            self.hop_dis[arrive_mat[d]] = d

        # compute adjacency
        adjacency = (self.hop_dis <= self.max_hop).astype(float)
        # valid_hop = range(0, self.max_hop + 1)
        # adjacency = np.zeros((self.num_node, self.num_node))
        # for hop in valid_hop:
        # 	adjacency[self.hop_dis == hop] = 1
        return adjacency

    def normalize_adjacency(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)

        valid_hop = range(0, self.max_hop + 1)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = AD[self.hop_dis == hop]
        return A


class HeteroGraphConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A, transform_matrix, **kwargs):
        assert A.size(1) == self.kernel_size # x: (batch_size, 64, T, V) = (batch, 64, 15, 260)
        x = self.conv(x)
        n, kc, t, v = x.size()
        assert kc // self.kernel_size == transform_matrix.size(1) == transform_matrix.size(2)
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x_transformed = torch.einsum('nkctv,kcx->nkxtv', [x, transform_matrix]) # the shape of x_transformed is the same as x
        x_convoluted = torch.einsum('nkxtv,nkvw->nxtw', [x_transformed, A])
        #x = torch.einsum('nkctv,nkvw->nctw', (x, A))

        return x_convoluted.contiguous(), A

class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A, **kwargs):
        assert A.size(1) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()

        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,nkvw->nctw', (x, A))

        return x.contiguous(),





class Graph_Conv_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 use_hetero_graph=False,
                 **kwargs):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        #
        if use_hetero_graph:
            self.gcn = HeteroGraphConv(in_channels, out_channels, kernel_size[1])
        else:
            self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=False),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, A, transform_matrix=None):
        res = self.residual(x)
        x, A = self.gcn(x, A, transform_matrix=transform_matrix)
        x = self.tcn(x) + res
        return self.relu(x),


class EncoderRNN(nn.Module):
    def __init__(self, type, input_size, hidden_size, num_layers):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.lstm = nn.LSTM(input_size, hidden_size*30, num_layers, batch_first=True)
        if type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size * 30, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size * 30, num_layers, batch_first=True)

    def forward(self, input):
        output, hidden = self.rnn(input)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, type, hidden_size, output_size, num_layers, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        # self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
        if type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, output_size * 30, num_layers, batch_first=True)
        else:
            self.rnn = nn.GRU(hidden_size, output_size * 30, num_layers, batch_first=True)

        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(output_size * 30, output_size)
        self.tanh = nn.Tanh()

    def forward(self, encoded_input, hidden):
        decoded_output, hidden = self.rnn(encoded_input, hidden)
        # decoded_output = self.tanh(decoded_output)
        # decoded_output = self.sigmoid(decoded_output)
        decoded_output = self.dropout(decoded_output)
        # decoded_output = self.tanh(self.linear(decoded_output))
        decoded_output = self.linear(decoded_output)
        # TODO decoded_output = self.sigmoid(self.linear(decoded_output))
        decoded_output = self.tanh(decoded_output)
        return decoded_output, hidden

class SelfAttention(nn.Module):
    """
    Implementation of plain self attention mechanism with einsum operations
    Paper: https://arxiv.org/abs/1706.03762
    Blog: https://theaisummer.com/transformer/
    """
    def __init__(self, dim):
        """
        Args:
            dim: for NLP it is the dimension of the embedding vector
            the last dimension size that will be provided in forward(x),
            where x is a 3D tensor
        """
        super().__init__()
        # for Step 1
        self.to_qvk = nn.Linear(dim, dim * 3, bias=False)
        # for Step 2
        self.scale_factor = dim ** -0.5  # 1/np.sqrt(dim)

    def forward(self, x, mask=None):
        assert x.dim() == 3, '3D tensor must be provided'

        # Step 1
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3 ]

        # decomposition to q,v,k
        # rearrange tensor to [3, batch, tokens, dim] and cast to tuple
        q, k, v = tuple(rearrange(qkv, 'b t (d k) -> k b t d ', k=3))

        # Step 2
        # Resulting shape: [batch, tokens, tokens]
        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        # Step 3
        return torch.einsum('b i j , b j d -> b i d', attention, v)


class Seq2Seq(nn.Module):
    def __init__(self, type, input_size, hidden_size, num_layers, dropout=0.5, interact_in_decoding=False,
                 max_num_object=260):
        super(Seq2Seq, self).__init__()

        # hidden_size = 2, input_size = 64
        self.num_layers = num_layers
        self.max_num_object = max_num_object
        self.interact_in_decoding = interact_in_decoding
        self.dropout = nn.Dropout(p=dropout)
        decoder_in_size = hidden_size
        if self.interact_in_decoding:
            self.self_attention = SelfAttention(hidden_size * 30)
            self.linear_pos_to_hidden = nn.Linear(hidden_size, hidden_size * 30)
            decoder_in_size = hidden_size * 30 * 2  # 120 = 60 (position_hidden) + 60 (interacted_hidden)
        self.encoder = EncoderRNN(type, input_size, hidden_size, num_layers)
        self.decoder = DecoderRNN(type, decoder_in_size, hidden_size, num_layers, dropout)

    def forward(self, in_data, last_location, pred_length, teacher_forcing_ratio=0, teacher_location=None):
        batch_size = in_data.shape[0]
        out_dim = self.decoder.output_size

        outputs = torch.zeros(batch_size, pred_length, out_dim).to(in_data.device)
        encoded_output, hidden = self.encoder(in_data)  # (N * V, T, 60), (2, N * V, 60)
        # hidden_ = hidden[0] if isinstance(hidden, tuple) else hidden
        hidden_outputs = torch.zeros(batch_size, pred_length, encoded_output.size(-1) * self.num_layers).to(
            in_data.device)
        decoder_input = last_location  # (N * V, 1, 2)
        if self.interact_in_decoding:
            pos_hidden = self.linear_pos_to_hidden(last_location)  # (N * V, 1, 2) -> (N * V, 1, 60)
            interact = self.message_passing(hidden.mean(dim=0))  # hidden: (self.num_layers, N * V, 60) -> (N * V, 60)
            decoder_input = torch.cat([pos_hidden, interact.unsqueeze(1)], dim=-1)  # (N * V, 1, 60)

        for t in range(pred_length):
            # encoded_input = torch.cat((now_label, encoded_input), dim=-1) # merge class label into input feature
            now_out, hidden = self.decoder(decoder_input,
                                           hidden)  # now_out is shape of (N * V, 1, 2), hidden is (2, N * V, 60)
            # TODO we force the model to predict the change of velocity by adding a residual connection
            # now_out += last_location
            outputs[:, t:t + 1] = now_out
            hidden_ = hidden[0] if isinstance(hidden, tuple) else hidden  # because GRU and LSTM have different outputs
            hidden_outputs[:, t:t + 1] = hidden_.permute(1, 0, 2).contiguous().view(batch_size, 1,
                                                                                    -1)  # batch_size = N * V, (N * V, 1, 120)
            teacher_force = np.random.random() < teacher_forcing_ratio
            last_location = (teacher_location[:, t:t + 1] if (type(teacher_location) is not type(
                None)) and teacher_force else now_out)
            decoder_input = last_location

            if self.interact_in_decoding:
                pos_hidden = self.linear_pos_to_hidden(last_location)
                interact = self.message_passing(hidden_.mean(dim=0))
                decoder_input = torch.cat([pos_hidden, interact.unsqueeze(1)], dim=-1)  # (N * V, 1, 60)

        return outputs, hidden_outputs

    def message_passing(self, origin_input, mask=None):
        """
        origin_input: (2, N*V, 60)
        mask: (N, V, V)
        """
        # output_size, NV, hidden_size = origin_input.size()
        # (NV, 60) -> (N, V, 60)
        # einops.rearrange(origin_input, '(n v) o -> n v o', v=self.num_object)
        input = origin_input.view(-1, self.max_num_object, origin_input.size(-1))
        # input = origin_input.permute(1, 0, 2).contiguous().view(NV, output_size*hidden_size).view(-1, self.num_object, output_size*hidden_size)
        output = self.self_attention(input, mask)  # (N, V, 60)
        output = self.activate(self.dropout(output))

        return output.view(-1, origin_input.size(-1))  # (N * V, 60)



class GCN(nn.Module):
    def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A_shape = (graph_args['max_hop']+1, graph_args['num_node'], graph_args['num_node'])

        # build networks
        spatial_kernel_size = A_shape[0]
        temporal_kernel_size = 5 #9 #5 # 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # best
        self.st_gcn_networks = nn.ModuleList((
            nn.BatchNorm2d(in_channels),
            Graph_Conv_Block(in_channels, 64, kernel_size, 1, residual=True, **kwargs),
            Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
            Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(A_shape)) for i in self.st_gcn_networks]
                )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.num_node = num_node = self.graph.num_node
        self.out_dim_per_node = out_dim_per_node = 2 #(x, y) coordinate
        self.seq2seq_type = kwargs.get('seq2seq_type', 'gru')
        self.seq2seq = Seq2Seq(self.seq2seq_type, input_size=(64), hidden_size=2, num_layers=2, dropout=0.5)


    def reshape_for_lstm(self, feature):
        # prepare for skeleton prediction model
        '''
        N: batch_size
        C: channel
        T: time_step
        V: nodes
        '''
        N, C, T, V = feature.size()
        now_feat = feature.permute(0, 3, 2, 1).contiguous() # to (N, V, T, C)
        now_feat = now_feat.view(N*V, T, C)
        return now_feat

    def reshape_from_lstm(self, predicted):
        # predicted (N*V, T, C)
        NV, T, C = predicted.size()
        now_feat = predicted.view(-1, self.num_node, T, self.out_dim_per_node) # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
        now_feat = now_feat.permute(0, 3, 2, 1).contiguous() # (N, C, T, V)
        return now_feat

    def forward(self, pra_x, pra_A, pra_pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None):
        x = pra_x

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            if type(gcn) is nn.BatchNorm2d:
                x = gcn(x)
            else:
                x, _ = gcn(x, pra_A + importance)

        # prepare for seq2seq lstm model
        graph_conv_feature = self.reshape_for_lstm(x)
        last_position = self.reshape_for_lstm(pra_x[:,:2]) #(N, C, T, V)[:, :2] -> (N, T, V*2) [(N*V, T, C)]

        if pra_teacher_forcing_ratio > 0 and type(pra_teacher_location) is not type(None):
            pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)

        # now_predict.shape = (N, T, V*C)
        now_predict, _ = self.seq2seq(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
        now_predict = self.reshape_from_lstm(now_predict) # (N, C, T, V)
        return now_predict, None