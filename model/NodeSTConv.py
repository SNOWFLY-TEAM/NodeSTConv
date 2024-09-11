import torch
from pytorch_tcn import TCN
from torch import nn
from torch.nn import Sequential, Linear, Dropout, Tanh, Conv2d, ReLU, ModuleList


class MultiLoss(nn.Module):
    def __init__(self, alpha):
        super(MultiLoss, self).__init__()
        self.alpha = alpha
        self.normal_loss_fn = nn.L1Loss()
        self.class_loss_fn = nn.L1Loss()

    def forward(self, pred, target):
        target_diff = target[..., 1]
        pred_diff = pred[..., 1]
        pred_val = pred[..., 0]
        target_val = target[..., 0]

        loss_val = (self.alpha * self.normal_loss_fn(pred_val, target_val) +
                    (1 - self.alpha) * self.class_loss_fn(pred_diff, target_diff))
        return loss_val


class STDCBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel_size, padding, dilation, dropout):
        super(STDCBlock, self).__init__()
        pass

    def forward(self, x):
        pass


class STDC(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, K, dropout):
        super(STDC, self).__init__()
        self.K = K
        kernel_size = K
        conv2d_layers = []
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            conv2d_layers += [STDCBlock(input_dim, hidden_dim, kernel_size, padding, dilation, dropout)]
        self.conv2d_layers = ModuleList(conv2d_layers)

    def forward(self, x, pe):
        pass
        return x[:, :, self.K]


class ETC(nn.Module):
    def __init__(self, pred_len, hist_len, hidden_dim, dropout):
        super(ETC, self).__init__()
        self.pred_len = pred_len
        self.hist_len = hist_len
        self.dynamic_tcn_encoder = TCN(2 * hidden_dim, [hidden_dim],
                                       dropout=dropout, input_shape='NLC',
                                       use_skip_connections=True,
                                       output_projection=hidden_dim)
        self.pred_mlp = Sequential(Linear(hidden_dim, 1))
        self.diff_mlp = Sequential(Linear(hidden_dim, 1))

    def forward(self, xn, static_graph_emb):
        pred_pm25 = []
        for i in range(self.pred_len):
            graph_emb = static_graph_emb[:, :self.hist_len + i]
            graph_emb = torch.cat([graph_emb, xn[:, :self.hist_len + i]], dim=-1)
            graph_emb = self.dynamic_tcn_encoder(graph_emb)
            pred_emb = graph_emb[:, -1]
            cl = self.class_mlp(pred_emb)
            pred = self.pred_mlp(pred_emb)
            x = torch.cat([pred, cl], dim=1)
            pred_pm25.append(x)
            xn = torch.cat((xn, self.pm25_mlp(x).unsqueeze(1)), dim=1)
        pred_pm25 = torch.stack(pred_pm25, dim=1)
        return pred_pm25


class NodeSTConv(nn.Module):
    def __init__(self, hist_len, pred_len, device, batch_size, in_dim, hidden_dim, city_num,
                 dropout, layers, K):
        super(NodeSTConv, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.device = device
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.city_num = city_num
        self.dropout = dropout
        self.layers = layers
        self.date_emb = hidden_dim // 4
        self.loc_emb = hidden_dim // 4
        self.K = K

        self.emb_month = nn.Embedding(12, self.date_emb)
        self.emb_weekday = nn.Embedding(7, self.date_emb)
        self.emb_hour = nn.Embedding(24, self.date_emb)

        self.loc_mlp = Sequential(Linear(2, self.loc_emb),
                                  Tanh())
        self.pm25_mlp = Sequential(Linear(2, self.hidden_dim),
                                   Tanh())

        self.dynamic_mlp = Sequential(Linear(3 * self.date_emb + self.loc_emb, self.hidden_dim),
                                      Tanh(),
                                      Dropout(self.dropout),
                                      Linear(self.hidden_dim, self.hidden_dim),
                                      Tanh())

        self.stdc = STDC(self.in_dim, self.hidden_dim, self.layers, self.K, self.dropout)
        self.etc = ETC(self.pred_len, self.hist_len, self.hidden_dim, self.dropout)

    def forward(self, pm25_hist, features, city_locs, date_emb):
        features = features.transpose(1, 2)
        batch_size = features.shape[0]
        hist_len = pm25_hist.shape[1]
        pred_len = features.shape[1] - hist_len

        xn = pm25_hist
        all_month_emb = self.emb_month(date_emb[:, :, 2] - 1)
        all_weekday_emb = self.emb_weekday(date_emb[:, :, 1] - 1)
        all_hour_emb = self.emb_hour(date_emb[:, :, 0])
        city_loc_emb = self.loc_mlp(city_locs)
        dynamic_graph_emb = torch.cat([city_loc_emb.reshape(batch_size, 1, -1).repeat(1, hist_len + pred_len, 1),
                                       all_month_emb,
                                       all_weekday_emb,
                                       all_hour_emb], dim=-1)
        dynamic_graph_emb = self.dynamic_mlp(dynamic_graph_emb)
        xn = self.pm25_mlp(xn)
        static_graph_emb = self.stdc(features, dynamic_graph_emb)
        pred_pm25 = self.etc(xn, static_graph_emb)
        return pred_pm25
