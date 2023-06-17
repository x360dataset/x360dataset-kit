import torch
import torch.nn as nn


class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output



class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100, dropout_rate=0.1):
        super(ConcatFusion, self).__init__()
        
        self.fc_out = nn.Sequential(
            # nn.Dropout(dropout_rate),
            nn.Linear(input_dim, output_dim)
        )

        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        # self.attention_v = nn.Sequential(
        #     nn.Linear(input_dim - input_dim//2, input_dim - input_dim//2),
        #     nn.Sigmoid()
        # )

    def forward(self, x, y, use_audio=False):

        # x = self.attention_v(x) * x
        # y = self.attention_a(y) * y

        if use_audio:
            output = torch.cat((x, y), dim=1)
        else:
            output = x

        output = self.attention(output) * output
        output = self.fc_out(output)
        
        return output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()
        # self.attention_y = nn.Sequential(
        #     nn.Linear(input_dim, input_dim),
        #     nn.Sigmoid()
        # )
        # self.attention_x = nn.Sequential(
        #     nn.Linear(input_dim, input_dim),
        #     nn.Sigmoid()
        # )

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, use_audio=False):
        # x = self.attention_x(x) * x
        # y = self.attention_y(y) * y

        out_x = self.fc_x(x)

        if use_audio:
            out_y = self.fc_y(y)

            if self.x_gate:
                gate = self.sigmoid(out_x)
                output = self.fc_out(torch.mul(gate, out_y))
            else:
                gate = self.sigmoid(out_y)
                output = self.fc_out(torch.mul(out_x, gate))
        else:
            output = self.fc_out(out_x)
        # out_x, out_y,
        return output

