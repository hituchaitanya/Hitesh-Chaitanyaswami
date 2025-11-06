import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, imgH=32, nc=1, nclass=80, nh=256):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, "imgH must be multiple of 16"
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1), nn.ReLU(True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,1,1), nn.ReLU(True),
            nn.Conv2d(256,256,3,1,1), nn.ReLU(True),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512,512,3,1,1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(512,512,2,1,0), nn.ReLU(True)
        )
        self.rnn = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True, num_layers=2, batch_first=True)
        )
        self.embedding = nn.Linear(nh*2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "the height after conv must be 1"
        conv = conv.squeeze(2)  # b x c x w
        conv = conv.permute(0,2,1)  # b x w x c (seq_len)
        rnn_out, _ = self.rnn[0](conv)
        output = self.embedding(rnn_out)
        return output.log_softmax(2)
