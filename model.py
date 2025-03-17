import torch.nn as nn
import torch

class FSNet(nn.Module):
    def __init__(self, datainfo):
        super(FSNet, self).__init__()
        self.length_num = datainfo.length_num
        self.num_class = datainfo.num_class
        self.max_train_len = datainfo.max_train_len

        self.embedding = nn.Embedding(self.length_num, 128)

        self.encoder = nn.Sequential(
            nn.Dropout(0.3),
            nn.GRU(input_size=128, 
                hidden_size=128, 
                num_layers=2, 
                dropout=0.3, 
                bidirectional=True),
        )

        self.decoder = nn.GRU(input_size=2 * 2 * 128, 
                          hidden_size=128, 
                          num_layers=2, 
                          dropout=0.3, 
                          bidirectional=True)

        self.reconstruct = nn.Sequential(
            nn.Linear(2 * 128, 128),
            nn.SELU(),
            nn.Linear(128, self.length_num)
        )

        # 传入前 resharp一下，展成一维
        self.compress = nn.Sequential(
            nn.Linear(2 * 2 * 128 + 2 * 2 * 128, 128),
            nn.SELU(),
            nn.Dropout(0.3)
        )

        # 接compress的输出
        self.classify = nn.Sequential(
            nn.Linear(128, self.num_class)
        )

    def _transitions(self, h_t):
        fw1 = h_t[0, :, :]
        fw2 = h_t[1, :, :]
        bw1 = h_t[2, :, :]
        bw2 = h_t[3, :, :]
        return torch.cat((fw1, bw1, fw2, bw2), 1)

    def forward(self, x, y):
        rec_loss = 0
        c_loss = 0

        x = x.transpose(0, 1)

        #TODO
        #1.获取batch中的Max_seq_len -> 去除非零后的 -> 取最大值 -> 对所有序列裁切 Max_seq_len
        #2.获取对序列非零的mask，后续作为重构损失的加权
        Max_seq_len = self.max_train_len
        # mask = torch.where(x != 0, torch.tensor(1), torch.tensor(0))

        # 嵌入获取编码
        x_embed = self.embedding(x)

        # 编码器
        _, enc_h_t = self.encoder(x_embed)
        # enc_h_t -> (2 * 2, batch_size, 128)
        # enc_h_t -> 用于解码器 和 分类器

        # dec_h_t -> dec_input
        # 按照 fw1, bw1, fw2, bw2的顺序展平
        # 按照复制 max_seq_len 份 
        # 最后dec_input形状为[max_seq_len, batch_size, 2 * 2 * 128]
        dec_input = self._transitions(enc_h_t)
        dec_input = dec_input.unsqueeze(0).repeat(Max_seq_len, 1, 1)

        # 解码器
        decoder_h_i, dec_h_t = self.decoder(dec_input)
        # dec_h_t -> (2 * 2, batch_size, 128) -> 分类
        # decoder_h_i -> (L, batch_size, 2 * 128) -> 重构

        #TODO 重构损失
        # print(decoder_h_i.shape)
        x_reco = self.reconstruct(decoder_h_i)
        x_reco = x_reco.reshape(-1, self.length_num)
        label = x.reshape(-1)

        rec_loss = nn.functional.cross_entropy(x_reco, label, reduction='mean')


        #TODO 分类损失
        enc_h_t = self._transitions(enc_h_t)
        dec_h_t = self._transitions(dec_h_t)
        feature = torch.cat([enc_h_t,dec_h_t],dim=1)
        feature = self.compress(feature)
        logits = self.classify(feature)
        pred = torch.argmax(logits, dim=1)
        c_loss = nn.functional.cross_entropy(logits, y , reduction='mean')

        loss = c_loss + 1 * rec_loss

        return loss, pred


if __name__ == '__main__':
    model = FSNet()
    print(model)