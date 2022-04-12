# coding=utf-8
# Copyleft 2019 project LXRT.

import torch.nn as nn

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 20


class GQAModel(nn.Module):
    def __init__(self, num_answers, visual_pos_dim=4, gqa_dropout_rate=args.gqa_dropout_rate):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_GQA_LENGTH,
            visual_pos_dim = visual_pos_dim
        )
        hid_dim = self.lxrt_encoder.dim
        if args.gqa_dropout_rate > 0.0:
            self.logit_fc = nn.Sequential(
                nn.Dropout(p=gqa_dropout_rate),
                nn.Linear(hid_dim, hid_dim * 2),
                GeLU(),
                BertLayerNorm(hid_dim * 2, eps=1e-12),
                nn.Dropout(p=gqa_dropout_rate),
                nn.Linear(hid_dim * 2, num_answers)
            )
        else:
            self.logit_fc = nn.Sequential(
                nn.Linear(hid_dim, hid_dim * 2),
                GeLU(),
                BertLayerNorm(hid_dim * 2, eps=1e-12),
                nn.Linear(hid_dim * 2, num_answers)
            )

        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        #import pdb; pdb.set_trace()
        x = self.lxrt_encoder(sent, (feat, pos))
        logit = self.logit_fc(x)

        return logit


