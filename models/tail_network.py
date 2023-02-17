import torch.nn as nn
import math
import torch
import numpy as np
import os
from models.structure_model.structure_encoder import StructureEncoder
from models.text_label_mi_discriminator import TextLabelMIDiscriminator
from models.labelprior_discriminator import LabelPriorDiscriminator
import torch.nn.functional as F

class TailNet(nn.Module):
    def __init__(self, config, graph_model_label=None, label_map=None):
        super(TailNet, self).__init__()
        self.dataset = config.data.dataset
        self.label_map = label_map

        self.embedding_tail_net = nn.Sequential(nn.Linear(len(self.label_map) * config.model.linear_transformation.node_dimension, 300),
                                                nn.ReLU(),
                                                nn.Dropout(0.5),
                                                nn.Linear(300, 300),
                                                nn.ReLU(),
                                                nn.Dropout(0.5))
        # self.label_tail_encoder = nn.Sequential(nn.Linear(config.embedding.label.dimension, 200),
        #                                        nn.ReLU(),
        #                                        nn.Dropout(0.5),
        #                                        nn.Linear(200, 200),
        #                                        nn.ReLU(),
        #                                        nn.Dropout(0.5))

        self.graph_model = graph_model_label
        self.label_map = label_map

        self.dropout = nn.Dropout(p=config.model.classifier.dropout)
        self.linear = nn.Linear(600, len(label_map))


    def forward(self, text, head_label_list, label_repre):
        """
        forward pass of matching learning
        :param gather_positive ->  torch.BoolTensor, (batch_size, positive_sample_num), index of positive label
        :param gather_negative ->  torch.BoolTensor, (batch_size, negative_sample_num), index of negative label
        :param label_repre ->  torch.FloatTensor, (batch_size, label_size, label_feature_dim)
        """

        label_repre = label_repre.unsqueeze(0)
        # label_repre = label_repre.repeat(text.size(0), 1, 1)
        label_repre = self.graph_model(label_repre)
        label_repre = label_repre.squeeze()
        # label_repre = self.label_encoder(label_repre)

        for i in range(len(head_label_list)):
            if(i == 0):
                label1 = label_repre[head_label_list[i][0]].unsqueeze(0)
            else:
                label1 = torch.cat((label1, label_repre[head_label_list[i][0]].unsqueeze(0)), dim=0)
        text_encoder = self.embedding_tail_net(text)
        text_label_rep = torch.cat((text_encoder, label1), dim=1)

        logits = self.dropout(self.linear(text_label_rep))

        return logits