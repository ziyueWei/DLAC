#!/usr/bin/env python
# coding:utf-8

import torch.nn as nn
from models.structure_model.structure_encoder import StructureEncoder
from models.text_encoder import TextEncoder
from models.embedding_layer import EmbeddingLayer
from models.text_feature_propagation import DLACTP
import requests
import json

class DLAC(nn.Module):
    def __init__(self, config, vocab, model_mode='TRAIN'):
        """
        Hierarchy-Aware Global Model class
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param model_type: Str, ('HiAGM-TP' for the serial variant of text propagation,
                                 'HiAGM-LA' for the parallel variant of multi-label soft attention,
                                 'Origin' without hierarchy-aware module)
        :param model_mode: Str, ('TRAIN', 'EVAL'), initialize with the pretrained word embedding if value is 'TRAIN'
        """
        super(DLAC, self).__init__()
        self.config = config
        self.vocab = vocab
        self.device = config.train.device_setting.device
        self.dataset = config.data.dataset

        self.token_map, self.label_map = vocab.v2i['token'], vocab.v2i['label']

        # self.kg_sup(self.label_map)


        self.token_embedding = EmbeddingLayer(
            vocab_map=self.token_map,
            embedding_dim=config.embedding.token.dimension,
            vocab_name='token',
            config=config,
            padding_index=vocab.padding_index,
            pretrained_dir=config.embedding.token.pretrained_file,
            model_mode=model_mode,
            initial_type=config.embedding.token.init_type
        )

        self.text_encoder = TextEncoder(config)
        self.structure_encoder = StructureEncoder(config=config,
                                                      label_map=vocab.v2i['label'],
                                                      device=self.device,
                                                      graph_model_type=config.structure_encoder.type)
        self.structure_encoder_label = StructureEncoder(config=config,
                                                      label_map=vocab.v2i['label'],
                                                      device=self.device,
                                                      graph_model_type=config.structure_encoder.type)

        self.dlac = DLACTP(config=config,
                                 device=self.device,
                                 graph_model=self.structure_encoder,
                                 label_map=self.label_map,
                                 graph_model_label=self.structure_encoder_label,
                                 model_mode=model_mode)

    def optimize_params_dict(self):
        """
        get parameters of the overall model
        :return: List[Dict{'params': Iteration[torch.Tensor],
                           'lr': Float (predefined learning rate for specified module,
                                        which is different from the others)
                          }]
        """
        params = list()
        params.append({'params': self.text_encoder.parameters()})
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.dlac.parameters()})
        return params

    def kg_sup(self, label_map):
        # wzy补充:在这里插入知识图谱补充信息
        instance_list = []
        for i in range(len(label_map)):
            cur_label = self.vocab.i2v['label'][i]

            if cur_label == 'CS':
                cur_label = 'computer science'
            if cur_label == 'MAE':
                cur_label = 'aerospace engineering'

            cur_label_list = cur_label.split(" ")
            KGs_list = [i.replace("'s", "").lower() for i in cur_label_list]
            KGs_list.insert(0, cur_label)

            instance = []
            for source_label in KGs_list[1:]:
                instance.append(source_label)
            for j in range(len(KGs_list)):
                count = 0
                obj = requests.get(
                    'http://api.conceptnet.io/c/en/' + KGs_list[j] + '?rel=/r/RelatedTo&limit=1000').json()
                edges = obj['edges']

                for k in range(len(edges)):
                    edge = edges[k]

                    if edge['rel']['label'] == 'RelatedTo':
                        if edge['start']['language'] == 'en' and edge['end']['language'] == 'en':

                            temp_list_start = edge['start']['label'].split(" ")
                            for word in temp_list_start:
                               instance.append(word)

                            temp_list_end = edge['end']['label'].split(" ")
                            for word in temp_list_end:
                               instance.append(word)

                            count = count + 1

                    if count == 40:
                        break
            instance = set(instance)
            instance = list(instance)

            if cur_label == 'computer science':
                cur_label = 'CS'
            if cur_label == 'aerospace engineering':
                cur_label = 'MAE'

            instance_list.append({"label": [cur_label], "instance": instance})

        f = open('wos_kg_sup.json', 'w')
        for line in instance_list:
            line = json.dumps(line)
            f.write(line + '\n')
        f.close()


    def forward(self, inputs):
        """
        forward pass of the overall architecture
        :param batch: DataLoader._DataLoaderIter[Dict{'token_len': List}], each batch sampled from the current epoch
        :return:
        """
        if inputs[1] == "TRAIN":
            batch, mode, label_repre = inputs
            # get distributed representation of tokens, (batch_size, max_length, embedding_dimension)
            embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))
            # get the length of sequences for dynamic rnn, (batch_size, 1)
            seq_len = batch['token_len']
            token_output = self.text_encoder(embedding, seq_len)

            logits, text_repre, tail_logits = self.dlac(
                [token_output, mode, batch['head_label_list'], label_repre])
            return logits, text_repre, tail_logits

        else:
            batch, mode = inputs[0], inputs[1]
            embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))
            seq_len = batch['token_len']
            token_output = self.text_encoder(embedding, seq_len)

            logits = self.dlac([token_output, mode])
            return logits
            
    def get_embedding(self, inputs):
        batch, mode = inputs[0], inputs[1]
        embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))
        seq_len = batch['token_len']
        token_output = self.text_encoder(embedding, seq_len)
        return token_output.view(token_output.shape[0], -1)
        #return embedding.mean(1)
