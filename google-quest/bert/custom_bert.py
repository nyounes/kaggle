import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel
from transformers.modeling_bert import BertPreTrainedModel


class CustomBert(BertPreTrainedModel):

    def __init__(self, config, num_labels=30):
        super(CustomBert, self).__init__(config)

        config.output_hidden_states=True
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(4 * config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None,
                extra_feats=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        hidden_states = outputs[2] #hidden_states: 12 layers tuples each is of (batch_size, sequence_length, hidden_size) + embedding``

        # h12 = hidden_states[-1][:, 0].reshape((-1, 1, 768))
        # h11 = hidden_states[-2][:, 0].reshape((-1, 1, 768))
        # h10 = hidden_states[-3][:, 0].reshape((-1, 1, 768))
        # h9  = hidden_states[-4][:, 0].reshape((-1, 1, 768))
        h12 = hidden_states[-1][:, 0]
        h11 = hidden_states[-2][:, 0]
        h10 = hidden_states[-3][:, 0]
        h9  = hidden_states[-4][:, 0]
        # lin_h12 = F.relu(self.bn(self.linear(h12)))
        # lin_output = self.dropout(lin_h12)
        # lin_h11 = F.relu(self.bn(self.linear(h11)))
        # lin_output = self.dropout(lin_h11)
        # lin_h10 = F.relu(self.bn(self.linear(h10)))
        # lin_output = self.dropout(lin_h10)
        # lin_h9 = F.relu(self.bn(self.linear(h9)))
        # lin_output = self.dropout(lin_h9)

        all_h = torch.cat([h9, h10, h11, h12], 1)
        # mean_pool = torch.mean(all_h, 1)

        pooled_output = self.dropout(all_h)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs
