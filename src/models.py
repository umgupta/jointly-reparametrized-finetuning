import logging
from typing import List, Optional, Tuple, Union

import numpy
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertForTokenClassification,
    BertLayer,
    BertModel,
    BertOutput,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaAttention,
    RobertaEmbeddings,
    RobertaEncoder,
    RobertaForSequenceClassification,
    RobertaIntermediate,
    RobertaLayer,
    RobertaModel,
    RobertaOutput,
    RobertaPooler,
)

logger = logging.getLogger()


#####################################
#### Util Code for initialization ###
#####################################


def do_zinit(vector_size, mode):
    if mode == "normal":
        return torch.randn((1, vector_size)) / numpy.sqrt(vector_size)

    if mode == "normal-div-2":
        return torch.randn((1, vector_size)) / numpy.sqrt(4 * vector_size)

    if mode == "normal-div-root-2":
        return torch.randn((1, vector_size)) / numpy.sqrt(2 * vector_size)

    if mode == "normal-*-2":
        return torch.randn((1, vector_size)) / numpy.sqrt(vector_size / 4)

    if mode == "normal-*-root-2":
        return torch.randn((1, vector_size)) / numpy.sqrt(vector_size / 2)

    if mode == "zeros":
        return torch.zeros((1, vector_size))

    if mode == "uniform":
        return (torch.rand((1, vector_size)) - 0.5) / numpy.sqrt(vector_size / 12)

    if mode == "uniform-div-2":
        return (torch.rand((1, vector_size)) - 0.5) / numpy.sqrt(4 * vector_size / 12)

    if mode == "uniform-div-root-2":
        return (torch.rand((1, vector_size)) - 0.5) / numpy.sqrt(2 * vector_size / 12)

    if mode == "uniform-*-2":
        return (torch.rand((1, vector_size)) - 0.5) / numpy.sqrt(vector_size / (12 * 4))

    if mode == "uniform-*-root-2":
        return (torch.rand((1, vector_size)) - 0.5) / numpy.sqrt(vector_size / (12 * 2))

    raise "Incorrect value for z_init"


def do_winit(vector_size, hidden_size, mode):
    if mode == "normal":
        return torch.randn((vector_size, hidden_size)) / numpy.sqrt(vector_size)

    if mode == "normal-div-2":
        return torch.randn((vector_size, hidden_size)) / numpy.sqrt(4 * vector_size)

    if mode == "normal-div-root-2":
        return torch.randn((vector_size, hidden_size)) / numpy.sqrt(2 * vector_size)

    if mode == "normal-*-2":
        return torch.randn((vector_size, hidden_size)) / numpy.sqrt(vector_size / 4)

    if mode == "normal-*-root-2":
        return torch.randn((vector_size, hidden_size)) / numpy.sqrt(vector_size / 2)

    if mode == "identity":
        assert vector_size == hidden_size, "vector size and hidden size should be same"
        return torch.eye(vector_size)

    if mode == "uniform-div-2":
        return (torch.rand((vector_size, hidden_size)) - 0.5) / numpy.sqrt(4 * vector_size / 12)

    if mode == "uniform-div-root-2":
        return (torch.rand((vector_size, hidden_size)) - 0.5) / numpy.sqrt(2 * vector_size / 12)

    if mode == "uniform":
        return (torch.rand((vector_size, hidden_size)) - 0.5) / numpy.sqrt(vector_size / 12)

    if mode == "uniform-*-2":
        return (torch.rand((vector_size, hidden_size)) - 0.5) / numpy.sqrt(vector_size / (12 * 4))

    if mode == "uniform-*-root-2":
        return (torch.rand((vector_size, hidden_size)) - 0.5) / numpy.sqrt(vector_size / (12 * 2))


#############################################################################
# This is to override Roberta to have a linear clf head; by default it has a dense head in HF code.
#############################################################################
class RobertaClassificationHeadOverride(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.clf = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        # x = self.out_proj(x)
        x = self.clf(x)
        return x


#################################
#### Torch Utilities#############
#################################
def count_params(model):
    trainable_count = 0
    all_count = 0
    for param in model.parameters():
        all_count += numpy.prod(param.shape)
        if param.requires_grad:
            trainable_count += numpy.prod(param.shape)
    return {"trainable": int(trainable_count), "all": int(all_count)}


class TorchModelUtils(nn.Module):
    def count_params(self):
        trainable_count = 0
        all_count = 0
        for param in self.parameters():
            all_count += numpy.prod(param.shape)
            if param.requires_grad:
                trainable_count += numpy.prod(param.shape)
        return {"trainable": int(trainable_count), "all": int(all_count)}


######################################
########### Roberta Related Setup ####
######################################


######################################
####### Full-Finetuning ##############
######################################
class FullRobertaForSequenceClassification(RobertaForSequenceClassification, TorchModelUtils):
    """
    Copy of RobertaForSequenceClassification from transformers;
    This takes care of choosing mask or cls embedding
    """

    def __init__(self, config):
        super().__init__(config)
        self.mask_id = None

        # we don't consider the dense layer as in the HF code
        # classifier_dropout = (
        #         config.classifier_dropout if config.classifier_dropout is not None else
        #         config.hidden_dropout_prob
        # )
        # self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def set_mask_id(self, x):
        self.mask_id = x

    def set_up_for_training(self, use_mask_embeddings, use_dropout):
        # remove the pooler, as we don't want to use CLS embeddings
        self.use_mask_embeddings = use_mask_embeddings
        if use_mask_embeddings:
            self.roberta.pooler = None

        if not use_dropout:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    # set dropout to 0
                    m.p = 0
        return

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be
            in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is
            computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.use_mask_embeddings:
            sequence_output = outputs.last_hidden_state

            idx = []
            n = input_ids.shape[0]
            for sample in input_ids:
                mask_not_found = True
                for i, v in enumerate(sample):
                    if v == self.mask_id:
                        idx.append(i)
                        mask_not_found = False
                        break
                if mask_not_found:
                    raise Exception(f"Mask token not found in {sample}")
            pooled_output = sequence_output[list(range(n)), idx]
        else:
            sequence_output = outputs[0]
            pooled_output = sequence_output[:, 0, :]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


######################################
####### Linear Clf ###################
######################################
class LinearRobertaForSequenceClassification(FullRobertaForSequenceClassification):
    def set_up_for_training(self, *args, **kwargs):
        super().set_up_for_training(*args, **kwargs)
        self.requires_grad_(False)
        self.classifier.requires_grad_(True)

"""We override roberta's internal modules for SLaSh"""
######################################
####### SLaSh ########################
######################################
class SLaShRobertaOutput(RobertaOutput):
    def __init__(self, config, z_vector=None, weight=None):
        super().__init__(config)
        self.z_vector = z_vector
        self.weight = weight

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # insert just before the layer norm
        bias = (self.z_vector @ self.weight).unsqueeze(dim=1)
        hidden_states = self.LayerNorm(hidden_states + input_tensor + bias)
        return hidden_states


class SLaShRobertaIntermediate(RobertaIntermediate):
    def __init__(self, config, z_vector=None, weight=None):
        super().__init__(config)
        self.z_vector = z_vector
        self.weight = weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bias = (self.z_vector @ self.weight).unsqueeze(dim=1)
        # breakpoint()
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states + bias)
        return hidden_states


class SLaShRobertaAttention(RobertaAttention):
    def __init__(self, config, position_embedding_type=None, z_vector=None, weight=None):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.z_vector = z_vector
        self.weight = weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        outputs = super().forward(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        bias = (self.z_vector @ self.weight).unsqueeze(dim=1)
        new_output = outputs[0] + bias
        outputs = (new_output,) + outputs[1:]
        return outputs


class SLaShRobertaLayer(RobertaLayer):
    def __init__(self, config, z_vector, weight):
        super().__init__(config)
        if config.bias_position == "intermediate":
            self.intermediate = SLaShRobertaIntermediate(config, z_vector, weight)
        if config.bias_position == "output":
            self.output = SLaShRobertaOutput(config, z_vector, weight)
        if config.bias_position == "attention":
            self.attention = SLaShRobertaAttention(config, None, z_vector, weight)


class SLaShRobertaEncoder(RobertaEncoder):
    def __init__(self, config, z_vector, weights):
        super().__init__(config)
        self.config = config
        self.layer = nn.ModuleList(
            [SLaShRobertaLayer(config, z_vector, weight) for weight in weights]
        )
        self.gradient_checkpointing = False


class SLaShRobertaModel(RobertaModel):
    def __init__(self, config, add_pooling_layer=True, z_vector=None, weights=None):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaEmbeddings(config)
        self.encoder = SLaShRobertaEncoder(config, z_vector, weights)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.init_weights()


class SLaShRobertaForSequenceClassification(FullRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        z_size = config.z_size
        if config.bias_position == "intermediate":
            d_size = config.intermediate_size
        else:
            d_size = config.hidden_size
        self.z_vector = nn.Parameter(do_zinit(z_size, mode=config.z_init))
        self.weights = nn.ParameterList(
            [
                nn.Parameter(do_winit(z_size, d_size, mode=config.w_init))
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.roberta = SLaShRobertaModel(
            config, add_pooling_layer=True, z_vector=self.z_vector, weights=self.weights
        )
        self.init_weights()
        self.init_params()

    def init_params(self):
        config = self.config
        z_size = config.z_size
        if config.bias_position == "intermediate":
            d_size = config.intermediate_size
        else:
            d_size = config.hidden_size
        self.z_vector.data = do_zinit(z_size, mode=config.z_init)
        for i in range(len(self.weights)):
            self.weights[i].data = do_winit(z_size, d_size, mode=config.w_init)

    def set_up_for_training(self, use_mask_embeddings, use_dropout):
        super().set_up_for_training(use_mask_embeddings, use_dropout)
        self.requires_grad_(False)
        self.classifier.requires_grad_(True)
        self.z_vector.requires_grad_(True)



"""We override roberta's internal modules for WARP"""
######################################
####### WARP #########################
######################################
class WARPRobertaLayer(RobertaLayer):
    def __init__(self, config, z_vector=None):
        super().__init__(config)
        self.config = config
        self.z = z_vector

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attn_seq_len = attention_mask.shape[3]
        seq_len = hidden_states.shape[1]
        BS = hidden_states.shape[0]

        # only add on the first layer
        if attn_seq_len != seq_len:
            # breakpoint()
            transform = self.z.expand(BS, -1, -1)
            hidden_states = torch.cat([transform, hidden_states], dim=1)
        # breakpoint()
        return super().forward(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )


class WARPRobertaEncoder(RobertaEncoder):
    def __init__(self, config, z_vector=None):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [WARPRobertaLayer(config, z_vector=z_vector) for w in range(config.num_hidden_layers)]
        )


class WARPRobertaModel(RobertaModel):
    def __init__(self, config, add_pooling_layer=True, z_vector=None):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.z = z_vector
        self.insert_seq_len = config.insert_seq_len

        self.encoder = WARPRobertaEncoder(config, z_vector=z_vector)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        assert past_key_values is None

        # fixing attention mask
        BS = attention_mask.shape[0]
        # at this point attention mask is [B, MaxSeqLen], and so we need to add the one for past key values
        attention_mask = torch.cat(
            [torch.ones(BS, self.insert_seq_len, device=attention_mask.device), attention_mask],
            dim=1,
        )
        # there is a bug that causes lengths more than 512 to crash, so we account for this here.
        # if input_ids.shape[1] + self.insert_seq_len >= 512:
        #     input_ids = input_ids[:, :512 - self.insert_seq_len]
        #     attention_mask = attention_mask[:, :512]

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class WARPRobertaForSequenceClassification(FullRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        z_size = config.hidden_size  # z has to be same as embeddings
        self.insert_seq_len = config.insert_seq_len
        z = torch.cat(
            [do_zinit(z_size, mode=config.z_init) for _ in range(self.insert_seq_len)], dim=0
        )
        self.z_vector = nn.Parameter(z)

        self.roberta = WARPRobertaModel(config, add_pooling_layer=True, z_vector=self.z_vector)
        self.init_weights()

    def set_up_for_training(self, use_mask_embeddings, use_dropout):
        super().set_up_for_training(use_mask_embeddings, use_dropout)
        self.requires_grad_(False)
        self.classifier.requires_grad_(True)
        self.z_vector.requires_grad_(True)


"""We override roberta's internal modules for JR-WARP"""
######################################
####### JR WARP ########################
######################################
class JR_WARPRobertaLayer(RobertaLayer):
    def __init__(self, config, z_vector=None, weight=None, add_identity=False):
        super().__init__(config)
        self.config = config
        self.z = z_vector
        self.weight = weight

        self.add_identity = add_identity

        # this is a trick to enable computing sample wise gradients in privacy experiments by creating a dummy layer
        # This is disabled in non-private experiments
        # and is not updated in DP experiments
        if self.add_identity:
            self.id = torch.nn.Linear(config.hidden_size, config.hidden_size)
            self.id.weight.data = torch.zeros_like(self.id.weight.data)
            self.id.bias.data = torch.zeros_like(self.id.bias.data)
            self.id_input = torch.zeros(self.config.hidden_size, device="cuda")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attn_seq_len = attention_mask.shape[3]
        seq_len = hidden_states.shape[1]
        BS = hidden_states.shape[0]

        transform = (self.z @ self.weight).expand(BS, 1, -1)

        if self.add_identity:
            id_input = self.id_input.expand(BS, -1)
            if attn_seq_len == seq_len:
                hidden_states = torch.cat(
                    [
                        transform + hidden_states[:, :1] + self.id(id_input).reshape(BS, 1, -1),
                        hidden_states[:, 1:],
                    ],
                    dim=1,
                )
            else:
                hidden_states = torch.cat(
                    [transform + self.id(id_input).reshape(BS, 1, -1), hidden_states], dim=1
                )
        else:
            if attn_seq_len == seq_len:
                hidden_states = torch.cat(
                    [transform + hidden_states[:, :1], hidden_states[:, 1:]], dim=1
                )
            else:
                hidden_states = torch.cat([transform, hidden_states], dim=1)
        return super().forward(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )


class JR_WARPRobertaEncoder(RobertaEncoder):
    def __init__(self, config, z_vector=None, weights=None, add_identity=False):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [
                JR_WARPRobertaLayer(config, z_vector=z_vector, weight=w, add_identity=add_identity)
                for (_, w) in zip(range(config.num_hidden_layers), weights)
            ]
        )


class JR_WARPRobertaModel(RobertaModel):
    def __init__(
        self, config, add_pooling_layer=True, z_vector=None, weights=None, add_identity=False
    ):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.z = z_vector
        self.weights = weights
        self.insert_seq_len = 1

        self.encoder = JR_WARPRobertaEncoder(
            config, z_vector=z_vector, weights=weights, add_identity=add_identity
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        assert past_key_values is None

        # fixing attention mask
        BS = attention_mask.shape[0]
        # at this point attention mask is [B, MaxSeqLen], and so we need to add the one for past key values
        attention_mask = torch.cat(
            [torch.ones(BS, self.insert_seq_len, device=attention_mask.device), attention_mask],
            dim=1,
        )
        # there is a bug that causes lengths more than 512 to crash, so we account for this here.
        if input_ids.shape[1] + self.insert_seq_len >= 512:
            input_ids = input_ids[:, : 512 - self.insert_seq_len]
            attention_mask = attention_mask[:, :512]

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class JR_WARPRobertaForSequenceClassification(FullRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        z_size = config.z_size
        d_size = config.hidden_size
        self.z_vector = nn.Parameter(do_zinit(z_size, mode=config.z_init))
        self.weights = nn.ParameterList(
            [
                nn.Parameter(do_winit(z_size, d_size, mode=config.w_init))
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.roberta = JR_WARPRobertaModel(
            config,
            add_pooling_layer=True,
            z_vector=self.z_vector,
            weights=self.weights,
            add_identity=getattr(self, "add_identity"),
        )
        self.init_weights()

    def set_up_for_training(self, use_mask_embeddings, use_dropout):
        super().set_up_for_training(use_mask_embeddings, use_dropout)
        self.requires_grad_(False)
        self.z_vector.requires_grad_(True)
        self.classifier.requires_grad_(True)


"""Code adapted from BitFit's official repository"""
##############################################
####### BitFit ###############################
##############################################
class BitFitRobertaForSequenceClassification(RobertaForSequenceClassification, TorchModelUtils):
    BIAS_TERMS_DICT = {
        "intermediate": "intermediate.dense.bias",
        "key": "attention.self.key.bias",
        "query": "attention.self.query.bias",
        "value": "attention.self.value.bias",
        "output": "output.dense.bias",
        "output_layernorm": "output.LayerNorm.bias",
        "attention_layernorm": "attention.output.LayerNorm.bias",
        "all": "bias",
    }

    def set_up_for_training(self, use_mask_embeddings, use_dropout):
        # we are only interested in "all"
        trainable_components = [self.BIAS_TERMS_DICT["all"]]
        for param in self.parameters():
            param.requires_grad = False
        # if trainable_components:
        #     trainable_components = trainable_components + ['pooler.dense.bias']
        trainable_components = trainable_components + ["classifier.out_proj"]
        for name, param in self.named_parameters():
            for component in trainable_components:
                if component in name:
                    param.requires_grad = True
                    break

        if not use_dropout:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    # set dropout to 0
                    m.p = 0

    def set_mask_id(self, *args, **kwargs):
        return


###########################################################
######### SLaSh for Private Training ######################
###########################################################
""" 
We had to use a specific trick for private training to simplify per sample
 gradient computation using torch's extended weights module. It was
 experimental at the time of writing and had problem computing gradient of the 
 z parameter. So we computed gradient of output activation's bias and b
 backpropogate manually from there. This module sets gradient of output 
 activation's bias to True for that reason. Manual computation of gradients is
 handled in `src/utils/opacus_utils.py`. 

 I still don't exactly understand how the extended weights module works, but 
 this should compute correct gradients as we verified the outputs of this with 
 gradient from batch size 1 training. 
"""


class PrivateSLaShRobertaForSequenceClassification(SLaShRobertaForSequenceClassification):
    def forward(
        self,
        x,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return super(PrivateSLaShRobertaForSequenceClassification, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def set_up_for_training(self, use_mask_embeddings, use_dropout):
        super(PrivateSLaShRobertaForSequenceClassification, self).set_up_for_training(
            use_mask_embeddings, use_dropout
        )
        [layer.output.dense.bias.requires_grad_(True) for layer in self.roberta.encoder.layer]


###########################################################
######### JR-WARP for Private Training#####################
###########################################################
class PrivateJR_WARPRobertaForSequenceClassification(JR_WARPRobertaForSequenceClassification):
    def __init__(self, config):
        self.add_identity = True
        super().__init__(config)

    def forward(
        self,
        x,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return super(PrivateJR_WARPRobertaForSequenceClassification, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def set_up_for_training(self, use_mask_embeddings, use_dropout):
        super(PrivateJR_WARPRobertaForSequenceClassification, self).set_up_for_training(
            use_mask_embeddings, use_dropout
        )

        [layer.id.bias.requires_grad_(True) for layer in self.roberta.encoder.layer]


##########################################
####### Token Classification Models ######
##########################################


##########################################
###### Full Finetuning ###################
##########################################
class FullBertForTokenClassification(BertForTokenClassification, TorchModelUtils):
    def __init__(self, config):
        super().__init__(config)
        self.mask_id = None

    def set_mask_id(self, x):
        self.mask_id = x

    def set_up_for_training(self, use_dropout):
        if not use_dropout:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    # set dropout to 0
                    m.p = 0
        return


##########################################
###### Linear Clf ########################
##########################################
class LinearBertForTokenClassification(FullBertForTokenClassification):
    def set_up_for_training(self, *args, **kwargs):
        super().set_up_for_training(*args, **kwargs)
        self.requires_grad_(False)
        self.classifier.requires_grad_(True)


##########################################
###### SLaSh #### ########################
##########################################
class SLaSHBertOutput(BertOutput):
    def __init__(self, config, z_vector=None, weight=None):
        super().__init__(config)
        self.z_vector = z_vector
        self.weight = weight

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # insert just before the layer norm
        bias = (self.z_vector @ self.weight).unsqueeze(dim=1)
        hidden_states = self.LayerNorm(hidden_states + input_tensor + bias)
        return hidden_states


class SLaSHBertLayer(BertLayer):
    def __init__(self, config, z_vector, weight):
        super().__init__(config)
        self.output = SLaSHBertOutput(config, z_vector, weight)


class SLaSHBertEncoder(BertEncoder):
    def __init__(self, config, z_vector, weights):
        super().__init__(config)
        self.layer = nn.ModuleList([SLaSHBertLayer(config, z_vector, weight) for weight in weights])


class SLaSHBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True, z_vector=None, weights=None):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.encoder = SLaSHBertEncoder(config, z_vector, weights)
        self.post_init()


class SLaSHBertForTokenClassification(FullBertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        z_size = config.z_size
        d_size = config.hidden_size
        self.z_vector = nn.Parameter(do_zinit(z_size, mode=config.z_init))
        self.weights = nn.ParameterList(
            [
                nn.Parameter(do_winit(z_size, d_size, mode=config.w_init))
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.bert = SLaSHBertModel(
            config, add_pooling_layer=True, z_vector=self.z_vector, weights=self.weights
        )
        self.init_weights()
        self.init_params()

    def init_params(self):
        config = self.config
        z_size = config.z_size
        d_size = config.hidden_size
        self.z_vector.data = do_zinit(z_size, mode=config.z_init)
        for i in range(len(self.weights)):
            self.weights[i].data = do_winit(z_size, d_size, mode=config.w_init)

    def set_up_for_training(self, use_dropout):
        super().set_up_for_training(use_dropout)
        self.requires_grad_(False)
        self.classifier.requires_grad_(True)
        self.z_vector.requires_grad_(True)


##########################################
###### BitFit ############################
##########################################
"""Adapted from BitFit's official implementation"""


class BitFitBertForTokenClassification(BertForTokenClassification, TorchModelUtils):
    BIAS_TERMS_DICT = {
        "intermediate": "intermediate.dense.bias",
        "key": "attention.self.key.bias",
        "query": "attention.self.query.bias",
        "value": "attention.self.value.bias",
        "output": "output.dense.bias",
        "output_layernorm": "output.LayerNorm.bias",
        "attention_layernorm": "attention.output.LayerNorm.bias",
        "all": "bias",
    }

    def set_up_for_training(self, use_dropout):
        # we are only interested in "all"
        trainable_components = [self.BIAS_TERMS_DICT["all"]]
        for param in self.parameters():
            param.requires_grad = False
        trainable_components = trainable_components + ["classifier"]
        for name, param in self.named_parameters():
            for component in trainable_components:
                if component in name:
                    param.requires_grad = True
                    break

        if not use_dropout:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    # set dropout to 0
                    m.p = 0

    def set_mask_id(self, *args, **kwargs):
        return


##########################################
###### WARP ##############################
##########################################
class WARPBertLayer(BertLayer):
    def __init__(self, config, z_vector=None):
        super().__init__(config)
        self.config = config
        self.z = z_vector

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attn_seq_len = attention_mask.shape[3]
        seq_len = hidden_states.shape[1]
        BS = hidden_states.shape[0]

        # only add on the first layer
        if attn_seq_len != seq_len:
            # breakpoint()
            transform = self.z.expand(BS, -1, -1)
            hidden_states = torch.cat([transform, hidden_states], dim=1)
        # breakpoint()
        return super().forward(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )


class WARPBertEncoder(BertEncoder):
    def __init__(self, config, z_vector=None):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [WARPBertLayer(config, z_vector=z_vector) for w in range(config.num_hidden_layers)]
        )


class WARPBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True, z_vector=None):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.z = z_vector
        self.insert_seq_len = config.insert_seq_len

        self.encoder = WARPBertEncoder(config, z_vector=z_vector)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        assert past_key_values is None

        # fixing attention mask
        BS = attention_mask.shape[0]
        # at this point attention mask is [B, MaxSeqLen], and so we need to add the one for past key values
        attention_mask = torch.cat(
            [torch.ones(BS, self.insert_seq_len, device=attention_mask.device), attention_mask],
            dim=1,
        )
        # there is a bug that causes lengths more than 512 to crash, so we account for this here.
        # if input_ids.shape[1] + self.insert_seq_len >= 512:
        #     input_ids = input_ids[:, :512 - self.insert_seq_len]
        #     attention_mask = attention_mask[:, :512]

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class WARPBertForTokenClassification(FullBertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        z_size = config.hidden_size  # z has to be same as embeddings
        self.insert_seq_len = config.insert_seq_len
        z = torch.cat(
            [do_zinit(z_size, mode=config.z_init) for _ in range(self.insert_seq_len)], dim=0
        )
        self.z_vector = nn.Parameter(z)

        self.bert = WARPBertModel(config, add_pooling_layer=True, z_vector=self.z_vector)
        self.init_weights()

    def set_up_for_training(self, use_dropout):
        super().set_up_for_training(use_dropout)
        self.requires_grad_(False)
        self.classifier.requires_grad_(True)
        self.z_vector.requires_grad_(True)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # breakpoint()
        logits = (logits[:, self.insert_seq_len :, :]).contiguous()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


##########################################
###### JR WARP ##############################
##########################################
class JR_WARPBertLayer(BertLayer):
    def __init__(self, config, z_vector=None, weight=None):
        super().__init__(config)
        self.config = config
        self.z = z_vector
        self.weight = weight

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attn_seq_len = attention_mask.shape[3]
        seq_len = hidden_states.shape[1]
        BS = hidden_states.shape[0]

        transform = (self.z @ self.weight).expand(BS, 1, -1)

        if attn_seq_len == seq_len:
            hidden_states = torch.cat(
                [transform + hidden_states[:, :1], hidden_states[:, 1:]], dim=1
            )
        else:
            hidden_states = torch.cat([transform, hidden_states], dim=1)
        return super().forward(
            hidden_states,
            attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )


class JR_WARPBertEncoder(BertEncoder):
    def __init__(self, config, z_vector=None, weights=None):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [
                JR_WARPBertLayer(config, z_vector=z_vector, weight=w)
                for (_, w) in zip(range(config.num_hidden_layers), weights)
            ]
        )


class JR_WARPBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True, z_vector=None, weights=None):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
        self.z = z_vector
        self.weights = weights
        self.insert_seq_len = 1
        self.encoder = JR_WARPBertEncoder(config, z_vector=z_vector, weights=weights)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        assert past_key_values is None

        # fixing attention mask
        BS = attention_mask.shape[0]
        # at this point attention mask is [B, MaxSeqLen], and so we need to add the one for past key values
        attention_mask = torch.cat(
            [torch.ones(BS, self.insert_seq_len, device=attention_mask.device), attention_mask],
            dim=1,
        )
        # there is a bug that causes lengths more than 512 to crash, so we account for this here.
        # if input_ids.shape[1] + self.insert_seq_len >= 512:
        #     input_ids = input_ids[:, :512 - self.insert_seq_len]
        #     attention_mask = attention_mask[:, :512]

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class JR_WARPBertForTokenClassification(FullBertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        z_size = config.z_size
        d_size = config.hidden_size
        self.z_vector = nn.Parameter(do_zinit(z_size, mode=config.z_init))
        self.weights = nn.ParameterList(
            [
                nn.Parameter(do_winit(z_size, d_size, mode=config.w_init))
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.insert_seq_len = 1
        self.bert = JR_WARPBertModel(
            config, add_pooling_layer=True, z_vector=self.z_vector, weights=self.weights
        )
        self.init_weights()

    def set_up_for_training(self, use_dropout):
        super().set_up_for_training(use_dropout)
        self.requires_grad_(False)
        self.classifier.requires_grad_(True)
        self.z_vector.requires_grad_(True)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # breakpoint()
        logits = (logits[:, self.insert_seq_len :, :]).contiguous()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
