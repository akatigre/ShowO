import os
import types
import torch
from dataclasses import dataclass

from typing import Optional, Tuple, Union, List
from models.phi import PhiForCausalLM, PhiModel, PhiDecoderLayer, PhiSdpaAttention
from transformers.modeling_outputs import ModelOutput, CausalLMOutput
from transformers.cache_utils import Cache, DynamicCache
from models.phi import apply_rotary_pos_emb, repeat_kv

import logging
from rich.logging import RichHandler
from rich.theme import Theme
from rich.console import Console
FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(
        console=Console(theme=Theme({"logging.level.success": "green"}))
    )]
)

logger = logging.getLogger("rich")

@dataclass
class CausalLMOutputWithPast(CausalLMOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_value_pag: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_value_pag: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

def change_showo_forward(model: PhiForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_key_values_pag: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        enable_pag: Optional[bool] = None,
        enable_cfg: Optional[bool] = None,
        prefix_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            past_key_values_pag=past_key_values_pag,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            enable_pag=enable_pag,
            enable_cfg=enable_cfg,
            prefix_len=prefix_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    model.forward = types.MethodType(forward, model)
    return model

def change_phi_forward(model: PhiModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_key_values_pag: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        enable_pag: Optional[bool] = None,
        enable_cfg: Optional[bool] = None,
        prefix_len: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_key_values_pag = DynamicCache.from_legacy_cache(past_key_values_pag)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = self.embed_dropout(inputs_embeds)
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        next_decoder_cache_pag = None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    past_key_values_pag,
                    output_attentions,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    past_key_value_pag=past_key_values_pag,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    enable_pag=enable_pag,
                    enable_cfg=enable_cfg,
                    prefix_len=prefix_len,
                )

            hidden_states = layer_outputs[0]

            if use_cache and not enable_pag:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            elif use_cache and enable_pag:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
                next_decoder_cache_pag = layer_outputs[3 if output_attentions else 2]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        next_cache_pag = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
            
            if enable_pag: next_cache_pag = next_decoder_cache_pag.to_legacy_cache() if use_legacy_cache else next_decoder_cache_pag
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            past_key_value_pag=next_cache_pag,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    model.forward = types.MethodType(forward, model)
    return model

def change_phi_decoder_layer_forward(model: PhiDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        past_key_value_pag: Optional[Tuple[torch.Tensor]] = None,
        enable_pag: Optional[bool] = False,
        enable_cfg: Optional[bool] = False,
        prefix_len: Optional[int] = 0,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            past_key_value_pag=past_key_value_pag,
            output_attentions=output_attentions,
            use_cache=use_cache,
            enable_cfg=enable_cfg,
            enable_pag=enable_pag,
            prefix_len=prefix_len,
        )
        if enable_pag:
            attn_outputs, self_attn_weights, present_key_value, present_key_value_pag = outputs
        else:
            attn_outputs, self_attn_weights, present_key_value = outputs
            present_key_value_pag = None
        attn_outputs = self.resid_dropout(attn_outputs)

        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        hidden_states = attn_outputs + feed_forward_hidden_states + residual
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache and present_key_value is not None:
            outputs += (present_key_value,)
        if use_cache and present_key_value_pag is not None:
            outputs += (present_key_value_pag,)

        return outputs
    model.forward = types.MethodType(forward, model)
    return model

def cfg_pag_forward(model: PhiSdpaAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        past_key_value_pag: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        enable_pag: Optional[bool] = False,
        enable_cfg: Optional[bool] = False,
        prefix_len: Optional[int] = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "PhiModel is using PhiSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not "
                "support `output_attentions=True`. Falling back to the manual attention implementation, but specifying "
                "the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can "
                'be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        # cond, cfg, pag | cond, cfg | cond, pag | cond
        if enable_cfg and enable_pag:
            hidden_states_cond, hidden_states_uncond, hidden_states_pag = hidden_states.chunk(3)
            attention_mask_cond, attention_mask_uncond, attention_mask_pag = attention_mask.chunk(3)
            if self.pag_layer:
                hidden_states = torch.cat([hidden_states_cond, hidden_states_uncond], dim=0)
                attention_mask = torch.cat([attention_mask_cond, attention_mask_uncond], dim=0)
            else:
                pass
        elif enable_cfg and not enable_pag:
            hidden_states_cond, hidden_states_uncond = hidden_states.chunk(2)
            attention_mask_cond, attention_mask_uncond = attention_mask.chunk(2)
            hidden_states, attention_mask = hidden_states_cond, attention_mask_cond
            hidden_states_pag, attention_mask_pag = None
        elif not enable_cfg and enable_pag:
            hidden_states_cond, hidden_states_pag = hidden_states.chunk(2)
            attention_mask_cond, attention_mask_pag = attention_mask.chunk(2)
            hidden_states_uncond = None
            if self.pag_layer:
                hidden_states, attention_mask = hidden_states_cond, attention_mask_cond
            else:
                pass
            
        else:
            hidden_states_cond, hidden_states_uncond, hidden_states_pag = None, None, None
            attention_mask_cond, attention_mask_uncond, attention_mask_pag = None, None, None
            pass 

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.qk_layernorm:
            query_states = self.q_layernorm(query_states)
            key_states = self.k_layernorm(key_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Partial rotary embedding
        query_rot, query_pass = (
            query_states[..., : self.rotary_emb.dim],
            query_states[..., self.rotary_emb.dim :],
        )
        key_rot, key_pass = (
            key_states[..., : self.rotary_emb.dim],
            key_states[..., self.rotary_emb.dim :],
        )
        # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

        # [batch_size, seq_length, num_heads, head_dim]
        query_states = torch.cat((query_rot, query_pass), dim=-1)
        key_states = torch.cat((key_rot, key_pass), dim=-1)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "partial_rotation_size": self.rotary_emb.dim}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        if self.require_contiguous_qkv and query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if self.is_causal and attention_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if enable_pag:
            bsz_ptb, q_len, _ = hidden_states_pag.shape
            query_states_ptb = self.q_proj(hidden_states_pag)
            key_states_ptb = self.k_proj(hidden_states_pag)
            value_states_ptb = self.v_proj(hidden_states_pag)

            query_states_ptb = query_states_ptb.view(bsz_ptb, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states_ptb = key_states_ptb.view(bsz_ptb, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states_ptb = value_states_ptb.view(bsz_ptb, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            cos, sin = self.rotary_emb(value_states_ptb, seq_len=kv_seq_len)
            # Partial rotary embedding
            query_rot, query_pass = (
                query_states_ptb[..., : self.rotary_emb.dim],
                query_states_ptb[..., self.rotary_emb.dim :],
            )
            key_rot, key_pass = (
                key_states_ptb[..., : self.rotary_emb.dim],
                key_states_ptb[..., self.rotary_emb.dim :],
            )
            # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
            query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

            # [batch_size, seq_length, num_heads, head_dim]
            query_states_ptb = torch.cat((query_rot, query_pass), dim=-1)
            key_states_ptb = torch.cat((key_rot, key_pass), dim=-1)
    
            if past_key_value_pag is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos,  "partial_rotation_size": self.rotary_emb.dim}
                key_states_ptb, value_states_ptb = past_key_value_pag.update(key_states_ptb, value_states_ptb, self.layer_idx, cache_kwargs)
        
        ########### PAG path ############
        if enable_pag and self.pag_layer:    
                
            key_states_ptb = repeat_kv(key_states_ptb, self.num_key_value_groups)
            value_states_ptb = repeat_kv(value_states_ptb, self.num_key_value_groups)
            if self.require_contiguous_qkv and query_states.device.type == "cuda" and attention_mask is not None:
                query_states_ptb = query_states_ptb.contiguous()
                key_states_ptb = key_states_ptb.contiguous()
                value_states_ptb = value_states_ptb.contiguous()
            attention_mask_pag[ ... , prefix_len : -1] = float("-inf")

            # expand the mask to match the attention weights shape
            attn_output_ptb = torch.nn.functional.scaled_dot_product_attention(
                query_states_ptb,
                key_states_ptb,
                value_states_ptb,
                attn_mask=attention_mask_pag,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            attn_output_ptb = attn_output_ptb.transpose(1, 2).contiguous()
            attn_output_ptb = attn_output_ptb.view(bsz_ptb, q_len, -1)
            attn_output = torch.cat([attn_output, attn_output_ptb], dim=0)
            attn_output = self.dense(attn_output)
        else:
            attn_output = self.dense(attn_output)

        if enable_pag:
            outputs = [attn_output, None, past_key_value, past_key_value_pag]
        else:
            outputs = [attn_output, None, past_key_value]
        return outputs
    model.forward = types.MethodType(forward, model)
    return model