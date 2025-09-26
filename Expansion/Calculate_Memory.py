'''
 Calculate a LLM agent's memory usage.
 1. Causal_LM_param():  calculate the memory usage of a causal language model. Need specific model architecture.
 2. Pretrained_LM_param(): Need the model.config can be called.
 3. Calculate_Memory():  Calculate the total inference memory usage.
'''

def Causal_LM_param(
        vocab_size:int,
        hidden_size:int,
        num_layers:int,
        num_attention_heads:int,
        intermediate_size:int,
):

    ''' Calculate the amount of parameters of a transformer-based causal language model.'''

    # Embedding layer
    embedding_params = vocab_size * hidden_size

    # Transformer layers (per layer)
    attn_params = 4 * hidden_size * hidden_size   # Q, K, V, O projections   + bias(if needed)

    # Forward pass
    ffn_params = 2 * hidden_size * intermediate_size # Two linear layers in the feed-forward network + bias(if needed)

    # LayerNorm parameters
    layernorm_params = 2 * hidden_size  # gamma and beta parameters

    # Total parameters per layer
    params_per_layer = attn_params + ffn_params + layernorm_params

    # Total parameters for all layers
    total_transformer_params = num_layers * params_per_layer + embedding_params

    return {
        'total_params': total_transformer_params,
        'config': {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_attention_heads': num_attention_heads
        }
    }

def Pretrained_LM_param(model) -> int:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model)

    # get model parameters
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads

    total_params = Causal_LM_param(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=hidden_size,
    )

    return {
        'total_params': total_params,
        'config': {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_attention_heads': num_attention_heads
        }
    }


def Calculate_Memory(
        params:dict,
        precision: str = 'fp16',
        batch_size: int = 1,
        seq_length: int = 2048,
        include_kv_cache: bool = True,
):

    # Bytes-precision mapping
    precision_map = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5
    }

    if precision not in precision_map:
        raise ValueError(f"Unsupported precision: {precision}. Choose from {list(precision_map.keys())}")

    # Memory of the weights in model
    weight_memory = params['total_params'] * precision_map[precision] / (1024 ** 3)  # in GB

    # Memory of the activations during inference
    activation_memory = batch_size * seq_length * params['hidden_size'] * precision_map[precision] / (1024 ** 3)  # in GB

    # Memory of the key-value cache (if applicable)
    if include_kv_cache:
        kv_cache_memory = batch_size * seq_length * params['hidden_size'] * 2 * params['num_layers'] * precision_map[precision] / (1024 ** 3)  # in GB
    else:
        kv_cache_memory = 0

    # Total memory usage
    total_memory = weight_memory + activation_memory + kv_cache_memory

    return {
        'weight_memory_GB': weight_memory,
        'activation_memory_GB': activation_memory,
        'kv_cache_memory_GB': kv_cache_memory,
        'total_memory_GB': total_memory
    }


