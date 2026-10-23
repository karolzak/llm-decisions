"""Small training experiments on decision readouts (Part 3 of the replication study).

    lora    minimal LoRA adapters for the decoder layers (no extra dependency)
    corpus  multi-task decision corpus from public datasets (Hugging Face dataset viewer API)
    train   trains the letter readout of `edb.local.chat_readout` with CE or Brier loss,
            with or without option shuffling
"""
