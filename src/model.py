import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create matrix of shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # shape: (1, max_len, d_model)
        
        # Register as a buffer so it saves with the model but doesn't get optimized
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model) because we use batch_first=True
        # Add positional encoding up to the current sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens):
        # Multiply by sqrt(d_model) standard for transformers
        return self.embedding(tokens) * math.sqrt(self.d_model)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, 
                 num_encoder_layers, 
                 num_decoder_layers, 
                 d_model, 
                 nhead, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 dim_feedforward, 
                 dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Ensure shapes are uniformly (batch, seq, feature)
        )
        
        self.src_tok_emb = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        """
        src: Tensor of shape (batch_size, src_seq_len)
        tgt: Tensor of shape (batch_size, tgt_seq_len)
        *mask: Matrices protecting model from generating invalid attenion over padding/future chunks.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.generator(outs)

    def encode(self, src, src_mask):
        # Used independently during Inference (Beam Search)
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), 
            src_mask
        )

    def decode(self, tgt, memory, tgt_mask):
         # Used independently during Inference (Beam Search)
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), 
            memory, 
            tgt_mask
        )

def generate_square_subsequent_mask(sz):
    """
    Generates a causal mask (upper triangular with -inf).
    Positions predicting future tokens are masked out.
    """
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask

def create_mask(src, tgt, pad_idx):
    """
    Calculates dynamic attention masks based on sequence sizes and padding logic.
    """
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    # Causal Lookahead Mask preventing future peeking in the decoder
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(src.device)
    # Complete un-mask for the source (entire sentence is visible logically)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

    # Padding Masks identifying the explicit locations of <pad> (True = ignore)
    src_padding_mask = (src == pad_idx).to(src.device)
    tgt_padding_mask = (tgt == pad_idx).to(src.device)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

if __name__ == "__main__":
    from config import *
    print("Initiating PyTorch Transformer Plumbing Test...")
    model = Seq2SeqTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_model=D_MODEL,
        nhead=N_HEADS,
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Instantiated Model - Total Trainable Parameters: {total_params:,}")
    
    # Generate mock batches (batch=2, length=5)
    test_src = torch.tensor([[2, 45, 12, 0, 0], [2, 10, 89, 78, 0]])
    test_tgt = torch.tensor([[2, 88, 12, 0, 0], [2, 33, 44, 99, 11]])
    
    src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(test_src, test_tgt, PAD_IDX)
    
    print("Executing forward pass test...")
    out = model(test_src, test_tgt, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask, src_pad_mask)
    
    expected_shape = [2, 5, VOCAB_SIZE] # [batch, seq_len, vocab_size]
    print(f"Output Matrix Shape: {list(out.shape)}")
    
    if list(out.shape) == expected_shape:
        print("Model plumbing test passed successfully! PyTorch architecture confirmed.")
    else:
        print("Model generated incorrect shape dimensions.")
