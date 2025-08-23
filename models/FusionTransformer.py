import torch
import torch.nn as nn

class FusionTransformer(nn.Module):
    def __init__(self, d_model=768, nhead=12, num_layers=2, d_ff=3072):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, img_feats, txt_feats):
        fusion_input = torch.cat([img_feats, txt_feats], dim=1)   # [B, Q+L, d_model]          
        return self.transformer(fusion_input)                      # [B, Q+L, d_model]
    
if __name__ == "__main__":
    # Example usage
    model = FusionTransformer()
    img_feats = torch.randn(2, 10, 768)  # Batch size 2, 10 image features
    txt_feats = torch.randn(2, 5, 768)   # Batch size 2, 5 text features
    output = model(img_feats, txt_feats)
    print(output.shape)  # Should be [2, 15, 768] since we concatenate along the feature dimension
