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

    def forward(self, *args):
        # args = (feat1, feat2, ..., featN), each [B, L_i, d_model]
        fusion_input = torch.cat(args, dim=1)   # [B, sum(L_i), d_model]          
        return self.transformer(fusion_input)                      # [B, sum(L_i), d_model]
    
if __name__ == "__main__":
    # Example usage
    model = FusionTransformer()
    img_feats = torch.randn(2, 10, 768)  # Batch size 2, 10 image features
    txt_feats = torch.randn(2, 5, 768)   # Batch size 2, 5 text features
    img_feats_224 = torch.randn(2, 15, 768)  # Batch size 2, 15 additional image features
    output = model(img_feats,img_feats_224, txt_feats)
    print(output.shape)  # Should be [2, 15, 768] since we concatenate along the feature dimension
