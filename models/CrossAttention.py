import torch
import torch.nn as nn

class CrossAttentionTransformer(nn.Module):
    def __init__(self, d_model=768, nhead=12, num_layers=2, d_ff=3072):
        super().__init__()
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, d_ff)
            for _ in range(num_layers)
        ])
        
    def forward(self, image_feat, text_feat):
        """
        Returns combined features for language model input
        """
        img_out = image_feat
        txt_out = text_feat
        
        for layer in self.cross_attention_layers:
            img_out, txt_out = layer(img_out, txt_out)
        
        # Concatenate for language model
        return torch.cat([img_out, txt_out], dim=1)  # [B, L_img + L_text, d_model]


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        
        # Shared cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        
        # Shared components
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, image_feat, text_feat):
        # Image attends to text
        img_cross_out, _ = self.cross_attn(image_feat, text_feat, text_feat)
        image_feat = self.norm1(image_feat + img_cross_out)
        
        # Text attends to image  
        text_cross_out, _ = self.cross_attn(text_feat, image_feat, image_feat)
        text_feat = self.norm1(text_feat + text_cross_out)
        
        # Feed-forward
        img_ffn_out = self.ffn(image_feat)
        image_feat = self.norm2(image_feat + img_ffn_out)
        
        text_ffn_out = self.ffn(text_feat)
        text_feat = self.norm2(text_feat + text_ffn_out)
        
        return image_feat, text_feat
    
if __name__ == "__main__":
    # Example usage
    model = CrossAttentionTransformer(num_layers=1)
    img_feats = torch.randn(2, 32, 768)  # Batch size 2, 10 image features
    txt_feats = torch.randn(2, 1, 768)   # Batch size 2, 5 text features
    output = model(img_feats, txt_feats)
    print(output.shape)  # Should be [2, 33, 768] since we concatenate along the feature dimension