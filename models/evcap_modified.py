import logging
import torch
from torch.cuda.amp import autocast as autocast
from models.blip2 import Blip2Base, disabled_train
import faiss

class EVCap(Blip2Base):
    
    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        low_resource=False,
    ):
        super().__init__()

        self.low_resource = low_resource

        ##### Image 
        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = True
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()
        batch_size, nums, dims = query_features.shape
        query_features = query_features.view(-1,dims)   

        query_features_cpu = query_features.detach().cpu().numpy()
        faiss.normalize_L2(query_features_cpu)
        top_k_similarities, top_k_indices = feat_index.search(query_features_cpu, top_k)

        top_k_indices = torch.tensor(top_k_indices).to(device = query_features.device)
        top_k_similarities = torch.tensor(top_k_similarities).to(device = query_features.device)
        top_k_similarities = top_k_similarities.view(batch_size, -1)

        indices = top_k_indices.view(batch_size, -1)

        re_txt_list_all = []    
        for batch_i in range(batch_size):
            indices_list = indices[batch_i]
            re_txt_batch_list = []
            for i in indices_list: 
                re_txt_batch_list.append(image_id[i])
            re_txt_list_all.append(re_txt_batch_list)
         
        sorted_batched_ret = []
        sorted_batched_ret_similarity_score = []
        for listA, listB in zip(top_k_similarities, re_txt_list_all):
            sorted_listA, indices = listA.sort(descending=True)
            sorted_listB = [self.pre_name(listB[idx]) for idx in indices]
            sorted_listA = sorted_listA[:sub_top_k] ##
            sorted_listB = sorted_listB[:sub_top_k]
            sorted_batched_ret_similarity_score.append(sorted_listA) ##
            sorted_batched_ret.append(sorted_listB)
        return sorted_batched_ret, sorted_batched_ret_similarity_score

    def just_encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")
            
        with self.maybe_autocast():
            
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs_img = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            query_output_img = query_outputs_img.last_hidden_state
            
        return query_output_img
