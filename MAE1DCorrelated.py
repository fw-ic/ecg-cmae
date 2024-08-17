# Author: Nabil Ibtehaz (https://github.com/nibtehaz)
# Insprired from https://github.com/facebookresearch/MAE

from functools import partial

import torch
import torch.nn as nn
from util.pos_embed import get_1d_sincos_pos_embed
from typing import Callable, List, Optional, Tuple, Union
from timm.models.vision_transformer import Block
from swin_utils import Swin1DBlock


def cosine_distance(x1,x2):
    return 1 - torch.nn.functional.cosine_similarity(x1,x2)



class PatchEmbed1D(nn.Module):
    """ 1D signal to window Embedding
    """

    def __init__(
            self,
            signal_len: Optional[int] = 2500,
            window_len: int = 100,
            in_chans: int = 1,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__()
        self.window_len = window_len
        
        self.signal_len = signal_len        
        self.num_patches = self.signal_len // self.window_len

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=window_len, stride=window_len, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
                
        x = self.proj(x)                
        x = self.norm(x)
        x = x.transpose(1,2)
        
        return x


class Encoder(nn.Module):

    def __init__(self,sig_len=2500, window_len=100, in_chans=1,
                 embed_dim=1024, depth=8, num_heads=8, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()


        self.patch_embed = PatchEmbed1D(sig_len, window_len, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding


        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, x, mask_ratio, ids_shuffle, ids_restore, ids_keep):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :] 

        
        # masking: length -> length * mask_ratio
        x, mask = self.perform_masking(x, mask_ratio, ids_restore, ids_keep)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask

    def perform_masking(self, x, mask_ratio, ids_restore, ids_keep):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask


class Correlator(nn.Module):

    def __init__(self,channel_id,embed_dim,embed_squeeze):

        super().__init__()

        self.channel_id = channel_id

        self.corrlerators = torch.nn.ModuleDict()

        for i in range(12):
            if i!=self.channel_id:
                self.corrlerators[f'{i}'] = torch.nn.Sequential(
                                                                    torch.nn.Linear(embed_dim, int(embed_dim*embed_squeeze)),
                                                                    torch.nn.GELU(),
                                                                    torch.nn.Dropout(0.1),
                                                                    torch.nn.Linear(int(embed_dim*embed_squeeze),1)
                                                                )        

    def forward(self, x1, x2, chnl_id):
        
        return self.corrlerators[chnl_id](x1*x2)

class Decoder(nn.Module):

    def __init__(self,in_chans,window_len,num_patches,embed_dim=1024,decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        self.num_patches = num_patches

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, window_len * in_chans, bias=True) # decoder to patch

        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        
        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        
        torch.nn.init.normal_(self.mask_token, std=.02)


        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        fin_x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(fin_x)

        # remove cls token
        fin_x = fin_x[:, 1:, :]
        x = x[:, 1:, :]


        ## handle nan
        ####x = torch.where(x.isnan(),0,x)

        return x, fin_x

class DecoderStrong(nn.Module):

    def __init__(self,decoder_embed_dim=512):
        super().__init__()

        drop_rate = [0,0.02,0.04,0.06,0.08,0.1]

        self.attn1 = Swin1DBlock(dim=decoder_embed_dim,num_heads=8,window_size=5,drop_path=drop_rate[0])
        self.up1 = torch.nn.ConvTranspose1d(decoder_embed_dim,decoder_embed_dim//2,kernel_size=2,stride=2)
        self.attn2 = Swin1DBlock(dim=decoder_embed_dim//2,num_heads=8,window_size=10,drop_path=drop_rate[1])
        self.up2 = torch.nn.ConvTranspose1d(decoder_embed_dim//2,decoder_embed_dim//4,kernel_size=2,stride=2)
        self.attn3 = Swin1DBlock(dim=decoder_embed_dim//4,num_heads=8,window_size=20,drop_path=drop_rate[2])
        self.up3 = torch.nn.ConvTranspose1d(decoder_embed_dim//4,decoder_embed_dim//8,kernel_size=2,stride=2)
        self.attn4 = Swin1DBlock(dim=decoder_embed_dim//8,num_heads=8,window_size=40,drop_path=drop_rate[3])
        self.up4 = torch.nn.ConvTranspose1d(decoder_embed_dim//8,decoder_embed_dim//16,kernel_size=2,stride=2)
        self.attn5 = Swin1DBlock(dim=decoder_embed_dim//16,num_heads=8,window_size=80,drop_path=drop_rate[4])
        self.up5 = torch.nn.ConvTranspose1d(decoder_embed_dim//16,decoder_embed_dim//32,kernel_size=2,stride=2)
        self.attn6 = Swin1DBlock(dim=decoder_embed_dim//32,num_heads=8,window_size=160,drop_path=drop_rate[5])
        self.up6 = torch.nn.ConvTranspose1d(decoder_embed_dim//32,decoder_embed_dim//32,kernel_size=2,stride=2)
        self.out = torch.nn.Conv1d(decoder_embed_dim//32,1,kernel_size=5,padding=2)

    def forward(self, x):
        
        x1 = self.attn1(x)
        x1 = self.up1(x1.permute(0,2,1)).permute(0,2,1)
        x2 = self.attn2(x1)
        x2 = self.up2(x2.permute(0,2,1)).permute(0,2,1)
        x3 = self.attn3(x2)
        x3 = self.up3(x3.permute(0,2,1)).permute(0,2,1)
        x4 = self.attn4(x3)
        x4 = self.up4(x4.permute(0,2,1)).permute(0,2,1)
        x5 = self.attn5(x4)
        x5 = self.up5(x5.permute(0,2,1)).permute(0,2,1)
        x6 = self.attn6(x5)
        x6 = self.up6(x6.permute(0,2,1)).permute(0,2,1)
        x7 = self.out(x6.permute(0,2,1))

        return x7


class MaskedAutoencoderViT1DCorrelated(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, channel_id, sig_len=2500, window_len=100, in_chans=1,
                 embed_dim=1024, embed_squeeze=0.5, depth=8, num_heads=8,
                 decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Correlators

        print(f"Initializing MAE {channel_id+1}")

        self.channel_id = channel_id

        self.window_len = window_len
        self.num_patches = sig_len//window_len

        self.encoder = Encoder(sig_len, window_len, in_chans, embed_dim, depth, num_heads, mlp_ratio, norm_layer)

        #self.correlator = Correlator(channel_id, embed_dim, embed_squeeze)

        self.decoder = Decoder(in_chans, self.window_len, self.num_patches, embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio, norm_layer)

        self.str_decoder = DecoderStrong(decoder_embed_dim)

        
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss


        self.algnment_loss_crt = torch.nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=0.5)

        


    def patchify(self, sigs):
        """
        sigs: (N, 1, L)
        x: (N, num_patches , L)
        """
        p = self.window_len
        l = sigs.shape[-1] // p

        x = sigs.reshape(shape=(sigs.shape[0], 1, l, p))
        x = torch.einsum('nchp->nhpc', x)
        x = x.reshape(shape=(sigs.shape[0], l, p * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, num_patches , L)
        sigs: (N, 1, L)
        """
        p = self.window_len
        l = x.shape[1]

        
        #x = x.reshape(shape=(x.shape[0], l, p))
        #x = torch.einsum('nhpc->nchp', x)
        sigs = x.reshape(shape=(x.shape[0], 1, l * p))
        return sigs

        

    def propose_masking(self, batch_size, num_patches, mask_ratio, devic):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        len_keep = int(num_patches * (1 - mask_ratio))
        
        noise = torch.rand(batch_size, num_patches, device=devic)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        noise = noise.to('cpu')
        
        return ids_shuffle, ids_restore, ids_keep


    def perform_masking(self, x, mask_ratio, ids_restore, ids_keep):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask

    def forward_encoder(self, x, mask_ratio, ids_shuffle, ids_restore, ids_keep):
        # embed patches

        x, mask = self.encoder(x, mask_ratio, ids_shuffle, ids_restore, ids_keep)
        return x, mask

        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :] 

        
        # masking: length -> length * mask_ratio
        x, mask = self.perform_masking(x, mask_ratio, ids_restore, ids_keep)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask

    def get_cls_token(self, x, mask_ratio, ids_shuffle, ids_restore, ids_keep):
        # embed patches

        x, mask = self.encoder(x, mask_ratio, ids_shuffle, ids_restore, ids_keep)
        return x, mask

        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :] 

        
        # masking: length -> length * mask_ratio
        x, mask = self.perform_masking(x, mask_ratio, ids_restore, ids_keep)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask

    def forward_decoder(self, x, ids_restore):

        x,fin_x = self.decoder(x, ids_restore)
        return x,fin_x
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]


        ## handle nan
        ####x = torch.where(x.isnan(),0,x)

        return x

    def forward_decoder_strong(self, x):
        return self.str_decoder(x)

    def alignment_loss(self, own_embds, all_embds):



        anchors = torch.flatten(torch.unsqueeze(own_embds,dim=1).repeat(1,12,1,1), end_dim=-2)
        positvs = torch.flatten(all_embds,end_dim=-2)
        negatvs = torch.flatten(torch.roll(all_embds,1,0),end_dim=-2)


        return self.algnment_loss_crt(anchors, positvs, negatvs)


        #torch.nn.functional.normalize(torch.flatten(all_embds,end_dim=-2)) - torch.nn.functional.normalize(torch.flatten(all_embds,end_dim=-2))
        
        #return torch torch.unsqueeze(own_embds,dim=1).repeat(1,12,1)

        #torch.nn.functional.cosine_similarity(torch.flatten(all_embds,end_dim=-2), torch.flatten(torch.unsqueeze(own_embds,dim=1).repeat(1,12,1,1), end_dim=-2), dim=-1)
        
        ## COSINE ONLY SIMILAR
        return 1 - torch.mean(torch.nn.functional.cosine_similarity(torch.flatten(all_embds,end_dim=-2), torch.flatten(torch.unsqueeze(own_embds,dim=1).repeat(1,12,1,1), end_dim=-2), dim=-1))

        return 0
        
        triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
        loss_val = triplet_loss(embds, correlated_embds, uncorrelated_embds)

        return loss_val

    def reconstruction_loss(self, sigs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(sigs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


    def strong_reconstruction_loss(self, sigs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = sigs[:,:,450:-450]

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean()

        return loss



    def predict_reconstruction(self, sigs_all,num_patches, mask_ratio=0.75):

        devic = sigs_all.device

        ids_shuffle, ids_restore, ids_keep = self.propose_masking(len(sigs_all), num_patches, mask_ratio, devic)

        latent, mask = self.forward_encoder(sigs_all, mask_ratio, ids_shuffle, ids_restore, ids_keep)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]        
        return pred


    def log(self,sig_len, window_len, in_chans,
                 embed_dim, depth, num_heads,
                 decoder_embed_dim, decoder_depth, decoder_num_heads,
                 mlp_ratio, norm_layer, norm_pix_loss):
        print('Model config')
        print(f'MaskedAutoencoderViT1D(sig_len={sig_len}, window_len={window_len}, in_chans={in_chans},embed_dim={embed_dim}, depth={depth}, num_heads={num_heads},decoder_embed_dim={decoder_embed_dim}, decoder_depth={decoder_depth}, decoder_num_heads={decoder_embed_dim},mlp_ratio={mlp_ratio}, norm_layer={norm_layer}, norm_pix_loss={norm_pix_loss})')

