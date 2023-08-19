# Author: Nabil Ibtehaz (https://github.com/nibtehaz)


from functools import partial

import torch
import torch.nn as nn
from util.pos_embed import get_1d_sincos_pos_embed
from typing import Callable, List, Optional, Tuple, Union
from timm.models.vision_transformer import Block
from MAE1DCorrelated import MaskedAutoencoderViT1DCorrelated
from util.ecg_dataset import ECGPretrainDataset
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
mp.set_sharing_strategy('file_system')
import numpy as np
import gc
import os
import sys
import math
import timm.optim.optim_factory as optim_factory
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

NUM_CHANNELS = 12

#def generate_embedding_mp(mdl_data_device_mskshfl_msk_rstr):
#            mdl, dta, dvc, ids_shuffle, ids_restore = mdl_data_device_mskshfl_msk_rstr

class Message:

    def __init__(self, msg_type, msg_content, msg_seq_no):

        self.msg_type = msg_type
        self.msg_content = msg_content
        self.msg_seq_no = msg_seq_no

def get_time():
    return datetime.now().strftime("%H:%M:%S")



def mp_train_mae_coordinate(mae_mdl_bnk, msg_in_queues, msg_out_queues, args):

    
    sys.stdout = open(os.path.join(args.output_dir,'logs','log_coordinator.log'),'w')

    dataset_train = ECGPretrainDataset(os.path.join(args.data_path),args.input_size,mode='1D',channels='all')
    print(dataset_train, len(dataset_train))

    
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    #tflog_writer = SummaryWriter(log_dir=args.output_dir)

    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    avg_losses = [{"reconstruction_loss":0,"alignment_loss":0} for i in range(NUM_CHANNELS+1)]

    cur_best_loss = 1000000


    for epoch in range(args.start_epoch, args.epochs):

        total_samples = 0

        for i in range(len(avg_losses)):
            avg_losses[i]['reconstruction_loss'] = 0
            avg_losses[i]['alignment_loss'] = 0

        print(f'---EPOCH = {epoch}---')
        for data_iter_step, samples in enumerate(data_loader_train):

        
            X = samples.to(torch.float32)
            cur_seq_no = f'{epoch}.{data_iter_step}'

            batch_size = len(X)
            total_samples += batch_size
            ids_shuffle, ids_restore, ids_keep = mae_mdl_bnk.propose_masking(batch_size, mae_mdl_bnk.num_patches, mask_ratio=args.mask_ratio)

            for mdl_idx in range(NUM_CHANNELS):
                try:
                    msg_out_queues[mdl_idx].put(Message('INPUT_SIGNAL',
                                            {
                                                    'data':X[:,mdl_idx:(mdl_idx+1),:].clone(),
                                                    'ids_shuffle':ids_shuffle,
                                                    'ids_restore':ids_restore,
                                                    'ids_keep':ids_keep,
                                                    'mask_ratio':args.mask_ratio,
                                                    'recnstrcn_ls_wgt': math.cos((epoch/args.epochs)*(math.pi/2))+1,#1.0,
                                                    'algnmnt_ls_wgt': math.sin((epoch/args.epochs)*(math.pi/2))#0.0

                                            },cur_seq_no)
                                            , block=True, timeout=None)
                    if args.debug:
                        print(f"<{get_time()}>[Coordinator] put input signal in {mdl_idx} queue({msg_out_queues[mdl_idx].qsize()}) : {cur_seq_no}")

                except Exception as e:
                    print(f"<{get_time()}>@@@ERROR@@@{e}")
                
            if args.debug:
                print(f'<{get_time()}>coordinator put all input signals : {cur_seq_no}')

            computed_embeddings = []
            
            for mdl_idx in range(NUM_CHANNELS):

                rcvd_embdng = msg_in_queues[mdl_idx].get(block=True, timeout=None).msg_content
                rcvd_embdng = torch.unsqueeze(rcvd_embdng, dim=1)

                computed_embeddings.append(rcvd_embdng)
                if args.debug:
                    print(f'<{get_time()}>[Coordinator] computed_embeddings received from {mdl_idx} : {cur_seq_no}')

            computed_embeddings = torch.cat(computed_embeddings, dim=1)

        

            for mdl_idx in range(NUM_CHANNELS):

                msg_out_queues[mdl_idx].put(Message('ALIGN_EMBEDDING',
                                          {
                                                'computed_embeddings':computed_embeddings.clone(),

                                          },cur_seq_no)
                                          , block=True, timeout=None)

                if args.debug:
                    print(f'<{get_time()}>[Coordinator] put ALIGN_EMBEDDING request in {mdl_idx} : {cur_seq_no}')

            #print(f'coordinator put all align request : {epoch}')
            

            for mdl_idx in range(NUM_CHANNELS):
            
                msg = msg_in_queues[mdl_idx].get(block=True, timeout=None)
                                
                avg_losses[mdl_idx]['reconstruction_loss'] += msg.msg_content['reconstruction_loss'].item() * batch_size
                avg_losses[mdl_idx]['alignment_loss'] += msg.msg_content['alignment_loss'].item() * batch_size                
                
                
        print(f'({str(datetime.now()).split(" ")[1][:8]}) [{epoch}] losses')
        
        for mdl_idx in range(NUM_CHANNELS):

            avg_losses[mdl_idx]['reconstruction_loss'] /= total_samples
            avg_losses[mdl_idx]['alignment_loss'] /= total_samples
            
            avg_losses[-1]['reconstruction_loss'] += avg_losses[mdl_idx]['reconstruction_loss'] / NUM_CHANNELS
            avg_losses[-1]['alignment_loss'] += avg_losses[mdl_idx]['alignment_loss'] / NUM_CHANNELS

            print(f"\t\t<Channel {mdl_idx+1}> reconstruction_loss {avg_losses[mdl_idx]['reconstruction_loss']} | alignment_loss {avg_losses[mdl_idx]['alignment_loss']}")

            #tflog_writer.add_scalar(f'Channel {mdl_idx+1}/reconstruction_loss', avg_losses[mdl_idx]['reconstruction_loss'], epoch)
            #tflog_writer.add_scalar(f'Channel {mdl_idx+1}/alignment_loss', avg_losses[mdl_idx]['alignment_loss'], epoch)

        print(f"\t\t<Combined> reconstruction_loss {avg_losses[-1]['reconstruction_loss']} | alignment_loss {avg_losses[-1]['alignment_loss']}")

        #tflog_writer.add_scalar(f'Combined/reconstruction_loss', avg_losses[-1]['reconstruction_loss'], epoch)
        #tflog_writer.add_scalar(f'Combined/alignment_loss', avg_losses[-1]['alignment_loss'], epoch)

        total_loss = avg_losses[-1]['reconstruction_loss'] + avg_losses[-1]['alignment_loss']

        if epoch%5==0:
            for mdl_idx in range(NUM_CHANNELS):

                msg_out_queues[mdl_idx].put(Message('SAVE_MODEL',
                                          {
                                                'save':1,

                                          },f'epoch-{epoch}')
                                          , block=True, timeout=None)


        if(cur_best_loss >total_loss):
            
            print(f'[{epoch}] loss improved from {cur_best_loss} to {total_loss}')
            cur_best_loss = total_loss

            for mdl_idx in range(NUM_CHANNELS):

                msg_out_queues[mdl_idx].put(Message('SAVE_MODEL',
                                          {
                                                'save':1,

                                          },'best')
                                          , block=True, timeout=None)

        else:

            print(f'[{epoch}] loss didn\'t improved from {cur_best_loss}')


    for mdl_idx in range(NUM_CHANNELS):

        msg_out_queues[mdl_idx].put(Message('TRAINING_COMPLETED',
                                        {
                                            'data':0,

                                        },cur_seq_no)
                                        , block=True, timeout=None)



def mp_train_mae_process(mae_mdl, mae_optmzr, mae_schdlr, dvc, chnl_id, msg_in_queue, msg_out_queue, args):

    sys.stdout = open(os.path.join(args.output_dir,'logs',f'log_process_{chnl_id}.log'),'w')

    mae_mdl.to(dvc)
    mae_mdl.train()
    
    param_groups = optim_factory.add_weight_decay(mae_mdl, args.weight_decay)

    if mae_optmzr is None:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    else:
        optimizer = mae_optmzr

    if mae_schdlr is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,eta_min=1e-5)
    else:
        scheduler = mae_schdlr
        

    cur_epoch = args.start_epoch

    cur_latent = None
    cur_rec_loss = 100000
    cur_align_loss = 1000000
    recnstrcn_ls_wgt = 1.0
    algnmnt_ls_wgt = 1.0
    

    optimizer.zero_grad()


    while True:
        if args.debug:
            print(f'<{get_time()}>process {chnl_id} looking for msg, in queue({msg_in_queue.qsize(),msg_out_queue.qsize()})')
        try:
            msg = msg_in_queue.get(block=True, timeout=None)
        except Exception as e:
            print(f"@@@ERROR@@@{e}")
            continue


        if args.debug:
            print(f'<{get_time()}>MAE model {chnl_id} received {msg.msg_type}:{msg.msg_seq_no}')

        if msg.msg_type == 'INPUT_SIGNAL':
            
            dta = msg.msg_content['data'].to(dvc)
            ids_shuffle = msg.msg_content['ids_shuffle'].to(dvc)
            ids_restore = msg.msg_content['ids_restore'].to(dvc)
            ids_keep = msg.msg_content['ids_keep'].to(dvc)
            mask_ratio = msg.msg_content['mask_ratio']
            recnstrcn_ls_wgt = msg.msg_content['recnstrcn_ls_wgt']
            algnmnt_ls_wgt = msg.msg_content['algnmnt_ls_wgt']

            if cur_epoch < int(msg.msg_seq_no.split('.')[0]):
                cur_epoch = int(msg.msg_seq_no.split('.')[0])

                scheduler.step()


            cur_latent, mask = mae_mdl.forward_encoder(dta, mask_ratio, ids_shuffle, ids_restore, ids_keep)

            embd_response = Message(msg_type='Generated_Embedding', msg_content=cur_latent.detach().to('cpu').clone(), msg_seq_no=msg.msg_seq_no)
            msg_out_queue.put(embd_response, block=True, timeout=None)

            pred = mae_mdl.forward_decoder(cur_latent, ids_restore)
            cur_rec_loss = mae_mdl.reconstruction_loss(dta, pred, mask)

            mask = mask.to('cpu')
            dta = dta.to('cpu')
            ids_shuffle = ids_shuffle.to('cpu')
            ids_restore = ids_restore.to('cpu')
            ids_keep = ids_keep.to('cpu')
            pred = pred.to('cpu')

        elif msg.msg_type == 'ALIGN_EMBEDDING':

            all_latents = msg.msg_content['computed_embeddings'].to(dvc)
            cur_align_loss = mae_mdl.alignment_loss(cur_latent,all_latents)

            loss_response = Message(msg_type='Computed_Loss', 
                                    msg_content={
                                        'reconstruction_loss':cur_rec_loss.detach().to('cpu').clone(),
                                        'alignment_loss':cur_align_loss.detach().to('cpu').clone()
                                        },
                                    msg_seq_no=msg.msg_seq_no)
            msg_out_queue.put(loss_response, block=True, timeout=None)

            cur_total_loss = (cur_rec_loss  * recnstrcn_ls_wgt + cur_align_loss * algnmnt_ls_wgt) / ( recnstrcn_ls_wgt+algnmnt_ls_wgt )
            cur_total_loss = cur_total_loss / args.grad_accum_steps
            cur_total_loss.backward()

            if (int(msg.msg_seq_no.split('.')[1])==0) or ((int(msg.msg_seq_no.split('.')[1])+1)% args.grad_accum_steps==0):
                optimizer.step()
                optimizer.zero_grad()

            cur_total_loss = cur_total_loss.detach().to('cpu')
            cur_rec_loss = cur_rec_loss.detach().to('cpu')
            cur_align_loss = cur_align_loss.detach().to('cpu')
            all_latents = all_latents.detach().to('cpu')
            cur_latent = cur_latent.detach().to('cpu')

            print(f"({str(datetime.now()).split(' ')[1][:8]}) [{msg.msg_seq_no}]<Channel {chnl_id}> reconstruction_loss {cur_rec_loss.item()} | alignment_loss {cur_align_loss.item()}")

            

        elif msg.msg_type == 'SAVE_MODEL':
            
            torch.save(
                {'model':mae_mdl.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),}, 
                os.path.join(args.output_dir,'saved_models',f'mae_channel_{chnl_id}_{msg.msg_seq_no}.pth'))



        elif msg.msg_type == 'TRAINING_COMPLETED':
            
            return


class MAEBank(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, devices, sig_len=2500, window_len=100, learnable_channel_encoding=False, in_chans=1,
                 embed_dim=768,embed_squeeze=0.5, depth=12, num_heads=8,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True):
        super().__init__()

        self.log(sig_len, window_len, in_chans,
                 embed_dim, depth, num_heads,
                 decoder_embed_dim, decoder_depth, decoder_num_heads,
                 mlp_ratio, norm_layer, norm_pix_loss)

        # --------------------------------------------------------------------------
        # MAEs
        self.maes = torch.nn.ModuleList()
        
        
        for channel_id in range(NUM_CHANNELS):
            self.maes.append(
                MaskedAutoencoderViT1DCorrelated(channel_id,sig_len, window_len, in_chans,embed_dim, embed_squeeze, depth, num_heads,decoder_embed_dim, decoder_depth, decoder_num_heads,mlp_ratio, norm_layer, norm_pix_loss)
            )

        self.devices = devices

        self.num_patches = sig_len // window_len
                
        self.current_mask = None

        self.optimzers = [None] * 12
        self.schdulers = [None] * 12

        self.coordinate_process_queues = [ mp.Queue() for i in range(NUM_CHANNELS)]
        self.process_coordinate_queues = [ mp.Queue() for i in range(NUM_CHANNELS)]
        


    def propose_masking(self, batch_size, num_patches, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        len_keep = int(num_patches * (1 - mask_ratio))
        
        noise = torch.rand(batch_size, num_patches, device=self.devices[0])  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        noise = noise.to('cpu')
        ids_shuffle = ids_shuffle.to('cpu')
        ids_restore = ids_restore.to('cpu')
        ids_keep = ids_keep.to('cpu')

        return ids_shuffle, ids_restore, ids_keep

    
    
    def start_mp_daemons(self,args):
        

        processes = []

        
        self.share_memory()
                
        processes.append((mp.Process(target=mp_train_mae_coordinate, args=(self, self.process_coordinate_queues, self.coordinate_process_queues, args))))

        for i in range(NUM_CHANNELS):
            self.maes[i].share_memory()
            processes.append(mp.Process(target=mp_train_mae_process, args=(self.maes[i], self.optimzers[i], self.schdulers[i], f'cuda:{i%len(self.devices)}',i+1,self.coordinate_process_queues[i],self.process_coordinate_queues[i], args)))


 
        for p in processes:
            p.start()

        for p in processes:
            p.join()

    
    def generate_encoded_embedding(self,x,mask_ratio):
        
        batch_size = len(x)
        ids_shuffle, ids_restore, ids_keep = self.propose_masking(batch_size, self.num_patches, mask_ratio)

        n_jobs = len(self.devices)

        with mp.Pool(processes=n_jobs) as pool:  
            results = pool.starmap(generate_embedding_mp, [(self.maes[i],x[:,i:(i+1),:],self.devices[i%n_jobs],mask_ratio,ids_shuffle, ids_restore,ids_keep) for i in range(NUM_CHANNELS)])          

        return results
        #        encoded_embeddings = [pool.apply_async(generate_embedding_mp, args=((self.maes[i],x[:,i:(i+1),:],self.devices[i%n_jobs],ids_shuffle, ids_restore),)) for i in range(12)]
        #        pred_embeddings = [encoded_embedding.get() for encoded_embedding in encoded_embeddings]

        #processes = []
        """
        mdl_id = 0

        for i in range(len(self.devices)):
            p = mp.Process(target=generate_embedding_mp, args=(self.maes[mdl_id],x[:,mdl_id:(mdl_id+1),:],self.devices[mdl_id%n_jobs],mask_ratio, ids_shuffle, ids_restore, ids_keep))
            mdl_id += 1
            p.start()
            processes.append(p)

        for i in range(len(processes)):
            processes[i].join()
        #p1 = mp.Process(target=train, args=(model1, data_loader, optimizer1, criterion))
        #p1.start()
        #processes.append(p1)
        #p2 = mp.Process(target=train, args=(model2, data_loader, optimizer2, criterion))
        #p2.start()
        #processes.append(p2)

        #for p in processes:
        #    p.join()

        #with mp.Pool(processes=n_jobs) as pool:            
        #        encoded_embeddings = [pool.apply_async(generate_embedding_mp, args=((self.maes[i],x[:,i:(i+1),:],self.devices[i%n_jobs],ids_shuffle, ids_restore),)) for i in range(12)]
        #        pred_embeddings = [encoded_embedding.get() for encoded_embedding in encoded_embeddings]
        """
        gc.collect()

        #return processes



    def log(self,sig_len, window_len, in_chans,
                 embed_dim, depth, num_heads,
                 decoder_embed_dim, decoder_depth, decoder_num_heads,
                 mlp_ratio, norm_layer, norm_pix_loss):
        print('Model config')
        print(f'MaskedAutoencoderViT1D(sig_len={sig_len}, window_len={window_len}, in_chans={in_chans},embed_dim={embed_dim}, depth={depth}, num_heads={num_heads},decoder_embed_dim={decoder_embed_dim}, decoder_depth={decoder_depth}, decoder_num_heads={decoder_embed_dim},mlp_ratio={mlp_ratio}, norm_layer={norm_layer}, norm_pix_loss={norm_pix_loss})')

