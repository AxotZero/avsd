from pdb import set_trace as bp
import os
import json
from tqdm import tqdm
import torch
import time
# from time import time
from avsd_tan.utils import get_valid_position
import torch.nn.functional as F

from model.masking import mask
from evaluation.evaluate import AVSD_eval
from utilities.captioning_utils import HiddenPrints, get_lr
import wandb




def teacher_force_decoder(model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality):
    assert model.training is False, 'call model.eval first'

    with torch.no_grad():
        
        if 'audio' in modality:
            B, _Sa_, _Da_ = feature_stacks['audio'].shape
            device = feature_stacks['audio'].device
        elif modality == 'video':
            B, _Sv_, _Drgb_ = feature_stacks['rgb'].shape
            device = feature_stacks['rgb'].device
        else:
            raise Exception(f'Unknown modality: {modality}')

        # a mask containing 1s if the ending tok occured, 0s otherwise
        # we are going to stop if ending token occured in every sequence
        completeness_mask = torch.zeros(B, 1).byte().to(device)
        trg = (torch.ones(B, 1) * start_idx).long().to(device)

        while (trg.size(-1) <= max_len) and (not completeness_mask.all()):
            masks = make_masks(feature_stacks, trg, modality, pad_idx)
            preds = model(feature_stacks, trg, masks)
            next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)
            trg = torch.cat([trg, next_word], dim=-1)
            completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()

        return trg


def greedy_decoder(model, batch, max_len, start_idx, end_idx, pad_idx, modality,
                           sent_start_idx=None, sent_end_idx=None, last_only=False):
    assert model.training is False, 'call model.eval first'
    feature_stacks = batch['feature_stacks']
    dialog_idx = batch['dialog']
    start_pos, end_pos = get_context_positions(dialog_idx, sent_start_idx, sent_end_idx)
    sources = []
    targets = []
    attweights = []
    with torch.no_grad():

        if 'audio' in modality:
            B, _Sa_, _Da_ = feature_stacks['audio'].shape
            device = feature_stacks['audio'].device
        elif modality == 'video':
            B, _Sv_, _Drgb_ = feature_stacks['rgb'].shape
            device = feature_stacks['rgb'].device
        else:
            raise Exception(f'Unknown modality: {modality}')
        
        # for each QA turn, only last if last_only
        
        s = 0 if not last_only else start_pos.size(-1)-1
        for t in range(s, start_pos.size(1)):
            # store source information
            max_src_context_len = int(torch.max(end_pos[:, t] - start_pos[:, t]))
            src = torch.full((B, max_src_context_len), end_idx, dtype=torch.long, device=device)
            for b, (s, e) in enumerate(zip(start_pos[:, t], end_pos[:, t])):
                if e >= 0:
                    src[b, :e-s] = dialog_idx[b, s:e]
            sources.append(src)
            # prepare context used for teacher forcing
            max_context_len = int(torch.max(end_pos[:, t])) + 1
            trg = torch.full((B, max_context_len), pad_idx, dtype=torch.long, device=device)
            completeness_mask = torch.zeros(B, 1).byte().to(device)
            current_position = torch.zeros(B, dtype=torch.long, device=device)
            for b, e in enumerate(end_pos[:, t]):
                if e >= 0:
                    trg[b, :e+1] = dialog_idx[b, :e+1]
                    current_position[b] = e
                else: # no more sentences
                    completeness_mask[b] = 1
            # greedy decoding
            out = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
            pad_idx_ = torch.full((B, 1), pad_idx, dtype=torch.long, device=device)
            end_idx_ = torch.full((B, 1), end_idx, dtype=torch.long, device=device)
            
            batch_indices = torch.arange(B, dtype=torch.long, device=device)
            map2d = None
            while (out.size(-1) <= max_len) and (not completeness_mask.all()):
                
                # masks = make_masks(feature_stacks, trg, modality, pad_idx)
                # pad_mask, text_mask = make_text_masks(trg, pad_idx)
                # preds, attn, map2d = model(feature_stacks, , pad_mask, text_mask, )
                # bp()
                preds, attn, map2d = model(
                    batch['feature_stacks'], batch['visual_mask'], batch['audio_mask'],
                    dialog_x=trg, map2d=map2d, ret_map2d=True, compute_loss=False
                )
                preds[:, :, 0] = float('-inf')  # suppress UNK
                next_word = torch.where(completeness_mask==0,
                                        preds[batch_indices, current_position].max(dim=-1)[1].unsqueeze(1),
                                        end_idx_)
                out = torch.cat([out, next_word], dim=-1)
                trg = torch.cat([trg, pad_idx_], dim=-1)
                current_position += 1
                trg[batch_indices, current_position] = next_word[:, 0]
                completeness_mask = completeness_mask | torch.eq(next_word, end_idx_).byte()
            targets.append(out)
            attweights.append(attn[:, -1])
    return sources, targets, attweights


def topk_topp_decoder(model, batch, max_len, start_idx, end_idx, pad_idx, modality,
                      sent_start_idx=None, sent_end_idx=None, last_only=False, 
                      topk=0, topp=0.0, repetition_penalty=2.0, filter_value=0):
    assert model.training is False, 'call model.eval first'
    feature_stacks = batch['feature_stacks']
    caption_idx = batch['caption']
    start_pos, end_pos = get_context_positions(caption_idx, sent_start_idx, sent_end_idx)
    sources = []
    targets = []
    attweights = []
    with torch.no_grad():

        if 'audio' in modality:
            B, _Sa_, _Da_ = feature_stacks['audio'].shape
            device = feature_stacks['audio'].device
        elif modality == 'video':
            B, _Sv_, _Drgb_ = feature_stacks['rgb'].shape
            device = feature_stacks['rgb'].device
        else:
            raise Exception(f'Unknown modality: {modality}')
        
        # for each QA turn, only last if last_only
        
        s = 0 if not last_only else start_pos.size(-1)-1
        for t in range(s, start_pos.size(1)):
            # store source information
            max_src_context_len = int(torch.max(end_pos[:, t] - start_pos[:, t]))
            src = torch.full((B, max_src_context_len), end_idx, dtype=torch.long, device=device)
            for b, (s, e) in enumerate(zip(start_pos[:, t], end_pos[:, t])):
                if e >= 0:
                    src[b, :e-s] = caption_idx[b, s:e]
            sources.append(src)
            # prepare context used for teacher forcing
            max_context_len = int(torch.max(end_pos[:, t])) + 1
            trg = torch.full((B, max_context_len), pad_idx, dtype=torch.long, device=device)
            completeness_mask = torch.zeros(B, 1).byte().to(device)
            current_position = torch.zeros(B, dtype=torch.long, device=device)
            for b, e in enumerate(end_pos[:, t]):
                if e >= 0:
                    trg[b, :e+1] = caption_idx[b, :e+1]
                    current_position[b] = e
                else: # no more sentences
                    completeness_mask[b] = 1
            # greedy decoding
            out = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
            pad_idx_ = torch.full((B, 1), pad_idx, dtype=torch.long, device=device)
            end_idx_ = torch.full((B, 1), end_idx, dtype=torch.long, device=device)
            
            batch_indices = torch.arange(B, dtype=torch.long, device=device)
            map2d = None
            while (out.size(-1) <= max_len) and (not completeness_mask.all()):
                
                # masks = make_masks(feature_stacks, trg, modality, pad_idx)
                pad_mask, text_mask = make_text_masks(trg, pad_idx)
                # preds, attn, map2d = model(feature_stacks, , pad_mask, text_mask, )

                preds, attn, map2d = model(
                    batch['feature_stacks'], trg, 
                    batch['visual_mask'], batch['audio_mask'], 
                    padding_mask=pad_mask, text_mask=text_mask,
                    map2d=map2d, ret_map2d=True
                )
                preds[:, :, [0, sent_start_idx, sent_start_idx]] = float('-inf')  # suppress UNK


                # filter topk
                preds = preds[batch_indices, current_position]
                preds = F.softmax(preds, dim=-1)

                for index in range(B):
                    for token_id in set(out[index].cpu().numpy()):
                        preds[index][token_id] /= repetition_penalty
                preds = preds / preds.sum(-1)
                if topk > 0:
                    for pred in preds:
                        indices_to_remove = pred < torch.topk(pred, topk)[0][..., -1, None]
                        pred[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷
                # filter topp
                if topp > 0.0:
                    sorted_logits, sorted_indices = torch.sort(preds, descending=True, dim=-1)  # 对logits进行递减排序
                    cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > topp
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    for index, pred in enumerate(preds):
                        indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
                        pred[indices_to_remove] = filter_value
                
                # select word
                next_word = torch.where(completeness_mask==0,
                                        torch.multinomial(preds, num_samples=1),
                                        end_idx_)
                
                out = torch.cat([out, next_word], dim=-1)
                trg = torch.cat([trg, pad_idx_], dim=-1)
                current_position += 1
                trg[batch_indices, current_position] = next_word[:, 0]
                completeness_mask = completeness_mask | torch.eq(next_word, end_idx_).byte()
            targets.append(out)
            attweights.append(attn[:, -1])
    return sources, targets, attweights


def beam_search_decoder(model, batch, max_len, start_idx, end_idx, pad_idx, modality,
                           sent_start_idx=None, sent_end_idx=None, last_only=False, beam_size=5):
    assert model.training is False, 'call model.eval first'
    feature_stacks = batch['feature_stacks']
    caption_idx = batch['caption']
    start_pos, end_pos = get_context_positions(caption_idx, sent_start_idx, sent_end_idx)
    sources = []
    targets = []
    attweights = []
    with torch.no_grad():

        if 'audio' in modality:
            B, _Sa_, _Da_ = feature_stacks['audio'].shape
            device = feature_stacks['audio'].device
        elif modality == 'video':
            B, _Sv_, _Drgb_ = feature_stacks['rgb'].shape
            device = feature_stacks['rgb'].device
        else:
            raise Exception(f'Unknown modality: {modality}')
        
        # for each QA turn, only last if last_only
        
        s = 0 if not last_only else start_pos.size(-1)-1
        for t in range(s, start_pos.size(1)):
            # store source information
            max_src_context_len = int(torch.max(end_pos[:, t] - start_pos[:, t]))
            src = torch.full((B, max_src_context_len), end_idx, dtype=torch.long, device=device)
            for b, (s, e) in enumerate(zip(start_pos[:, t], end_pos[:, t])):
                if e >= 0:
                    src[b, :e-s] = caption_idx[b, s:e]
            sources.append(src)
            # prepare context used for teacher forcing
            max_context_len = int(torch.max(end_pos[:, t])) + 1
            trg = torch.full((B, max_context_len), pad_idx, dtype=torch.long, device=device)
            completeness_mask = torch.zeros(B, 1).byte().to(device)
            current_position = torch.zeros(B, dtype=torch.long, device=device)
            for b, e in enumerate(end_pos[:, t]):
                if e >= 0:
                    trg[b, :e+1] = caption_idx[b, :e+1]
                    current_position[b] = e
                else: # no more sentences
                    completeness_mask[b] = 1
            # greedy decoding
            out = torch.full((B, 1), start_idx, dtype=torch.long, device=device)
            pad_idx_ = torch.full((B, 1), pad_idx, dtype=torch.long, device=device)
            end_idx_ = torch.full((B, 1), end_idx, dtype=torch.long, device=device)
            
            batch_indices = torch.arange(B, dtype=torch.long, device=device)
            map2d = None
            while (out.size(-1) <= max_len) and (not completeness_mask.all()):
                pad_mask, text_mask = make_text_masks(trg, pad_idx)

                preds, attn, map2d = model(
                    batch['feature_stacks'], trg, 
                    batch['visual_mask'], batch['audio_mask'], 
                    padding_mask=pad_mask, text_mask=text_mask,
                    map2d=map2d, ret_map2d=True
                )
                preds[:, :, 0] = float('-inf')  # suppress UNK
                next_word = torch.where(completeness_mask==0,
                                        preds[batch_indices, current_position].max(dim=-1)[1].unsqueeze(1),
                                        end_idx_)
                out = torch.cat([out, next_word], dim=-1)
                trg = torch.cat([trg, pad_idx_], dim=-1)
                current_position += 1
                trg[batch_indices, current_position] = next_word[:, 0]
                completeness_mask = completeness_mask | torch.eq(next_word, end_idx_).byte()
            targets.append(out)
            attweights.append(attn[:, -1])
    return sources, targets, attweights


def save_model(cfg, epoch, model, optimizer, val_loss_value,
               val_metrics, vocab_size):
    
    dict_to_save = {
        'config': cfg,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss_value,
        'val_metrics': val_metrics,
        'vocab_size': vocab_size,
    }
    
    # in case TBoard is not defined make logdir (can be deleted if Config is used)
    os.makedirs(cfg.model_checkpoint_path, exist_ok=True)
    
#     path_to_save = os.path.join(cfg.model_checkpoint_path, f'model_e{epoch}.pt')
    path_to_save = os.path.join(cfg.model_checkpoint_path, f'best_cap_model.pt')
    torch.save(dict_to_save, path_to_save)


def make_masks(feature_stacks, captions, modality, pad_idx):
    masks = {}

    if modality == 'video':
        if captions is None:
            masks['V_mask'] = mask(feature_stacks['rgb'][:, :, 0], None, pad_idx)
        else:
            masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
    elif modality == 'audio':
        assert len(feature_stacks['audio'].shape) == 3
        if captions is None:
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
        else:
            masks['A_mask'], masks['C_mask'] = mask(feature_stacks['audio'][:, :, 0], captions, pad_idx)
    elif modality == 'audio_video':
        assert len(feature_stacks['audio'].shape) == 3
        if captions is None:
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
            masks['V_mask'] = mask(feature_stacks['rgb'][:, :, 0], None, pad_idx)
        else:
            masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
            masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
    elif modality == 'subs_audio_video':
        assert len(feature_stacks['audio'].shape) == 3
        masks['V_mask'], masks['C_mask'] = mask(feature_stacks['rgb'][:, :, 0], captions, pad_idx)
        masks['A_mask'] = mask(feature_stacks['audio'][:, :, 0], None, pad_idx)
        masks['S_mask'] = mask(feature_stacks['subs'], None, pad_idx)

    return masks


def get_context_positions(caption_idx, sent_start_idx, sent_end_idx):
    """ obtain context end positions based on sent_start_idx and sent_end_idx
    """
    B, L = caption_idx.size()
    cap = caption_idx.view(-1)
    positions = torch.arange(len(cap), device=cap.device)
    start_positions = positions[cap == sent_start_idx]
    end_positions = positions[cap == sent_end_idx]
    assert len(start_positions) == len(end_positions)
    start_pos_list = [[] for _ in range(B)]
    end_pos_list = [[] for _ in range(B)]
    for s, e in zip(start_positions.tolist(), end_positions.tolist()):
        start_pos_list[s // L].append(s % L)
        end_pos_list[e // L].append(e % L)
    max_npos = max([len(pl) for pl in start_pos_list])
    start_pos = torch.full((B, max_npos), -1, dtype=torch.long, device=cap.device)
    end_pos = torch.full((B, max_npos), -1, dtype=torch.long, device=cap.device)
    for b in range(B):
        start_pos[b, :len(start_pos_list[b])] = torch.tensor(start_pos_list[b], dtype=torch.long)
        end_pos[b, :len(end_pos_list[b])] = torch.tensor(end_pos_list[b], dtype=torch.long)
    return start_pos, end_pos


def get_context_masked_target(caption_idx, sent_start_idx, sent_end_idx, end_idx, pad_idx):
    """ replace token_ids between context_start_idx and context_end_idx with pad_idx, and
        also replace context_start_idx with end_idx unless it is in the beginning,
        e.g. in QA dialog '<s> Q: w1 w2 w3 A: w4 w5 Q: w6 w7 A: w8 w9 </s>' is converted to
        '- - - - - w4 w5 </s> - - - w8 w9 </s>',
        where 'Q:', 'A:', and '-' represent context_start, context_end, and pad tokens.
    """
    caption_idx_y = caption_idx[:, 1:]
    if sent_start_idx is not None and sent_end_idx is not None:
        L = caption_idx_y.size(1)
        cap = torch.clone(caption_idx_y).view(-1)
        positions = torch.arange(len(cap), device=cap.device)
        context_start_positions = positions[cap == sent_start_idx]
        context_end_positions = positions[cap == sent_end_idx]
        assert len(context_start_positions) == len(context_end_positions)
        for i in range(len(context_start_positions)):
            cap[context_start_positions[i]] = pad_idx if context_start_positions[i] % L == 0 else end_idx
            cap[context_start_positions[i] + 1 : context_end_positions[i] + 1] = pad_idx
        return cap.view(caption_idx_y.size())
    else:
        return caption_idx_y


# def text2xy(batch_text_indices, sent_start_idx, sent_end_idx, end_idx, pad_idx):
#     return batch_text_indices[:, :-1], get_context_masked_target(batch_text_indices, sent_start_idx, sent_end_idx, end_idx, pad_idx)


def batch_to_device(batch, device):
    batch['starts'] = batch['starts'].to(device)
    batch['ends'] = batch['ends'].to(device)
    batch['feature_stacks']['rgb'] = batch['feature_stacks']['rgb'].to(device)
    batch['feature_stacks']['flow'] = batch['feature_stacks']['flow'].to(device)
    batch['feature_stacks']['audio'] = batch['feature_stacks']['audio'].to(device)
    batch['visual_mask'] = batch['visual_mask'].to(device)
    batch['audio_mask'] = batch['audio_mask'].to(device)
    batch['caption'] = batch['caption'].to(device)
    batch['summary'] = batch['summary'].to(device)
    batch['dialog'] = batch['dialog'].to(device)
    batch['tan_label'] = batch['tan_label'].to(device)
    batch['tan_mask'] = batch['tan_mask'].to(device)
    return batch


def training_loop(cfg, model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    total_sim_loss = 0
    total_dialog_loss = 0
    total_caption_loss = 0
    total_tan_loss = 0

    loader.dataset.update_iterator()
    pbar = tqdm(loader, ncols=100)
    for i, batch in enumerate(pbar):
        batch = batch_to_device(batch, torch.device(cfg.device))

        optimizer.zero_grad()
        dialog_x = batch['dialog'][:, :-1]
        dialog_y = get_context_masked_target(
            batch['dialog'],
            loader.dataset.sent_start_idx,
            loader.dataset.sent_end_idx,
            loader.dataset.end_idx,
            loader.dataset.pad_idx
        )

        summary_x, summary_y = batch['summary'][:, :-1], batch['summary'][:, 1:]

        sim_loss, tan_loss, dialog_loss, caption_loss = model(
            batch['feature_stacks'], batch['visual_mask'], batch['audio_mask'],
            dialog_x, dialog_y,
            summary_x, summary_y,
            batch['tan_label'], batch['tan_mask'],
            compute_loss=True
        )

        # multi device
        sim_loss = sim_loss.mean()
        tan_loss = tan_loss.mean()
        dialog_loss = dialog_loss.mean()
        caption_loss = caption_loss.mean()

        loss = (
            cfg.sim_weight * sim_loss + 
            cfg.tan_weight * tan_loss +
            cfg.dialog_weight * dialog_loss +
            cfg.caption_weight * caption_loss
        )
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_sim_loss += sim_loss.item()
        total_dialog_loss += dialog_loss.item()
        total_caption_loss += caption_loss.item()
        total_tan_loss += tan_loss.item()

        pbar.set_description(
            '{:<5} {}, sim:{:.3f}, tan:{:.3f}, cap:{:.3f}, dig:{:.3f}'.format(
                'train', epoch, 
                sim_loss.item(), tan_loss.item(), caption_loss.item(), dialog_loss.item()
            )
        )
        pbar.update()

    total_loss /= len(loader)
    total_sim_loss /= len(loader)
    total_dialog_loss /= len(loader)
    total_caption_loss /= len(loader)
    total_tan_loss /= len(loader)

    if cfg.wandb:
        wandb.log(
            {
                'train/loss': total_loss,
                'train/sim_loss': total_sim_loss,
                'train/tan_loss': total_tan_loss,
                'train/dialog_loss': total_dialog_loss,
                'train/caption_loss': total_caption_loss,
            },
            step=epoch
        )

    time.sleep(1)
    print('train {}, sim:{:.3f}, tan:{:.3f}, cap:{:.3f}, dig:{:.3f}'.format(
        epoch, total_sim_loss, total_tan_loss, 
        total_caption_loss, total_dialog_loss
    ))
            

def validation_next_word_loop(cfg, model, loader, epoch):
    model.eval()
    total_loss = 0
    total_sim_loss = 0
    total_dialog_loss = 0
    total_caption_loss = 0
    total_tan_loss = 0

    loader.dataset.update_iterator()
    phase = loader.dataset.phase
    # progress_bar_name = f'{cfg.exp_name}: {phase:<5} {epoch} @ {cfg.device}'

    pbar = tqdm(loader, ncols=100)
    for i, batch in enumerate(pbar):
        batch = batch_to_device(batch, torch.device(cfg.device))

        dialog_x = batch['dialog'][:, :-1]
        dialog_y = get_context_masked_target(
            batch['dialog'],
            loader.dataset.sent_start_idx,
            loader.dataset.sent_end_idx,
            loader.dataset.end_idx,
            loader.dataset.pad_idx
        )

        summary_x, summary_y = batch['summary'][:, :-1], batch['summary'][:, 1:]

        with torch.no_grad():
            sim_loss, tan_loss, dialog_loss, caption_loss = model(
                batch['feature_stacks'], batch['visual_mask'], batch['audio_mask'],
                dialog_x, dialog_y,
                summary_x, summary_y,
                batch['tan_label'], batch['tan_mask'],
                compute_loss=True
            )

            # multi device
            sim_loss = sim_loss.mean()
            tan_loss = tan_loss.mean()
            dialog_loss = dialog_loss.mean()
            caption_loss = caption_loss.mean()

            loss = (
                cfg.sim_weight * sim_loss + 
                cfg.tan_weight * tan_loss +
                cfg.dialog_weight * dialog_loss +
                cfg.caption_weight * caption_loss
            )

            total_loss += loss.item()
            total_sim_loss += sim_loss.item()
            total_dialog_loss += dialog_loss.item()
            total_caption_loss += caption_loss.item()
            total_tan_loss += tan_loss.item()

            pbar.set_description(
                '{:<5} {}, sim:{:.3f}, tan:{:.3f}, cap:{:.3f}, dig:{:.3f}'.format(
                    phase, epoch, 
                    sim_loss.item(), tan_loss.item(), caption_loss.item(), dialog_loss.item()
                )
            )
            pbar.update()
            
    total_loss /= len(loader)
    total_sim_loss /= len(loader)
    total_dialog_loss /= len(loader)
    total_caption_loss /= len(loader)
    total_tan_loss /= len(loader)

    if cfg.wandb:
        wandb.log(
            {
                'valid/loss': total_loss,
                'valid/sim_loss': total_sim_loss,
                'valid/tan_loss': total_tan_loss,
                'valid/dialog_loss': total_dialog_loss,
                'valid/caption_loss': total_caption_loss,
            },
            step=epoch
        )

    time.sleep(1)
    print('{:<5} {}, sim:{:.3f}, tan:{:.3f}, cap:{:.3f}, dig:{:.3f}'.format(
        phase, epoch, total_sim_loss, total_tan_loss, 
        total_caption_loss, total_dialog_loss
    ))

    return total_loss


def validation_1by1_loop(cfg, model, loader, epoch):
    start_timer = time.time()
    
    # init the dict with results and other technical info
    predictions = {
        'dialogs': [],
    }
    model.eval()
    loader.dataset.update_iterator()
    
    start_idx = loader.dataset.start_idx
    end_idx = loader.dataset.end_idx
    pad_idx = loader.dataset.pad_idx
    sent_start_idx = loader.dataset.sent_start_idx
    sent_end_idx = loader.dataset.sent_end_idx
    phase = loader.dataset.phase
    # feature_names = loader.dataset.feature_names
    
    reference_paths = cfg.reference_paths
    progress_bar_name = f'{cfg.exp_name}: {phase} 1by1 {epoch} @ {cfg.device}'
    
    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        ### PREDICT TOKENS ONE-BY-ONE AND TRANSFORM THEM INTO STRINGS TO FORM A SENTENCE

        batch = batch_to_device(batch, torch.device(cfg.device))

        if cfg.decoding_method == 'greedy':
            ints_stack_list = greedy_decoder(
                model, batch, cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality,
                sent_start_idx=sent_start_idx, sent_end_idx=sent_end_idx, last_only=cfg.last_only,
            )
        elif cfg.decoding_method == 'topk_topp':
            ints_stack_list = topk_topp_decoder(
                model, batch, cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality,
                sent_start_idx=sent_start_idx, sent_end_idx=sent_end_idx, last_only=cfg.last_only,
                topk=cfg.topk, topp=cfg.topp
            )


        input_lengths = torch.sum(mask(batch['feature_stacks']['rgb'][:, :, 0], None, pad_idx), dim=-1).cpu().view(-1)
        list_of_lists_with_filtered_sentences = [[] for _ in range(len(ints_stack_list[0][0]))]
        for ints_stack1, ints_stack2, attw_stack in zip(ints_stack_list[0], ints_stack_list[1], ints_stack_list[2]):
            ints_stack1 = ints_stack1.cpu().numpy()  # what happens here if I use only cpu?
            ints_stack2 = ints_stack2.cpu().numpy()  # what happens here if I use only cpu?
            attw_stack = attw_stack.cpu()
            # transform integers into strings
            list_of_lists_with_strings1 = [[loader.dataset.train_vocab.itos[i] for i in ints] for ints in ints_stack1]
            list_of_lists_with_strings2 = [[loader.dataset.train_vocab.itos[i] for i in ints] for ints in ints_stack2]
            ### FILTER PREDICTED TOKENS
            # initialize the list to fill it using indices instead of appending them

            for b, (strings1, strings2, attw) in enumerate(zip(list_of_lists_with_strings1, list_of_lists_with_strings2, attw_stack)):
                # remove starting token and everything after ending token
                if len(strings1) > 0:
                    strings1 = strings1[1:]  # skip Q:
                else:
                    continue  # no more turns
                if len(strings2) > 0:
                    strings2 = strings2[1:]  # skip <s>
                try:
                    first_entry_of_eos1 = strings1.index('</s>')
                    strings1 = strings1[:first_entry_of_eos1]
                except ValueError:
                    pass
                try:
                    first_entry_of_eos2 = strings2.index('</s>')
                    strings2 = strings2[:first_entry_of_eos2]
                except ValueError:
                    pass
                if len(strings1) == 0:
                    continue
                sentence1 = ' '.join(strings1)
                sentence2 = ' '.join(strings2)

                # find regions for reasoning with attention weights over visual feature frames
                # TODO: detection of multiple regions and audio features should be considered
                ilen = input_lengths[b]
                s, e = get_valid_position(cfg.num_seg)[int(torch.argmax(attw))]
                start_time = s / cfg.num_seg
                # end_time = e / cfg.num_seg
                end_time = (e+1) / cfg.num_seg
                # attw_mean = torch.mean(attw[:len(strings2)], dim=0)[:ilen]
                # frame_indices = torch.arange(ilen, dtype=torch.float) / ilen  # relative frame positions
                # frame_mean = float((frame_indices * attw_mean).sum())  # expected value of attended frame
                # frame_std = float(((frame_indices - frame_mean) ** 2 * attw_mean).sum().sqrt())
                # start_time = max(0.0, (frame_mean - cfg.region_std_coeff * frame_std))
                # end_time = min(1.0, (frame_mean + cfg.region_std_coeff * frame_std))
                list_of_lists_with_filtered_sentences[b].append((sentence1, sentence2, (start_time, end_time)))

        ### ADDING RESULTS TO THE DICT WITH RESULTS
        for video_id, start, end, sents in zip(batch['video_ids'], batch['starts'], batch['ends'],
                                               list_of_lists_with_filtered_sentences):
            # 
            segment = []
            for sent in sents:
                start_time, end_time = sent[2]
                dur = end.item() - start.item()
                start_time = start_time * dur + start.item()
                end_time = end_time * dur + start.item()
                segment.append({
                    'question': sent[0],
                    'answer': sent[1],
                    'reason': [{'timestamp': [start_time, end_time], 'sentence': ''}]
                })
            predictions['dialogs'].append({'image_id': video_id, 'dialog': segment})

    if cfg.log_path is None:
        return None
    else:
        # SAVING THE RESULTS IN A JSON FILE
        if cfg.procedure == 'train_test':
            save_filename = f'captioning_results_{phase}_e{epoch}.json'
        else:
            save_filename = f'captioning_results_{phase}.json'
        submission_path = os.path.join(cfg.log_path, save_filename)

        # in case TBoard is not defined make logdir
        os.makedirs(cfg.log_path, exist_ok=True)

        # rename if already exists
        if os.path.exists(submission_path):
            root, ext = os.path.splitext(submission_path)
            n = 1
            while os.path.exists(submission_path):
                submission_path = f'{root}-{n}{ext}'
                n += 1

        with open(submission_path, 'w') as outf:
            json.dump(predictions, outf, indent=2)
        duration = time.time() - start_timer
        # blocks the printing
        with HiddenPrints():
            val_metrics = AVSD_eval(ground_truth_filenames=reference_paths,
                                    prediction_filename=submission_path,
                                    stopwords_filename=cfg.stopwords,
                                    last_only=cfg.last_only,
                                    verbose=False).evaluate()

        return val_metrics, duration