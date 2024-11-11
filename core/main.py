import os
import argparse
import random
import math
import json
import copy

import torch
import torchmetrics

import numpy as np
import cv2
from PIL import Image
import openslide
from sklearn.metrics import precision_recall_curve

# to use pROC to count auc with ci (R language)
import rpy2
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)
proc = importr("pROC")


def feature_processor(args):
    print('start feature processing ...')
    dataset_info = json.load(open(args.dataset_info))
    os.makedirs(args.dump_features, exist_ok=True)

    for k, v in dataset_info.items():
        if os.path.exists(os.path.join(args.dump_features, k + '.npy')):
            continue

        feats, names, patch_label, wsi_label = [], [], [], -1
        
        wsi_label = v['wsi_label']
        if 'patch_labels' in v:
            mask = cv2.imread(v['patch_labels'])[:, :, 0]

        in_dir = os.path.join(args.raw_feature_path, k + '_files')
        in_dir = in_dir if in_dir[-1] != '/' else in_dir[:-1]
        patch_path = in_dir.replace(in_dir.split('/')[-2], 'images')
        ori_dir = sorted([int(_) for _ in os.listdir(patch_path)])[-1]
        patch_path = os.path.join(patch_path, str(ori_dir))

        for f in os.listdir(os.path.join(in_dir, 'x20')):
            feat = np.load(os.path.join(in_dir, 'x20', f))
            feat = feat / np.linalg.norm(feat, ord=2, axis=0)
            x, y = f.split('.')[0].split('_')
            x, y = int(x), int(y)

            if 'patch_labels' in v:
                if mask[y, x] == 255:
                    continue

                patch_label.append(mask[y, x])

            name = os.path.join(patch_path, f.replace('.npy', '.jpeg'))
            names.append(name)
            feats.append(feat)
        
        info = {'features': np.stack(feats, 0), 'patch_names': names, \
            'patch_labels': np.array(patch_label), 'wsi_label': wsi_label}
        np.save(os.path.join(args.dump_features, k + '.npy'), info)
    
    print('finish feature processing and saving!')


def save_matched_patchs(gallery_patch_names, query_patch_names, gallery_idxs,
    query_idxs, similarity, outpath='', tag='neg'):

    for i in query_idxs:
        save_dir = os.path.join(outpath, query_patch_names[i].split('/')[-1].split('.')[0])
        os.makedirs(save_dir, exist_ok=True)
        os.system('cp ' + query_patch_names[i] + ' ' + save_dir + '/')

        tag_dir = os.path.join(save_dir, tag)
        os.makedirs(tag_dir, exist_ok=True)
        if len(gallery_idxs.shape) == 2:
            for j, v in enumerate(np.random.choice(gallery_idxs[:, i].cpu(), 10)):
                new_name = gallery_patch_names[v].split('/')[-1].split('.')
                new_name = new_name[0] + '_' + str(similarity[j, i].item()) + '.' + new_name[1]
                os.system('cp ' + gallery_patch_names[v] + ' ' + tag_dir + '/' + new_name)
        else:
            for j, v in enumerate(np.random.choice(gallery_idxs[:10].cpu(), min(10, gallery_idxs.shape[0]))):
                new_name = gallery_patch_names[v].split('/')[-1].split('.')
                new_name = new_name[0] + '_' + str(similarity[j].item()) + '.' + new_name[1]
                os.system('cp ' + gallery_patch_names[v] + ' ' + tag_dir + '/' + new_name)


def save_wsi_heatmap(patch_pred, wsi_fn, vis_path=''):
    pred = patch_pred[patch_pred != 255]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    pred = (pred - pred.mean()).clip(0, 1)
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    pred = (pred * 255).astype('uint8')
    patch_pred[patch_pred != 255] = pred
    patch_pred[patch_pred == 255] = 0

    wsi = openslide.OpenSlide(wsi_fn)
    scale = -1 if 'GC' in vis_path else 6
    out = np.array(wsi.read_region((0, 0), scale, wsi.level_dimensions[scale]))[:, :, :3]
    
    vis = cv2.resize(np.stack([patch_pred, patch_pred, patch_pred], -1), \
            (out.shape[1], out.shape[0]), interpolation=cv2.INTER_CUBIC)
    vis = cv2.applyColorMap(vis[:, :, 0], cv2.COLORMAP_JET)
    out = vis
    
    os.makedirs(vis_path, exist_ok=True)
    cv2.imwrite(os.path.join(vis_path, wsi_fn.split('/')[-1].split('.')[0] + '.jpg'), out)


def topk_low_memory(inp, n, dim):
    scores, idxs = [], []
    offset = 0
    chunk_num = max(1, inp.shape[dim] // 10000)
    for i in inp.chunk(chunk_num, dim):
        score, idx = i.cuda().topk(min(n, i.shape[dim]), dim)
        scores.append(score)
        idxs.append(idx + offset)
        offset += i.shape[dim]

    scores = torch.cat(scores, dim)
    scores, idxs2 = scores.topk(min(n, scores.shape[dim]), dim)
    idxs = torch.cat(idxs, dim).transpose(0, dim) # ori index
    idxs2 = idxs2.transpose(0, dim) # idx after reduction
    out_idxs = []
    for i in range(idxs2.shape[-1]):
        out_idxs.append(idxs[idxs2[:, i], i])
    out_idxs = torch.stack(out_idxs, -1).transpose(0, dim)

    return scores, out_idxs


# large query number lead to "CUDA error: an illegal memory access was encountered"
# keep tok n per 20000
def topk_low_memory_(inp, n):
    scores, idxs = [], []
    for i in range(math.ceil(inp.shape[1] / 20000)):
        a, b = topk_low_memory(inp[:, i * 20000: i * 20000 + 20000], n, 0)
        scores.append(a)
        idxs.append(b)

    return torch.cat(scores, 1), torch.cat(idxs, 1)


def INC(args, gallery_feats, gallery_labels, gallery_patch_names, 
    query_feats, query_patch_names, wsi_size, top_instance=1, vis_info=None):

    query_feats = query_feats.t()

    # chunks to aviod out of memory
    cosine = []
    for i in range(math.ceil(gallery_feats.shape[0] / 10000)):
        cosine.append((gallery_feats[i * 10000: (i + 1) * 10000] @ query_feats).cpu())
    cosine = torch.cat(cosine, 0)
    
    pos_cosine = cosine[gallery_labels == 1]
    pos_cosine, pos_gallery_idxs = topk_low_memory_(pos_cosine, args.topk)
    neg_cosine = cosine[gallery_labels == 0]
    neg_cosine, neg_gallery_idxs = topk_low_memory_(neg_cosine, args.topk)

    # using related patchs and topk to generate simialrity logit
    wsi_pred_list, query_idxs = [], []
    top_query_num = min(top_instance, len(query_patch_names))
    query_logits = pos_cosine.cuda().mean(0) - neg_cosine.cuda().mean(0)
    top_query_logits, top_query_idxs = query_logits.topk(top_query_num)
    
    # retrieval aggregation
    for i in range(top_query_num):
        wsi_pred, query_idx_i = top_query_logits[i], top_query_idxs[i]
        sim = query_feats[:, query_idx_i: query_idx_i + 1].t() @ query_feats
        sim_score, sim_idxs = sim[0].sort()
        num = (sim_score > args.related_thresh).sum()
        related_preds = query_logits[sim_idxs[-int(num):]]
        w = (sim_score[-int(num):] * args.temperature).softmax(0)
        wsi_pred = (w * related_preds).sum()
        wsi_pred_list.append(wsi_pred)
    wsi_pred = sum(wsi_pred_list) / len(wsi_pred_list)
   
    # vis matched patchs
    if vis_info != None:
        outd = os.path.join(vis_info['save_path'], vis_info['wsi_label'], vis_info['wsi_name'])
        save_matched_patchs([gallery_patch_names[i] for i in (gallery_labels == 0).nonzero()[:, 0]], \
            query_patch_names, neg_gallery_idxs, top_query_idxs, neg_cosine, outd, 'neg')
        save_matched_patchs([gallery_patch_names[i] for i in (gallery_labels == 1).nonzero()[:, 0]], \
            query_patch_names, pos_gallery_idxs, top_query_idxs, pos_cosine, outd, 'pos')
        save_matched_patchs(query_patch_names, query_patch_names, sim_idxs[-int(num):], \
            top_query_idxs, sim_score[-int(num):], outd, 'query') # for camelyon query top 1

    # patch pred for vis
    # wsi_size = None
    if wsi_size != None:
        #patch_pred = torch.zeros(wsi_size).cuda() + query_logits.min()
        patch_pred = torch.zeros(wsi_size).cuda() + 255
        for i, n in enumerate(query_patch_names):
            x, y = n.split('/')[-1].split('.')[0].split('_')
            try:
                patch_pred[int(y), int(x)] = query_logits[i]
            except:
                continue
    else:
        patch_pred = None

    return patch_pred, wsi_pred


def named_rlist_to_pydict(rlist):
    values = list(rlist)
    if rlist.names == rpy2.rinterface.NULL:
        names = []
    else:
        names = list(rlist.names)

    if len(values) != len(names):
        raise Exception("Number of names doesn't match number of values")

    return dict(zip(names, values))


def macro_value(l, n):
    out = []
    for i in range(len(l) // n):
        v = sum(l[i * n: i * n + n]) / n
        out.append(v)
    return out


def get_gallery_names_at_same_num(all_names, dataset_info, gallery_num):
    record = {}
    for n in all_names:
        lb = dataset_info[n]['wsi_label']
        if lb not in record:
            record[lb] = []
        record[lb].append(n)

    names = []
    for k, v in record.items():
        names.extend(v[:gallery_num])

    return names


def evaluate(args, val_only=False):
    auc_list, f1_list, acc_list, gallery_list = [], [], [], []
    aucroc = torchmetrics.AUROC(task='binary', num_classes=1)
    dataset_info = json.load(open(args.dataset_info))
    all_names = dataset_info.keys()
    
    for i in range(args.runs):
        # data split
        gallery_names, test_names, rest_names = [], [], []
        for n in all_names:
            # split fixed test set
            if dataset_info[n]['fixed_test_set']:
                test_names.append(n)
            # split gallery set (prompt inputs)
            else:
                if 'pos_patch_num' in dataset_info[n]:
                    pn = dataset_info[n]['pos_patch_num']
                    if pn >= 1000 and pn < 3000:
                        gallery_names.append(n)

        # shuffle gallery till each run is different
        while True:
            random.shuffle(gallery_names)
            gallery_i = gallery_names[:args.gallery_num]

            gallery_i.sort()
            if gallery_i not in gallery_list:
                gallery_list.append(gallery_i)
                gallery_names = gallery_i
                break

        # split val set out of gallery and test set
        for n in all_names:
            if n not in gallery_names and dataset_info[n]['fixed_test_set'] == False:
                rest_names.append(n)
        random.shuffle(rest_names)
        val_names = rest_names[:args.val_num]

        # split test set by ratio, if no fixed test set
        if len(test_names) == 0:
            test_names = rest_names[-args.test_num:]
            if args.test_num + args.val_num > len(rest_names):
                print('wrong split size !!!')

        # load gallery
        gallery_feats, gallery_patch_names, gallery_labels = [], [], []
        for n in gallery_names:
            gallery_n = np.load(os.path.join(args.dump_features, n + '.npy'), allow_pickle=True).item()
            gallery_patch_names = gallery_patch_names + gallery_n['patch_names']
            gallery_feats.append(gallery_n['features'])
            pl = copy.deepcopy(gallery_n['patch_labels'])
            gallery_labels.append(pl)
        
        gallery_feats = torch.tensor(np.concatenate(gallery_feats, 0)).cuda()
        gallery_labels = torch.tensor(np.concatenate(gallery_labels, 0)).cuda().long()

        # predict query
        val_preds, test_preds, val_labels, test_labels = [], [], [], []
        wsi_suffix = os.listdir(args.wsi_path)[0].split('.')[-1]
        all_query_names = val_names if val_only else val_names + test_names
        for n in all_query_names:
            query_n = np.load(os.path.join(args.dump_features, n + '.npy'), allow_pickle=True).item()
            query_feats = torch.tensor(query_n['features']).cuda()
            query_patch_names = query_n['patch_names']
            label = query_n['wsi_label']
            
            wsi_path = os.path.join(args.wsi_path, n + '.' + wsi_suffix)
            if os.path.exists(wsi_path):
                wsi = openslide.OpenSlide(wsi_path)
                size = (wsi.level_dimensions[0][1] // args.patch_scale, wsi.level_dimensions[0][0] // args.patch_scale)
            else:
                size = None
            
            vis_info = None
            patch_pred, wsi_pred = INC(args, gallery_feats, gallery_labels, gallery_patch_names, \
                query_feats, query_patch_names, size, args.top_instance, vis_info)

            if patch_pred != None and args.vis_path != '':
                os.makedirs(args.vis_path, exist_ok=True)
                np.save(os.path.join(args.vis_path, n + '.npy'), patch_pred.cpu().numpy())
            
            if n in val_names:
                val_preds.append(wsi_pred)
                val_labels.append(torch.tensor(int(label)))
            else:
                test_preds.append(wsi_pred)
                test_labels.append(torch.tensor(int(label)))

        # Evaluate on the val set to make sure qualified results for application
        # Val set also guidances to select prediction threshod
        # F1 is not influenced by pos-neg ratio while accuracy does, thus we select threshold by f1
        val_preds = torch.stack(val_preds).cpu()
        val_labels = torch.stack(val_labels)
        val_auc = aucroc(val_preds, val_labels).item()
        if not val_only:
            test_preds = torch.stack(test_preds).cpu()
            test_labels = torch.stack(test_labels)
            test_auc = aucroc(test_preds, test_labels).item()
       
        precisions, recalls, thresholds = precision_recall_curve(val_labels.numpy(), val_preds.numpy())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
        best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
        thresh = thresholds[best_f1_score_index]

        if val_only:
            preds = val_preds
            thresh_preds = (val_preds > thresh).float()
            labels = val_labels
        else:
            preds = test_preds
            thresh_preds = (test_preds > thresh).float()
            labels = test_labels

        y = robjects.IntVector(labels.cpu().tolist())
        x = robjects.FloatVector(preds.cpu().tolist())
        res = proc.roc(y, x, ci=True, direction = "<")
        res = named_rlist_to_pydict(res)
        res = (round(res['ci'][1], 4), (round(res['ci'][0], 4), round(res['ci'][2], 4)))

        acc = ((thresh_preds == labels).sum() / labels.shape[0]).cpu().item()
        rec = ((thresh_preds * labels).sum() / labels.sum()).cpu().item()
        pre = ((thresh_preds * labels).sum() / thresh_preds.sum()).cpu().item()
        auc = aucroc(preds, labels).item()
        f1 = rec * pre * 2 / (rec + pre)
        auc_list.append(auc)
        f1_list.append(f1)
        acc_list.append(acc)
        if not val_only:
            print(' val auc:', round(val_auc, 4), ', test auc:', round(auc, 4), \
                ', val f1: ', round(best_f1_score, 4), ', test f1: ', round(f1, 4), \
                ', test acc: ', round(acc, 4), ', test auc with ci:', res)

    auc_mean = np.array(auc_list).mean()
    auc_std = np.array(macro_value(auc_list, 1)).std()
    f1_mean = np.array(f1_list).mean()
    f1_std = np.array(macro_value(f1_list, 1)).std()
    acc_mean = np.array(acc_list).mean()
    acc_std = np.array(macro_value(acc_list, 1)).std()
    print('auc mean:', round(auc_mean, 4), ', auc std:', round(auc_std, 4), \
        ', f1 mean:', round(f1_mean, 4), ', f1 std:', round(f1_std, 4), \
        ', acc mean:', round(acc_mean, 4), ', acc std:', round(acc_std, 4))
    return round(auc_mean, 4)


def evaluate_baseline(args, mode):
    auc_list, f1_list, acc_list, gallery_list = [], [], [], []
    aucroc = torchmetrics.AUROC(task='binary', num_classes=1)
    dataset_info = json.load(open(args.dataset_info))
    all_names = dataset_info.keys()
    
    for i in range(args.runs):

        # data split
        gallery_names, test_names, rest_names = [], [], []
        for n in all_names:
            # split fixed test set
            if dataset_info[n]['fixed_test_set']:
                test_names.append(n)
            # split gallery set (prompt inputs)
            else:
                if 'pos_patch_num' in dataset_info[n]:
                    pn = dataset_info[n]['pos_patch_num']
                    if pn >= 1000 and pn < 3000:
                        gallery_names.append(n)

        # shuffle gallery till each run is different
        while True:
            random.shuffle(gallery_names)
            gallery_i = gallery_names[:args.gallery_num]
            gallery_i.sort()
            if gallery_i not in gallery_list:
                gallery_list.append(gallery_i)
                gallery_names = gallery_i
                break

        # split val set out of gallery and test set
        for n in all_names:
            if n not in gallery_names and dataset_info[n]['fixed_test_set'] == False:
                rest_names.append(n)
        random.shuffle(rest_names)
        val_names = rest_names[:args.val_num]

        # split test set by ratio, if no fixed test set
        fixed_test = True
        if len(test_names) == 0:
            fixed_test = False
            test_names = rest_names[-args.test_num:]
            if args.test_num + args.val_num > len(rest_names):
                print('wrong split size !!!')
        
        # load gallery
        gallery_feats, gallery_labels = [], []
        if 'prototype' in mode:
            gallery_labels = [1, 0]
            pos_feats, neg_feats = [], []

            for n in gallery_names:
                gallery_n = np.load(os.path.join(args.dump_features, n + '.npy'), allow_pickle=True).item()
                pl = gallery_n['patch_labels']
                pos_feats.append(gallery_n['features'][pl == 1])
                neg_feats.append(gallery_n['features'][pl == 0])
            pos_feats = np.concatenate(pos_feats, 0)
            neg_feats = np.concatenate(neg_feats, 0)
            
            if 'simple_shot' in mode:
                mean_feat = np.concatenate([pos_feats, neg_feats], 0).mean(0)
                pos_feats -= mean_feat
                pos_feats = pos_feats.mean(0, keepdims=True)
                pos_feats = pos_feats / np.linalg.norm(pos_feats, 2, 1, keepdims=True)
                neg_feats -= mean_feat
                neg_feats = neg_feats.mean(0, keepdims=True)
                neg_feats = neg_feats / np.linalg.norm(neg_feats, 2, 1, keepdims=True)
                gallery_feats = [pos_feats, neg_feats]
            else:
                gallery_feats = [pos_feats.mean(0, keepdims=True), neg_feats.mean(0, keepdims=True)]

        elif 'knn' in mode:
            names = rest_names[args.val_num:] + gallery_names if fixed_test == True else val_names + gallery_names
            random.shuffle(names)
            for n in names: # knn need pos and neg wsi, use train set
                gallery_n = np.load(os.path.join(args.dump_features, n + '.npy'), allow_pickle=True).item()

                # balanced gallery for knn to aviod same label at few-shot
                if args.gallery_num < 20 and gallery_n['wsi_label'] == 0 and (np.array(gallery_labels) == 0).sum() == args.gallery_num // 2:
                    continue
                if args.gallery_num < 20 and gallery_n['wsi_label'] == 1 and (np.array(gallery_labels) == 1).sum() == args.gallery_num // 2:
                    continue
                gallery_labels.append(gallery_n['wsi_label'])
            
                if 'mean' in mode:
                    gallery_feats.append(gallery_n['features'].mean(0, keepdims=True))
                elif 'max' in mode:
                    gallery_feats.append(gallery_n['features'].max(0, keepdims=True))
                else:
                    print('false eval mode')
        
        else:
            print('false eval mode')

        gallery_feats = torch.tensor(np.concatenate(gallery_feats, 0)).cuda()
        gallery_labels = torch.tensor(gallery_labels).cuda()

        # predict query
        val_preds, test_preds, val_labels, test_labels = [], [], [], []
        all_query_names = val_names + test_names
        for n in all_query_names:
            query_n = np.load(os.path.join(args.dump_features, n + '.npy'), allow_pickle=True).item()
            query_feats = torch.tensor(query_n['features']).cuda()
            query_patch_names = query_n['patch_names']
            label = query_n['wsi_label']

            if 'prototype' in mode:
                if 'simple_shot' in mode:
                    query_feats -= torch.tensor(mean_feat).cuda()
                    query_feats = query_feats / torch.linalg.norm(query_feats, 2, 1, keepdims=True)

                topk = min(args.top_instance, query_feats.shape[0])
                prob = query_feats @ gallery_feats[0]
                wsi_pred = prob.topk(topk)[0].mean()

                if args.vis_path != '':
                    wsi_suffix = os.listdir(args.wsi_path)[0].split('.')[-1]
                    wsi_path = os.path.join(args.wsi_path, n + '.' + wsi_suffix)
                    wsi = openslide.OpenSlide(wsi_path)
                    size = (wsi.level_dimensions[0][1] // args.patch_scale, wsi.level_dimensions[0][0] // args.patch_scale)
                    patch_pred = torch.zeros(size).cuda() + 255
                    for i, pn in enumerate(query_patch_names):
                        x, y = pn.split('/')[-1].split('.')[0].split('_')
                        try:
                            patch_pred[int(y), int(x)] = prob[i]
                        except:
                            continue
                    os.makedirs(args.vis_path, exist_ok=True)
                    np.save(os.path.join(args.vis_path, n + '.npy'), patch_pred.cpu().numpy())

            elif 'knn' in mode:
                if 'mean' in mode:
                    query_feats = query_feats.mean(0)
                elif 'max' in mode:
                    query_feats = query_feats.max(0)[0]
                else:
                    print('false eval mode')
               
                pos_gallery_feats, neg_gallery_feats = gallery_feats[gallery_labels == 1], gallery_feats[gallery_labels == 0]
                wsi_pred = (pos_gallery_feats @ query_feats).topk(min(5, pos_gallery_feats.shape[0]))[0].mean() - \
                        (neg_gallery_feats @ query_feats).topk(min(5, neg_gallery_feats.shape[0]))[0].mean()
                
            else:
                print('false eval mode')

            if n in val_names:
                val_preds.append(wsi_pred)
                val_labels.append(torch.tensor(int(label)))
            else:
                test_preds.append(wsi_pred)
                test_labels.append(torch.tensor(int(label)))

        # search a threshold to predict label on val set for fair comparision with INC
        # The upper bound of KNN prob with threshold is higher than binary prediction
        val_preds = torch.stack(val_preds).cpu()
        val_labels = torch.stack(val_labels)
        val_auc = aucroc(val_preds, val_labels).item()
        test_preds = torch.stack(test_preds).cpu()
        test_labels = torch.stack(test_labels)
       
        precisions, recalls, thresholds = precision_recall_curve(val_labels.numpy(), val_preds.numpy())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
        best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
        thresh = thresholds[best_f1_score_index]

        preds = test_preds
        thresh_preds = (test_preds > thresh).float()
        labels = test_labels
        acc = ((thresh_preds == labels).sum() / labels.shape[0]).cpu().item()
        rec = ((thresh_preds * labels).sum() / labels.sum()).cpu().item()
        pre = ((thresh_preds * labels).sum() / thresh_preds.sum()).cpu().item()
        auc = aucroc(preds, labels).item()
        f1 = rec * pre * 2 / (rec + pre) if rec + pre > 0 else 0
        auc_list.append(auc)
        f1_list.append(f1)
        acc_list.append(acc)

    auc_mean = np.array(auc_list).mean()
    auc_std = np.array(auc_list).std()
    f1_mean = np.array(f1_list).mean()
    f1_std = np.array(f1_list).std()
    acc_mean = np.array(acc_list).mean()
    acc_std = np.array(acc_list).std()
    print('auc mean:', round(auc_mean, 4), ', auc std:', round(auc_std, 4), \
        ', f1 mean:', round(f1_mean, 4), ', f1 std:', round(f1_std, 4), \
        ', acc mean:', round(acc_mean, 4), ', acc std:', round(acc_std, 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Few-Shot Lymph Node Metastasis Classification Meets High Performance \
            on Whole Slide Images via the Informative Non-Parametric Classifier')
    parser.add_argument('--mode', default='default', type=str, help=" eval: inference and evaluation, \
            eval_baseline: evaluation of baseline methods, default: update feature and evaluate, \
            search: hyper-parameter search on val set.")

    # hyper-params
    parser.add_argument('--topk', default=40, type=int, help='Number of top patchs to take')
    parser.add_argument('--top_instance', default=1, type=int, help='Number of top patchs to take')
    parser.add_argument('--temperature', default=10, type=float, help='Temperature for sample reweights')
    parser.add_argument('--related_thresh', default=0.88, type=float, help='cosine similarity threshold to select related patchs')
    parser.add_argument('--gallery_num', default=8, type=int, help='number of wsi for init gallery')
    parser.add_argument('--multiple_num', type=int, nargs='+', default=None, help='multi gallery num')
    
    # dataset information and settings
    parser.add_argument('--raw_feature_path', default='/path/to/raw_feature/', type=str)
    parser.add_argument('--wsi_path', default='/path/to/WSI/', type=str)
    parser.add_argument('--dump_features', default=None, help='Path where to save features')
    parser.add_argument('--vis_path', default='', help='Path where to save heatmap')
    parser.add_argument('--dataset_info', default='/path/to/data_list_gt_and_split', type=str, help='json file recording dataset info')
    parser.add_argument('--patch_scale', default=512, type=int, help='one patch indicates 512 pixels on slide level 0')

    # test settings
    parser.add_argument('--seed', default=1024, type=int, help='for the reproduce of data split')
    parser.add_argument('--runs', default=5, type=int, help='number of test times')
    parser.add_argument('--val_num', default=100, type=int, help='number of validation WSIs')
    parser.add_argument('--test_num', default=129, type=int, help='number of test WSIs')
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.dump_features, exist_ok=True)

    # collect features
    if args.mode == 'default':
        feature_processor(args)

    # load gallery and query features; run for evaluation
    if args.mode == 'eval' or args.mode == 'default':
        num = [args.gallery_num] if args.multiple_num == None else args.multiple_num
        for p in num:
            print('gallery_num: ' + str(p))
            random.seed(args.seed)
            args.gallery_num = p
            auc = evaluate(args)
    
    if args.mode == 'eval_baseline':
        num = [args.gallery_num] if args.multiple_num == None else args.multiple_num
        for p in num:
            print('gallery_num: ' + str(p))
            args.gallery_num = p
            if args.gallery_num > 1:
                print('mode: knn_mean, gallery ' + str(args.gallery_num))
                random.seed(args.seed)
                evaluate_baseline(args, 'knn_mean')
            
                print('mode: knn_max, gallery ' + str(args.gallery_num))
                random.seed(args.seed)
                evaluate_baseline(args, 'knn_max')
            
            print('mode: prototype, gallery ' + str(args.gallery_num))
            random.seed(args.seed)
            evaluate_baseline(args, 'prototype')

            print('mode: prototype_simple_shot, gallery ' + str(args.gallery_num))
            random.seed(args.seed)
            evaluate_baseline(args, 'prototype_simple_shot')

    # grid search
    if args.mode == 'search': 
        print('eval and test with the default params ...')
        res = evaluate(args)

        print('start param search via val set ...')
        v, t = 0, 0
        for p in [5, 10, 20, 30, 40]:
            print('searching topk, param: ' + str(p))
            random.seed(args.seed) # validate params without influence from sampling
            args.topk = p
            res = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.topk = t
        print('params: topk, searched threshold: ' + str(t) + ', mean:' + str(v))
            
        v, t = 0, 0
        for p in [0.88, 0.89, 0.9, 0.91, 0.92]:
            print('searching related_thresh, param: ' + str(p))
            random.seed(args.seed)
            args.related_thresh = p
            res = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.related_thresh = t
        print('params: related_thresh, searched threshold: ' + str(t) + ', mean:' + str(v))
            
        v, t = 0, 0
        for p in [5, 10, 20, 30, 40]:
            print('searching temperature, param: ' + str(p))
            random.seed(args.seed)
            args.temperature = p
            res = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.temperature = t
        print('params: temperature, searched threshold: ' + str(t) + ', mean:' + str(v))
        
        # eval with searched params and test influence of gallery number
        print(args)
        v, t = 0, 0
        num = [args.gallery_num] if args.multiple_num == None else args.multiple_num
        for p in num:
            print('gallery_num: ' + str(p))
            random.seed(args.seed)
            args.gallery_num = p
            res = evaluate(args)
