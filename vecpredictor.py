# -*- coding: utf-8 -*-
import sys
import os

import numpy as np
from numpy.linalg import norm
from typing import Any, Dict
import pickle
import cv2
import json
import faiss
from tqdm import tqdm
import torchvision.transforms as T
# from timm import create_model as create
from PIL import Image
import logging
logger = logging.getLogger(__name__)

import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from pathlib import Path
from timm.models import create_model
import utils
import modeling_pretrain
from datasets import DataAugmentationForVideoMAE
from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from decord import VideoReader, cpu
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator

class DataAugmentationForVideoMAE(object):
    def __init__(self, window_size, mask_type="tube"):
        self.input_mean = [0.485, 0.456, 0.406] # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225] # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        # self.train_augmentation = GroupCenterCrop(224)
        self.train_augmentation = GroupResize(224)
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                window_size, 0.0
            )

    def __call__(self, images):
        process_data , _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def createInstance(classifier):
    vecpred=Vecpredictor(classifier)
    return vecpred

class Vecpredictor():
    def __init__(self) -> None:
        super().__init__()

    def loadModel(self, model_dir_path: os.PathLike="lib/weights", test_mode=True) -> bool:
        #模型和配置所在文件夹
        if not os.path.exists(model_dir_path):
            logger.error("'{}'does not exist".format(model_dir_path))
            return False
        
        self.model_dir = model_dir_path
        
        #配置文件路径（按照预设文件结构）
        json_file=os.path.join(model_dir_path,"predictor.json")
        if not os.path.exists(json_file):
            logger.error("'{}'does not exist".format(json_file))
            return False
        
        #导入为config
        with open(json_file,'r',encoding='utf-8') as pf:
            self.config=json.load(pf)
            pf.close()

        if test_mode:        
            #判断检索库文件夹是否存在
            if not self.loadBank(os.path.join(model_dir_path, "vector_model")):
                return False
            with open(os.path.join(self.bank_dir, "id_map.pkl"), "rb") as fd:
                self.id_map = pickle.load(fd)

        self.return_k = self.config['return_k']
        self.device = "cpu" if self.config["gpus"] == "-1" else "cuda:" + self.config["gpus"]

        #分类模型所在路径
        classifier_model=os.path.join(model_dir_path, self.config["rec_inference_model"])

        if not os.path.exists(classifier_model):
            logger.error("'{}'does not exist".format(classifier_model))
            return False

        # model = create('vit_tiny_r_s16_p8_224',pretrained=True, num_classes=1000)
        model = create_model(
            "pretrain_videomae_base_patch16_224",
            pretrained=False,
            drop_path_rate=0.0,
            drop_block_rate=None,
            decoder_depth=4
        )

        checkpoint = torch.load(classifier_model, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(self.device)
        model.eval()

        self.rec_predictor = model.encoder

        self.patch_size = model.encoder.patch_embed.patch_size
        self.input_size = 224
        self.num_frames = 16
        self.window_size = (
            self.num_frames // 2, 
            self.input_size // self.patch_size[0], 
            self.input_size // self.patch_size[1]
        )

        return True
    
    def loadBank(self, bank_dir_path: os.PathLike="lib/weights/vector_model") -> bool:
        if not os.path.exists(bank_dir_path):
            logger.error("'{}'does not exist".format(bank_dir_path))
            return False
        
        self.bank_dir=bank_dir_path
        self.Searcher = faiss.read_index(
            os.path.join(bank_dir_path, "vector.index"))
                
        return True
    
    def loadVideo(self, video_path: os.PathLike, part_info=None):
        with open(video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu(0))
        
        video_frames = vr._num_frame

        if video_frames <= self.num_frames:
            frame_id_list = [i for i in range(video_frames)] + [j for j in range(self.num_frames - video_frames)]
        else:
            step = video_frames / (self.num_frames)
            frame_id_list = [int(i * step) for i in range(self.num_frames)]

        video_data = vr.get_batch(frame_id_list).asnumpy()  # shape(16, 1920, 2560, 3)

        if part_info:
            xmin, ymin, xmax, ymax = part_info
        else:
            xmin, ymin = 0, 0
            _, ymax, xmax, _ = video_data.shape

        video = [Image.fromarray(video_data[vid, ymin: ymax, xmin: xmax, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        transforms = DataAugmentationForVideoMAE(self.window_size)

        video, bool_masked_pos = transforms((video, None)) # T*C,H,W
        video = video.view((16, 3) + video.size()[-2:]).transpose(0,1) # T*C,H,W -> T,C,H,W -> C,T,H,W
        bool_masked_pos = torch.from_numpy(bool_masked_pos)

        return video, bool_masked_pos
    
    def loadVideo(self, frames, part_info=None):
        num_frame, H, W, _ = frames.shape

        if num_frame <= self.num_frames:
            frame_id_list = [i for i in range(num_frame)] + [j for j in range(self.num_frames - num_frame)]
        else:
            step = num_frame / (self.num_frames)
            frame_id_list = [int(i * step) for i in range(self.num_frames)]

        # np.array(32, h, w, c) RGB
        if part_info:
            xmin, ymin, xmax, ymax = part_info
        else:
            xmin, ymin = 0, 0
            ymax, xmax = H, W

        video = [Image.fromarray(frames[vid, ymin: ymax, xmin: xmax, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        transforms = DataAugmentationForVideoMAE(self.window_size)

        video, bool_masked_pos = transforms((video, None)) # T*C,H,W
        video = video.view((16, 3) + video.size()[-2:]).transpose(0,1) # T*C,H,W -> T,C,H,W -> C,T,H,W
        bool_masked_pos = torch.from_numpy(bool_masked_pos)

        return video, bool_masked_pos
    
    def predict(self, video, part_info=None):
        preds = {}
        video, bool_masked_pos = self.loadVideo(video, part_info)

        with torch.no_grad():
            video = video.unsqueeze(0)
            bool_masked_pos = bool_masked_pos.unsqueeze(0)  # (1, 1568)

            video = video.to(self.device, non_blocking=True)  # (1, 3, 16, 224, 224)
            bool_masked_pos = bool_masked_pos.to(self.device, non_blocking=True).flatten(1).to(torch.bool)

            # print(video.shape)
            output_tensor = self.rec_predictor(video, bool_masked_pos)  
            # base-encoder(1, 1568, 768)
            # print(output_tensor.shape)
            output_tensor = output_tensor.view(1, -1)

            rec_tensor = output_tensor.cpu().detach().numpy()
            scores, docs = self.Searcher.search(rec_tensor, self.return_k)

            tar_tensor = self.Searcher.reconstruct(int(docs[0][0]))

            cos_sim = scores[0][0] / (norm(rec_tensor) * norm(tar_tensor))
            # print(cos_sim)

            if scores[0][0] >= self.config["score_thres"]:
                preds["error"]=None
                #这里把partid设为info，但是不清楚具体应用含义，后续可修改
                preds["obj_class"] = self.id_map[docs[0][0]].split()[1]
                preds["score"] = scores[0][0]
                preds["cosine"] = cos_sim
            else:
                #error这里具体内容可以修改
                preds["error"]="can not figure out image class"
                #这里把partid设为info，但是不清楚具体应用含义，后续可修改
                preds["obj_class"] =None
                preds["score"] = scores[0][0]          
                preds["cosine"] = 0.0  
            
            return preds
  
    def saveBank(self,index,ids) -> bool:
        #index,ids=self.genRetrievalBank(self.config["image_root"])
        if self.config["dist_type"] == "hamming":
            faiss.write_index_binary(
                index, os.path.join(self.bank_dir, "vector.index"))
        else:
            faiss.write_index(
                index, os.path.join(self.bank_dir, "vector.index"))

        with open(os.path.join(self.bank_dir, "id_map.pkl"), 'wb') as fd:
            pickle.dump(ids, fd)
  
    def genRetrievalBank(self, video_folder, operation_method='new', part_info=None):
        if not os.path.exists(video_folder):
            logger.error("'{}'does not exist".format(video_folder))
        
        gallery_videos, gallery_docs = self.split_datafile(video_folder)

        assert operation_method in [
            "new", "remove", "append"
        ], "Only append, remove and new operation are supported"

        if operation_method != "remove":
            gallery_features = self._extract_features(gallery_videos, part_info=part_info)

        index, ids = None, None
        if operation_method in ["remove", "append"]:
            # if remove or append, load vector.index and id_map.pkl
            index, ids = self._load_index(self.config)
            index_method = self.config.get("index_method", "IVF")
        else:
            index_method, index, ids = self._create_index()
        if index_method == "IVF":
            logger.warning(
                "The Flat method dose not support 'remove' operation")
            
        if operation_method != "remove":
            # calculate id for new data
            index, ids = self._add_gallery(index, ids, gallery_features, gallery_docs,operation_method)
        else:
            if index_method == "IVF":
                raise RuntimeError(
                    "The index_method: Flat dose not support 'remove' operation"
                )
            # remove ids in id_map, remove index data in faiss index
            index, ids = self._rm_id_in_galllery(index, ids, gallery_docs)

        # store faiss index file and id_map file
        #下面这部分放到savebank中
        self.saveBank(index, ids)

        return index,ids
        
    def _create_index(self):
        if not hasattr(self, 'class_attribute'):
            self.bank_dir = os.path.join(self.model_dir, "vector_model")
        if not os.path.exists(self.bank_dir):
            os.makedirs(self.bank_dir, exist_ok=True)
        index_method = self.config.get("index_method", "HNSW32")

        # if IVF method, cal ivf number automaticlly
        if index_method == "IVF":
            index_method = index_method + str(1500) + ",Flat"        #  更改检索算法的时候记得修改这里

        # for binary index, add B at head of index_method
        if self.config["dist_type"] == "hamming":
            index_method = "B" + index_method

        # dist_type
        dist_type = faiss.METRIC_INNER_PRODUCT if self.config[
                                                      "dist_type"] == "IP" else faiss.METRIC_L2

        # build index
        if self.config["dist_type"] == "hamming":
            index = faiss.index_binary_factory(self.config["embedding_size"],
                                               index_method)
        else:
            index = faiss.index_factory(self.config["embedding_size"],
                                        index_method, dist_type)
            index = faiss.IndexIDMap2(index)
        ids = {}
        return index_method, index, ids
            
    def _extract_features(self, gallery_videos, device="cuda:0", part_info=None):
        if self.config["dist_type"] == "hamming":
            gallery_features = np.zeros(
                [len(gallery_videos), self.config['embedding_size'] // 8],
                dtype=np.uint8)
        else:
            gallery_features = np.zeros(
                [len(gallery_videos), self.config['embedding_size']],
                dtype=np.float32)
            
        batch_size = self.config['batch_size']
        for i, video_file in enumerate(tqdm(gallery_videos)):
            video, bool_masked_pos = self.loadVideo(video_file, part_info)

            video = video.unsqueeze(0).to(device, non_blocking=True)
            bool_masked_pos = bool_masked_pos.unsqueeze(0).to(device, non_blocking=True).flatten(1).to(torch.bool)

            output_tensor = self.rec_predictor(video, bool_masked_pos)
            output_tensor = output_tensor.view(1, -1)

            rec_results = output_tensor.cpu().detach().numpy()
            gallery_features[i - batch_size + 1: i + 1, :] = rec_results
        
        return gallery_features    
    
    def _add_gallery(self, index, ids, gallery_features, gallery_docs,operation_method):
        start_id = max(ids.keys()) + 1 if ids else 0
        ids_now = (
            np.arange(0, len(gallery_docs)) + start_id).astype(np.int64)

        # only train when new index file
        if operation_method == "new":
            if self.config["dist_type"] == "hamming":
                index.add(gallery_features)
            else:
                index.train(gallery_features)

        if not self.config["dist_type"] == "hamming":
            index.add_with_ids(gallery_features, ids_now)

        for i, d in zip(list(ids_now), gallery_docs):
            ids[i] = d
        return index, ids

    def _load_index(self):
        assert os.path.join(
            self.config["index_dir"], "vector.index"
        ), "The vector.index dose not exist in {} when 'index_operation' is not None".format(
            self.config["index_dir"])
        
        assert os.path.join(
            self.config["index_dir"], "id_map.pkl"
        ), "The id_map.pkl dose not exist in {} when 'index_operation' is not None".format(
            self.config["index_dir"])
        
        index = faiss.read_index(
            os.path.join(self.config["index_dir"], "vector.index"))
        
        with open(os.path.join(self.config["index_dir"], "id_map.pkl"), 'rb') as fd:
            ids = pickle.load(fd)

        assert index.ntotal == len(ids.keys(
        )), "data number in index is not equal in in id_map"
        
        return index, ids
    
    def _save_gallery(self, index, ids):
        if self.config["dist_type"] == "hamming":
            faiss.write_index_binary(
                index, os.path.join(self.config["index_dir"], "vector.index"))
        else:
            faiss.write_index(
                index, os.path.join(self.config["index_dir"], "vector.index"))

        with open(os.path.join(self.config["index_dir"], "id_map.pkl"), 'wb') as fd:
            pickle.dump(ids, fd)

    def append_self(self, results, shape):
        results.append({
            "class_id": 0,
            "score": 1.0,
            "bbox":
            np.array([0, 0, shape[1], shape[0]]),  # xmin, ymin, xmax, ymax
            "label_name": "foreground",
        })
        return results
    
    def nms_to_rec_results(self, results, thresh=0.1):
        filtered_results = []
        x1 = np.array([r["bbox"][0] for r in results]).astype("float32")
        y1 = np.array([r["bbox"][1] for r in results]).astype("float32")
        x2 = np.array([r["bbox"][2] for r in results]).astype("float32")
        y2 = np.array([r["bbox"][3] for r in results]).astype("float32")
        scores = np.array([r["rec_scores"] for r in results])

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        while order.size > 0:
            i = order[0]
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
            filtered_results.append(results[i])

        return filtered_results    

    def split_datafile(self, datafile):
        gallery_images = []
        gallery_docs = []
        filedir = os.listdir(datafile)
        for file_dir in filedir:
            img_dir = os.path.join(datafile,file_dir)
            for img in os.listdir(img_dir):
                gallery_images.append(os.path.join(img_dir, img))
                gallery_docs.append(img + " " + file_dir)

        print(gallery_docs)

        return gallery_images, gallery_docs

# if __name__=="__main__":

#     classifier=AbstractPredictor()
#     img = cv2.imread(r"D:\lubo\315\data\data\train\0\img_0_0.jpg")
#     vecpred=createInstance(classifier)
#     if(vecpred.loadModel()):

#         output=vecpred.predict(img)
#         print(output)        
#         #vecpred.genRetrievalBank("D:/lubo/315/data/data/train")
#     #gallery_images, gallery_docs=split_datafile("D:/lubo/315/data/data/train")
