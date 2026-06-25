import torch
from torch import nn
import torch.nn.functional as F
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from config_local import *

@MODELS.register_module()
class SegEarthOV3Segmentation(BaseSegmentor):
    def __init__(self, classname_path,
                 device=torch.device('cuda'),
                 prob_thd=0.0,
                 bg_idx=0,
                 slide_stride=0,
                 slide_crop=0,
                 confidence_threshold=0.5,
                 use_sem_seg=True,
                 use_presence_score=True,
                 use_transformer_decoder=True,
                 **kwargs):
        super().__init__()
        
        self.device = device
        # Initialize SAM3 model
        model = build_sam3_image_model(
            bpe_path="./sam3/assets/bpe_simple_vocab_16e6.txt.gz", 
            checkpoint_path=SAM3_CHECKPOINT,
            device="cuda"
        )
        self.processor = Sam3Processor(model, confidence_threshold=confidence_threshold, device=device)
        self.query_words, self.query_idx = get_cls_idx(classname_path)
        self.num_cls = max(self.query_idx) + 1
        self.num_queries = len(self.query_idx)
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.confidence_threshold = confidence_threshold
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder

    def _inference_single_view(self, image):
        """Inference on a single PIL image or crop patch."""
        w, h = image.size
        seg_logits = torch.zeros((self.num_queries, h, w), device=self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = self.processor.set_image(image)
            
            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)

                if self.use_transformer_decoder:
                    if inference_state['masks_logits'].shape[0] > 0:
                        inst_len = inference_state['masks_logits'].shape[0]
                        for inst_id in range(inst_len):
                            instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                            instance_score = inference_state['object_score'][inst_id]
                            # instance_mask = inference_state['masks'][inst_id].squeeze()
                            
                            # Handle potential dimension mismatch if SAM3 output differs slightly
                            if instance_logits.shape != (h, w):
                                instance_logits = F.interpolate(
                                    instance_logits.view(1, 1, *instance_logits.shape), 
                                    size=(h, w), 
                                    mode='bilinear', 
                                    align_corners=False
                                ).squeeze()

                            seg_logits[query_idx] = torch.max(seg_logits[query_idx], instance_logits * instance_score)
                    
                if self.use_sem_seg:
                    semantic_logits = inference_state['semantic_mask_logits']
                    if semantic_logits.shape != (h, w):
                            semantic_logits = F.interpolate(
                                semantic_logits, 
                                size=(h, w), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze()
                    
                    seg_logits[query_idx] = torch.max(seg_logits[query_idx], semantic_logits)
                
                if self.use_presence_score:
                    seg_logits[query_idx] = seg_logits[query_idx] * inference_state["presence_score"]
                
        return seg_logits

    @staticmethod
    def _make_gaussian_kernel(h, w, device):
        """2D Gaussian weight kernel: center pixels get full weight, edges near-zero."""
        sigma_h, sigma_w = h / 4.0, w / 4.0
        y = torch.arange(h, device=device).float() - (h - 1) / 2.0
        x = torch.arange(w, device=device).float() - (w - 1) / 2.0
        kernel = torch.exp(-y[:, None] ** 2 / (2 * sigma_h ** 2)) * \
                 torch.exp(-x[None, :] ** 2 / (2 * sigma_w ** 2))
        return kernel  # (h, w)

    def slide_inference(self, image, stride, crop_size):
        """Inference by sliding-window with Gaussian-weighted overlap blending.

        Center pixels of each crop contribute with full weight; edge pixels are
        down-weighted exponentially, eliminating seam artifacts when patches overlap.
        """
        w_img, h_img = image.size

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size

        # Gaussian kernel created on GPU for crop inference, then moved to CPU.
        # Accumulation tensors live on CPU: a full-tile GPU tensor (queries × H × W)
        # exceeds VRAM on large tiles (e.g. 41 queries × 6000 × 6000 × 4B ≈ 5.9 GB).
        gaussian_kernel = self._make_gaussian_kernel(h_crop, w_crop, self.device).cpu()

        preds = torch.zeros((self.num_queries, h_img, w_img))  # CPU
        weight_mat = torch.zeros((h_img, w_img))               # CPU

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)

                # Adjust start to guarantee full crop size at image boundaries
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = image.crop((x1, y1, x2, y2))
                crop_seg_logit = self._inference_single_view(crop_img)  # GPU

                # Slice kernel for boundary crops smaller than nominal crop size
                actual_h, actual_w = y2 - y1, x2 - x1
                g = gaussian_kernel[:actual_h, :actual_w]

                # Move crop result to CPU immediately to free VRAM
                preds[:, y1:y2, x1:x2] += crop_seg_logit.cpu() * g.unsqueeze(0)
                weight_mat[y1:y2, x1:x2] += g

        assert (weight_mat == 0).sum() == 0, "Error: Sparse sliding window coverage."

        preds.div_(weight_mat.unsqueeze(0))
        torch.cuda.empty_cache()
        return preds.to(self.device)

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [data_sample.metainfo for data_sample in data_samples]
        else:
            # Fallback for meta info construction
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        
        for i, meta in enumerate(batch_img_metas):
            # Load original image to preserve details for SAM3
            image_path = meta.get('img_path')
            image = Image.open(image_path).convert('RGB')
            ori_shape = meta['ori_shape']

            # Determine inference mode
            if self.slide_crop > 0 and (self.slide_crop < image.size[0] or self.slide_crop < image.size[1]):
                seg_logits = self.slide_inference(image, self.slide_stride, self.slide_crop)
            else:
                seg_logits = self._inference_single_view(image)

            # Resize to original shape if necessary (e.g. padding effects)
            if seg_logits.shape[-2:] != ori_shape:
                seg_logits = F.interpolate(
                    seg_logits.unsqueeze(0), 
                    size=ori_shape, 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            # Post-processing: reduce per-synonym logits → per-class logits.
            # Avoid broadcasting [num_cls, num_queries, H, W] — too large for 16GB GPU.
            if self.num_cls != self.num_queries:
                torch.cuda.empty_cache()
                h, w = seg_logits.shape[-2:]
                result = torch.full((self.num_cls, h, w), float('-inf'), device=seg_logits.device)
                for cls_i in range(self.num_cls):
                    mask = (self.query_idx == cls_i)
                    if mask.any():
                        result[cls_i] = seg_logits[mask].max(0)[0]
                seg_logits = result

            seg_pred = torch.argmax(seg_logits, dim=0)
            
            # Apply probability threshold
            max_vals = seg_logits.max(0)[0]
            seg_pred[max_vals < self.prob_thd] = self.bg_idx

            data_samples[i].set_data({
                'seg_logits': PixelData(**{'data': seg_logits}),
                'pred_sem_seg': PixelData(**{'data': seg_pred.unsqueeze(0)})
            })
            
        return data_samples
    
    def _forward(data_samples):
            """
        """
    
    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """
    
    def extract_feat(self, inputs):
        """
        """
    
    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        names_i = [i.strip() for i in names_i]
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices