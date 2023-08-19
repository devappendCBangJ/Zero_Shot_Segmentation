# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt # 디버깅을 위한 시각화 ●

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from .modeling import Sam
from .predictor import SamPredictor
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)

# %%%%%%%%%%%%%%%%%%%%
# 디버깅을 위한 시각화 ●
# %%%%%%%%%%%%%%%%%%%%
def plot_all_mask_pred(pred_masks):
    plt.figure(figsize=(8, 8))
    img_iter = 0
    for idx, pm in enumerate(pred_masks):
        plot_idx = img_iter % (6 * 6)
        plt.subplot(6, 6, plot_idx + 1)
        plt.title(f'pred_mask_{img_iter}', fontsize=7)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(pm.cpu())
        plt.axis("off")
        img_iter += 1
        if plot_idx == 35:
            plt.figure(figsize=(8, 8))
    plt.show()
# %%%%%%%%%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%
# 디버깅을 위한 시각화 ●
# %%%%%%%%%%%%%%%%%%%%
# plot_all_mask_pred(data["masks"])
# plot_all_mask_pred(data["masks"] > self.predictor.model.mask_threshold)
# %%%%%%%%%%%%%%%%%%%%

class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
        sam_mask_size_list = [0, 1, 2]
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

        # %%%%%%%%%%%%%%%%%%%%
        # SAM Mask Size Setting ●
        # %%%%%%%%%%%%%%%%%%%%
        self.sam_mask_size_list = sam_mask_size_list

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(image)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        """
        orig_size shape : (2)
        orig_size : (428, 640)
        """
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )
        """
        crop_boxes shape : (1, 4)
        crop_boxes : [[0, 0, 640, 428]]
        """
        """
        layer_idxs shape : (1)
        layer_idxs : [0]
        """

        # Iterate over image crops
        data = MaskData()
        """
        data : <mobile_sam.utils.amg.MaskData object at 0x7f86820c46a0>
        """
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            """
            crop_data shape : -> _stats : dict 6 -> iou_preds : (51), points : (51, 2), stability_score : (51), boxes : (51, 4), rles : (51), crop_boxes : (51, 4)
            crop_data : <mobile_sam.utils.amg.MaskData object at 0x7f8594bc5fa0>
            """
            data.cat(crop_data)

        # Remove duplicate masks between crops (Crop들 간의 NMS)
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        """
        crop_box shape : (4)
        crop_box : [0, 0, 640, 428]
        """
        """
        image.shape : (428, 640, 3)
        image : [[[ 87  61  36],  [123 110  66],  [158 135  81],  ...,  [ 91  38   4],  [ 95  39   4],  [ 99  39   3]],, [[ 57  45  23],  [103  74  40],  [143 126  80],  ...,  [ 94  40   6],  [ 97  41   6],  [102  42   6]],, [[ 49  41  28],  [ 75  50  30],  [121  99  60],  ...,  [ 96  42   8],  [ 99  40   6],  [101  41   5]],, ...,, [[ 97  54  61],  [ 99  42  49],  [ 96  32  33],  ...,  [ 86 111 133],  [ 88 116 138],  [ 89 119 143]],, [[ 99  47  60],  [ 95  42  50],  [ 93  39  39],  ...,  [ 92 118 141],  [ 92 120 142],  [ 88 120 141]],, [[ 92  60  71],  [ 83  56  65],  [ 79  46  53],  ...,  [ 90 121 142],  [ 86 119 138],  [ 83 116 135]]]
        """
        cropped_im = image[y0:y1, x0:x1, :]
        """
        cropped_im.shape : (428, 640, 3)
        cropped_im : [[[ 87  61  36],  [123 110  66],  [158 135  81],  ...,  [ 91  38   4],  [ 95  39   4],  [ 99  39   3]],, [[ 57  45  23],  [103  74  40],  [143 126  80],  ...,  [ 94  40   6],  [ 97  41   6],  [102  42   6]],, [[ 49  41  28],  [ 75  50  30],  [121  99  60],  ...,  [ 96  42   8],  [ 99  40   6],  [101  41   5]],, ...,, [[ 97  54  61],  [ 99  42  49],  [ 96  32  33],  ...,  [ 86 111 133],  [ 88 116 138],  [ 89 119 143]],, [[ 99  47  60],  [ 95  42  50],  [ 93  39  39],  ...,  [ 92 118 141],  [ 92 120 142],  [ 88 120 141]],, [[ 92  60  71],  [ 83  56  65],  [ 79  46  53],  ...,  [ 90 121 142],  [ 86 119 138],  [ 83 116 135]]]
        """
        cropped_im_size = cropped_im.shape[:2]
        """
        cropped_im_size shape : (2)
        cropped_im_size : (428, 640)
        """
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        """
        points_scale.shape : (1, 2)
        points_scale : [[640 428]]
        """
        """
        crop_layer_idx shape : int
        crop_layer_idx : 0
        """
        points_for_image = self.point_grids[crop_layer_idx] * points_scale
        """
        self.point_grids[0].shape : (1024, 2)
        self.point_grids[0] : [[0.015625 0.015625], [0.046875 0.015625], [0.078125 0.015625], [0.109375 0.015625], [0.140625 0.015625], [0.171875 0.015625], [0.203125 0.015625], [0.234375 0.015625], [0.265625 0.015625], [0.296875 0.015625], [0.328125 0.015625], [0.359375 0.015625], [0.
        """
        """
        points_for_image.shape : (1024, 2)
        points_for_image : [[ 10.       6.6875], [ 30.       6.6875], [ 50.       6.6875], [ 70.       6.6875], [ 90.       6.6875], [110.       6.6875], [130.       6.6875], [150.       6.6875], [170.       6.6875], [190.       6.6875], [210.       6.6875], [230.       6.6875], [250.       6.6875], [270.       6.6875], [290.       6.6875], [310.       6.6875], [330.       6.6875], [350.       6.6875], [370.       6.6875], [390.       6.6875], [410.       6.6875], [430.       6.6875], [450.       6.6875], [470.       6.6875], [490.       6.6875], [510.       6.6875], [530.       6.6875], [550.       6.6875], [570.       6.6875], [590.       6.6875], [610.       6.6875], [630.       6.6875], [ 10.      20.0625], [ 30.      20.0625], [ 50.      20.0625], [ 70.      20.0625], [ 90.      20.0625], [110.      20.0625], [130.      20.0625], [150.      20.0625], [170.      20.0625], [190.      20.0625], [210.      20.0625], [230.      20.0625], [250.      20.0625], [270.      20.0625], [290.      20.0625], [310.      2...
        """

        # Generate masks for this crop in batches
        data = MaskData()
        """
        points.shape : (64, 2)
        points : [[ 10.       6.6875], [ 30.       6.6875], [ 50.       6.6875], [ 70.       6.6875], [ 90.       6.6875], [110.       6.6875], [130.       6.6875], [150.       6.6875], [170.       6.6875], [190.       6.6875], [210.       6.6875], [230.       6.6875], [250.       6.6875], [270.       6.6875], [290.       6.6875], [310.       6.6875], [330.       6.6875], [350.       6.6875], [370.       6.6875], [390.       6.6875], [410.       6.6875], [430.       6.6875], [450.       6.6875], [470.       6.6875], [490.       6.6875], [510.       6.6875], [530.       6.6875], [550.       6.6875], [570.       6.6875], [590.       6.6875], [610.       6.6875], [630.       6.6875], [ 10.      20.0625], [ 30.      20.0625], [ 50.      20.0625], [ 70.      20.0625], [ 90.      20.0625], [110.      20.0625], [130.      20.0625], [150.      20.0625], [170.      20.0625], [190.      20.0625], [210.      20.0625], [230.      20.0625], [250.      20.0625], [270.      20.0625], [290.      20.0625], [310.      2...
        """ # points를 64개씩 불러온 이유 : 32 x 32개의 point 중에서 2묶음씩 불러와서, 한번에 (32 x 2)개의 point를 불러옴. 그러니까 16번을 돌아서 1개의 이미지에 대한 Segment Everything을 수행할 수 있는 것 !!!
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            """
            batch_data shape : -> _stats : dict 5 -> iou_preds : (38), points : (38, 2), stability_score : (38), boxes : (38, 4), rles : (38)
            batch_data : <mobile_sam.utils.amg.MaskData object at 0x7f7f279c34c0>
            """
            data.cat(batch_data)
            del batch_data
        """
        data shape : -> _stats : dict 5 -> iou_preds : (592), points : (592, 2), stability_score : (592), boxes : (592, 4), rles : (592)
        data : <mobile_sam.utils.amg.MaskData object at 0x7f7f279c34c0>
        """
        self.predictor.reset_image()

        # Remove duplicates within this crop (1개의 Crop 내에서 NMS)
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        """
        keep_by_nms.shape : (51)
        keep_by_nms : tensor([515, 251,  14, 286,  66, 210,  12, 536, 274, 271, 482, 435, 260, 104,
        366, 513, 342, 307, 443, 211, 394, 229, 310, 362,  18, 360, 518, 433,
        501, 517, 396, 344, 225, 313, 308,  15, 364, 429, 350, 516, 535, 431,
        403, 438,  17, 129, 413, 326, 457, 483, 275], device='cuda:0')
        """
        data.filter(keep_by_nms)
        """
        data shape : -> _stats : dict 5 -> iou_preds : (51), points : (51, 2), stability_score : (51), boxes : (51, 4), rles : (51)
        data : <mobile_sam.utils.amg.MaskData object at 0x7f7f279c34c0>
        """

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        """
        data shape : -> _stats : dict 5 -> iou_preds : (51), points : (51, 2), stability_score : (51), boxes : (51, 4), rles : (51)
        data : <mobile_sam.utils.amg.MaskData object at 0x7f7f279c34c0>
        """
        data["points"] = uncrop_points(data["points"], crop_box)
        """
        data shape : -> _stats : dict 5 -> iou_preds : (51), points : (51, 2), stability_score : (51), boxes : (51, 4), rles : (51)
        data : <mobile_sam.utils.amg.MaskData object at 0x7f7f279c34c0>
        """
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])
        """
        data shape : -> _stats : dict 6 -> iou_preds : (51), points : (51, 2), stability_score : (51), boxes : (51, 4), rles : (51), crop_boxes : (51, 4)
        data : <mobile_sam.utils.amg.MaskData object at 0x7f7f279c34c0>
        """

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size
        """
        orig_size shape : (2)
        orig_size : (428, 640)
        """

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        """
        transformed_points.shape : (64, 2)
        transformed_points : [[  16.         10.703125], [  48.         10.703125], [  80.         10.703125], [ 112.         10.703125], [ 144.         10.703125], [ 176.         10.703125], [ 208.         10.703125], [ 240.         10.703125], [ 272.         10.703125], [ 304.         10.703125], [ 336.         10.703125], [ 368.         10.703125], [ 400.         10.703125], [ 432.         10.703125], [ 464.         10.703125], [ 496.         10.703125], [ 528.         10.703125], [ 560.         10.703125], [ 592.         10.703125], [ 624.         10.703125], [ 656.         10.703125], [ 688.         10.703125], [ 720.         10.703125], [ 752.         10.703125], [ 784.         10.703125], [ 816.         10.703125], [ 848.         10.703125], [ 880.         10.703125], [ 912.         10.703125], [ 944.         10.703125], [ 976.         10.703125], [1008.         10.703125], [  16.         32.109375], [  48.         32.109375], [  80.         32.109375], [ 112.         32.109375], [ 144.         32.109375], ...
        """
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        """
        in_points.shape : (64, 2)
        in_points : tensor([[  16.0000,   10.7031],
        [  48.0000,   10.7031],
        [  80.0000,   10.7031],
        [ 112.0000,   10.7031],
        [ 144.0000,   10.7031],
        [ 176.0000,   10.7031],
        [ 208.0000,   10.7031],
        [ 240.0000,   10.7031],
        [ 272.0000,   10.7031],
        [ 304.0000,   10.7031],
        [ 336.0000,   10.7031],
        [ 368.0000,   10.7031],
        [ 400.0000,   10.7031],
        [ 432.0000,   10.7031],
        [ 464.0000,   10.7031],
        [ 496.0000,   10.7031],
        [ 528.0000,   10.7031],
        [ 560.0000,   10.7031],
        [ 592.0000,   10.7031],
        [ 624.0000,   10.7031],
        [ 656.0000,   10.7031],
        [ 688.0000,   10.7031],
        [ 720.0000,   10.7031],
        [ 752.0000,   10.7031],
        [ 784.0000,   10.7031],
        [ 816.0000,   10.7031],
        [ 848.0000,   10.7031],
        [ 880.0000,   10.7031],
        [ 912.0000,   10.7031],
        [ 944.0000,   10.7031],
        [ 976.0000,   10.7031],
        [1008.0000,   10.7031],
        [  16.0000,   32.1094],
        [  48.0000,   32.1094],
        [  80.0000,   32.1094],
        [ 112.0000,   32.1094],
        [ 144.0000,   32.1094],
        [ 176.0000,   32.1094],
        [ 208.0000,   32.1094],
        [ 240.0000,   32.1094],
        [ 272.0000,   32.1094],
        [ 304.0000,   32.1094],
        [ 336.0000,   32.1094],
        [ 368.0000,   32.1094],
        [ 400.0000,   32.1094],
        [ 432.0000,   32.1094],
        [ 464.0000,   32.1094],
        [ 496.0000,   32.1094],
        [ 528.0000,   32.1094],
        [ 560.0000,   32.1094],
        [ 592.0000,   32.1094],
        [ 624.0000,   32.1094],
        [ 656.0000,   32.1094],
        [ 688.0000,   32.1094],
        [ 720.0000,   32.1094],
        [ 752.0000,   32.1094],
        [ 784.0000,   32.1094],
        [ 816.0000,   32.1094],
        [ 848.0000,   32.1094],
        [ 880.0000,   32.1094],
        [ 912.0000,   32.1094],
        [ 944.0000,   32.1094],
        [ 976.0000,   32.1094],
        [1008.0000,   32.1094]], device='cuda:0', dtype=torch.float64)
        """
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        """
        in_labels.shape : 64
        in_labels : tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0',
           dtype=torch.int32)
        """
        # 각 point 당 3개의 mask 추출 !!!
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )
        """
        masks.shape : (64, 3, 428, 640)
        masks : tensor([[[[ -3.0741,  -2.4351,  -0.2340,  ..., -12.9795, -12.7850, -12.7285],
          [ -3.4602,  -2.8116,  -0.5774,  ..., -13.1241, -12.9524, -12.9025],
          [ -4.7899,  -4.1081,  -1.7599,  ..., -13.6220, -13.5289, -13.5019],
          ...,
          [-14.2471, -14.5809, -15.7309,  ..., -15.9008, -14.8361, -14.5270],
          [-13.6957, -14.0359, -15.2078,  ..., -15.5235, -14.3640, -14.0274],
          [-11.5744, -11.8408, -12.7582,  ..., -12.7280, -12.3767, -12.2747]],

         [[ -3.6675,  -2.7088,   0.5933,  ..., -11.7157, -12.1674, -12.2986],
          [ -4.1411,  -3.1995,   0.0438,  ..., -11.8000, -12.1907, -12.3041],
          [ -5.7719,  -4.8891,  -1.8484,  ..., -12.0903, -12.2707, -12.3230],
          ...,
          [-13.5340, -13.9586, -15.4212,  ..., -15.3320, -14.0223, -13.6421],
          [-13.1365, -13.5705, -15.0655,  ..., -15.0800, -13.6620, -13.2504],
          [-11.0447, -11.3655, -12.4707,  ..., -12.1996, -11.7380, -11.6039]],

         [[ -1.9334,  -1.7710,  -1.2118,  ...,  -3.5801,  -3.6735,  -3.7006],
          [ -2.1655,  -1.9766,  -1.3261,  ...,  -3.6269,  -3.6956,  -3.7155],
          [ -2.9646,  -2.6845,  -1.7198,  ...,  -3.7882,  -3.7714,  -3.7666],
          ...,
          [ -3.7248,  -3.8641,  -4.3441,  ...,  -3.3719,  -3.1318,  -3.0621],
          [ -3.4248,  -3.5622,  -4.0358,  ...,  -3.2100,  -2.9029,  -2.8137],
          [ -2.7910,  -2.8931,  -3.2447,  ...,  -2.4999,  -2.4476,  -2.4325]]],


        [[[ -2.6170,  -2.0783,  -0.2225,  ..., -13.1957, -12.9262, -12.8480],
          [ -2.9472,  -2.3987,  -0.5097,  ..., -13.3211, -13.0927, -13.0264],
          [ -4.0840,  -3.5023,  -1.4988,  ..., -13.7531, -13.6660, -13.6407],
          ...,
          [-14.5317, -14.8815, -16.0864,  ..., -16.5954, -15.4468, -15.1133],
          [-13.8890, -14.2362, -15.4322,  ..., -16.1572, -14.9069, -14.5439],
          [-11.5983, -11.8748, -12.8272,  ..., -12.9863, -12.5412, -12.4120]],

         [[ -3.2180,  -2.1591,   1.4884,  ..., -13.1799, -13.3743, -13.4307],
          [ -3.6865,  -2.6549,   0.8986,  ..., -13.2041, -13.3589, -13.4038],
          [ -5.2999,  -4.3622,  -1.1323,  ..., -13.2875, -13.3059, -13.3112],
          ...,
          [-14.0624, -14.4542, -15.8036,  ..., -14.7035, -13.1981, -12.7611],
          [-13.9003, -14.3010, -15.6810,  ..., -14.5605, -12.9524, -12.4856],
          [-12.0819, -12.3541, -13.2916,  ..., -11.7169, -11.0588, -10.8678]],

         [[ -1.9829,  -1.8265,  -1.2880,  ...,  -3.5656,  -3.6623,  -3.6903],
          [ -2.1859,  -2.0065,  -1.3885,  ...,  -3.6151,  -3.6855,  -3.7059],
          [ -2.8851,  -2.6262,  -1.7346,  ...,  -3.7854,  -3.7653,  -3.7594],
          ...,
          [ -3.7706,  -3.9093,  -4.3867,  ...,  -3.3099,  -3.0361,  -2.9567],
          [ -3.4802,  -3.6141,  -4.0753,  ...,  -3.1459,  -2.8114,  -2.7143],
          [ -2.8497,  -2.9538,  -3.3122,  ...,  -2.3985,  -2.3120,  -2.2869]]],


        [[[ -4.1544,  -4.4985,  -5.6835,  ..., -11.8048, -11.5542, -11.4815],
          [ -3.9339,  -4.2647,  -5.4041,  ..., -11.7306, -11.4917, -11.4224],
          [ -3.1746,  -3.4598,  -4.4418,  ..., -11.4751, -11.2765, -11.2188],
          ...,
          [-13.6316, -13.6892, -13.8878,  ..., -14.4310, -13.6295, -13.3968],
          [-13.5990, -13.6675, -13.9034,  ..., -14.5676, -13.7700, -13.5384],
          [-12.7650, -12.8663, -13.2152,  ..., -12.8473, -12.6314, -12.5687]],

         [[ -3.3149,  -3.6037,  -4.5983,  ...,  -7.9334,  -7.9174,  -7.9127],
          [ -3.1002,  -3.3860,  -4.3705,  ...,  -7.8739,  -7.8492,  -7.8420],
          [ -2.3608,  -2.6366,  -3.5864,  ...,  -7.6687,  -7.6144,  -7.5987],
          ...,
          [  0.1912,   0.3852,   1.0533,  ...,  -0.4916,  -0.5488,  -0.5653],
          [ -0.4671,  -0.2939,   0.3028,  ...,  -0.6191,  -0.6993,  -0.7226],
          [ -1.5468,  -1.4467,  -1.1022,  ...,  -0.9793,  -1.0950,  -1.1286]],

         [[ -3.0405,  -3.1474,  -3.5157,  ...,  -4.5019,  -4.4269,  -4.4051],
          [ -2.9393,  -3.0430,  -3.4002,  ...,  -4.4473,  -4.3401,  -4.3090],
          [ -2.5909,  -2.6835,  -3.0024,  ...,  -4.2591,  -4.0411,  -3.9778],
          ...,
          [  1.9183,   1.9980,   2.2726,  ...,   1.6847,   1.5789,   1.5482],
          [  1.3852,   1.4403,   1.6299,  ...,   1.4223,   1.3089,   1.2760],
          [  0.3209,   0.3648,   0.5160,  ...,   0.8080,   0.7113,   0.6832]]],


        ...,


        [[[ -8.8104,  -8.7362,  -8.4804,  ...,   0.6259,   0.7978,   0.8477],
          [ -8.7230,  -8.6419,  -8.3623,  ...,   0.6154,   0.7393,   0.7753],
          [ -8.4220,  -8.3170,  -7.9554,  ...,   0.5792,   0.5379,   0.5259],
          ...,
          [-17.3571, -17.2757, -16.9950,  ..., -20.8365, -19.5192, -19.1368],
          [-17.4210, -17.3332, -17.0308,  ..., -20.7109, -19.2272, -18.7964],
          [-16.3390, -16.2576, -15.9774,  ..., -18.6492, -17.4772, -17.1370]],

         [[-10.2729, -10.2476, -10.1604,  ...,  -3.3144,  -3.3109,  -3.3099],
          [-10.1017, -10.0908, -10.0530,  ...,  -3.5802,  -3.5754,  -3.5740],
          [ -9.5124,  -9.5508,  -9.6832,  ...,  -4.4957,  -4.4863,  -4.4835],
          ...,
          [-20.2562, -20.4019, -20.9038,  ..., -23.9078, -22.3679, -21.9208],
          [-20.2595, -20.3766, -20.7800,  ..., -23.5736, -21.7574, -21.2301],
          [-18.5180, -18.7369, -19.4913,  ..., -20.5867, -19.4899, -19.1715]],

         [[-12.2032, -11.9849, -11.2330,  ...,   1.2089,   1.2050,   1.2038],
          [-12.0222, -11.8407, -11.2156,  ...,   1.1418,   1.1313,   1.1283],
          [-11.3987, -11.3440, -11.1558,  ...,   0.9106,   0.8779,   0.8683],
          ...,
          [-17.4003, -17.6827, -18.6557,  ..., -16.0430, -14.9763, -14.6666],
          [-17.5465, -17.7858, -18.6100,  ..., -15.8840, -14.5842, -14.2068],
          [-16.2739, -16.6137, -17.7842,  ..., -14.8935, -14.0957, -13.8640]]],


        [[[-20.7923, -20.7323, -20.5259,  ...,   1.0358,   0.6999,   0.6023],
          [-20.6790, -20.6434, -20.5206,  ...,   1.0374,   0.7136,   0.6196],
          [-20.2891, -20.3371, -20.5024,  ...,   1.0428,   0.7610,   0.6792],
          ...,
          [-24.6499, -24.7116, -24.9242,  ..., -26.3867, -23.9492, -23.2415],
          [-24.6572, -24.6542, -24.6437,  ..., -26.4575, -23.6924, -22.8897],
          [-22.8609, -23.1137, -23.9843,  ..., -23.5270, -22.3226, -21.9729]],

         [[-16.7978, -16.8757, -17.1438,  ...,   1.8288,   1.1670,   0.9749],
          [-16.8301, -16.9013, -17.1465,  ...,   2.0680,   1.4056,   1.2133],
          [-16.9414, -16.9896, -17.1557,  ...,   2.8916,   2.2273,   2.0344],
          ...,
          [-23.3886, -23.4946, -23.8601,  ..., -24.5235, -23.4209, -23.1008],
          [-23.4158, -23.5292, -23.9196,  ..., -24.7549, -23.6369, -23.3124],
          [-23.4979, -23.5252, -23.6193,  ..., -24.2480, -23.1712, -22.8586]],

         [[ -7.1121,  -6.8970,  -6.1561,  ...,  -0.0414,  -0.3952,  -0.4979],
          [ -7.0266,  -6.8356,  -6.1777,  ...,  -0.1880,  -0.4072,  -0.4709],
          [ -6.7324,  -6.6244,  -6.2522,  ...,  -0.6928,  -0.4487,  -0.3779],
          ...,
          [-11.3279, -11.3278, -11.3277,  ..., -11.3504, -10.9519, -10.8362],
          [-11.3557, -11.3008, -11.1118,  ..., -10.9530, -10.2125,  -9.9975],
          [-10.1432, -10.2025, -10.4066,  ..., -10.1647,  -9.7865,  -9.6767]]],


        [[[-19.7130, -19.7882, -20.0473,  ...,  -1.0168,  -0.9225,  -0.8952],
          [-19.7188, -19.8148, -20.1457,  ...,  -0.9427,  -0.9427,  -0.9427],
          [-19.7387, -19.9065, -20.4846,  ...,  -0.6878,  -1.0123,  -1.1065],
          ...,
          [-21.6847, -21.8578, -22.4538,  ..., -21.8265, -19.7100, -19.0955],
          [-21.4995, -21.6355, -22.1041,  ..., -22.0335, -19.6902, -19.0099],
          [-20.4736, -20.7724, -21.8018,  ..., -19.5770, -18.7047, -18.4515]],

         [[-19.4117, -19.4107, -19.4072,  ...,   2.8173,   2.1770,   1.9912],
          [-19.2646, -19.2842, -19.3516,  ...,   2.9070,   2.2404,   2.0469],
          [-18.7579, -18.8484, -19.1603,  ...,   3.2161,   2.4585,   2.2386],
          ...,
          [-24.9542, -24.9439, -24.9086,  ..., -27.5049, -25.2880, -24.6444],
          [-24.9388, -24.8903, -24.7232,  ..., -27.5781, -25.1704, -24.4714],
          [-23.9615, -24.0316, -24.2734,  ..., -25.4195, -23.7530, -23.2692]],

         [[ -7.1217,  -6.9535,  -6.3741,  ...,   0.9644,   0.4940,   0.3575],
          [ -7.0206,  -6.8766,  -6.3805,  ...,   0.8561,   0.5128,   0.4132],
          [ -6.6725,  -6.6118,  -6.4025,  ...,   0.4830,   0.5777,   0.6052],
          ...,
          [-11.4179, -11.4234, -11.4420,  ..., -11.6499, -11.2062, -11.0774],
          [-11.4632, -11.4141, -11.2452,  ..., -11.3248, -10.5688, -10.3493],
          [-10.3630, -10.4264, -10.6447,  ..., -10.5510, -10.1350, -10.0143]]]],
       device='cuda:0')
        """
        """
        iou_preds.shape : (64, 3)
        iou_preds : tensor([[0.9188, 0.9321, 0.6138],
        [0.9367, 0.9537, 0.6238],
        [0.8229, 0.7966, 0.9354],
        [0.8728, 0.9711, 0.5672],
        [0.8462, 0.9732, 0.5995],
        [0.9151, 0.9704, 0.6136],
        [0.7329, 0.9349, 0.9260],
        [0.5760, 0.9373, 0.8969],
        [0.6438, 0.9470, 0.9049],
        [0.7993, 0.9501, 0.9624],
        [0.8046, 0.9594, 0.9640],
        [0.8102, 0.9610, 0.9640],
        [0.8344, 0.9639, 0.9471],
        [0.8342, 0.9685, 0.9188],
        [0.8400, 0.8832, 0.7953],
        [0.7662, 0.9598, 0.9671],
        [0.7530, 0.9376, 0.9627],
        [0.7674, 0.9430, 0.9628],
        [0.7915, 0.9404, 0.9630],
        [0.7920, 0.7291, 0.8231],
        [0.5252, 0.8755, 0.9237],
        [0.5269, 0.9339, 0.8157],
        [0.9473, 0.9296, 0.5791],
        [0.9838, 0.9658, 0.9807],
        [0.9901, 0.9779, 0.9864],
        [0.9934, 0.9809, 0.9919],
        [0.9463, 0.9467, 0.8209],
        [0.8992, 0.9711, 0.9264],
        [0.8297, 0.9951, 0.9628],
        [0.8586, 0.9968, 0.9676],
        [0.9510, 0.9444, 0.9151],
        [0.9571, 0.9668, 0.9435],
        [0.8391, 0.9409, 0.9702],
        [0.9346, 0.9635, 0.6585],
        [0.9350, 0.9596, 0.6725],
        [0.9070, 0.7766, 0.9439],
        [0.9067, 0.7307, 0.9070],
        [0.9139, 0.9738, 0.5800],
        [0.6639, 0.9492, 0.8886],
        [0.5719, 0.9545, 0.8705],
        [0.7879, 0.9464, 0.9628],
        [0.8003, 0.9509, 0.9647],
        [0.8064, 0.9603, 0.9669],
        [0.8271, 0.9654, 0.9704],
        [0.8016, 0.9328, 0.9678],
        [0.7539, 0.7714, 0.8464],
        [0.7256, 0.9866, 0.7952],
        [0.8018, 0.8891, 0.8524],
        [0.7158, 0.9374, 0.9716],
        [0.8000, 0.9293, 0.9591],
        [0.8183, 0.9205, 0.9581],
        [0.6201, 0.8953, 0.9106],
        [0.8056, 0.8916, 0.9543],
        [0.5268, 0.9446, 0.7912],
        [0.5905, 0.9322, 0.8135],
        [0.9802, 0.9634, 0.9795],
        [0.9868, 0.9757, 0.9825],
        [0.9894, 0.9778, 0.9867],
        [0.9908, 0.9774, 0.9898],
        [0.9921, 0.9744, 0.9895],
        [0.9899, 0.9750, 0.9898],
        [0.8127, 0.9816, 0.9725],
        [0.8535, 0.8642, 0.8130],
        [0.7907, 0.8133, 0.8000]], device='cuda:0')
        """

        # %%%%%%%%%%%%%%%%%%%%
        # SAM Mask Size ●
        # %%%%%%%%%%%%%%%%%%%%
        # 1번째 mask : 작은 크기 mask
        # 2번째 mask : 중간 크기 mask
        # 3번째 mask : 최대 크기 mask
        masks = masks[:, self.sam_mask_size_list, :, :]
        iou_preds = iou_preds[:, self.sam_mask_size_list]
        # %%%%%%%%%%%%%%%%%%%%

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        """
        data shape : -> _stats : dict 3 -> masks : (192, 428, 640), iou_preds : (192), points : (192, 2)
        data : <mobile_sam.utils.amg.MaskData object at 0x7f694c1d45b0>
        """
        del masks

        # Filter by predicted IoU
        """
        self.pred_iou_thresh shape : float
        self.pred_iou_thresh : 0.95
        """
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            """
            keep_mask.shape : (192)
            keep_mask : tensor([False, False, False, False,  True, False, False, False, False, False,
             True, False, False,  True, False, False,  True, False, False, False,
            False, False, False, False, False, False, False, False,  True,  True,
            False,  True,  True, False,  True,  True, False,  True, False, False,
             True, False, False, False, False, False,  True,  True, False, False,
             True, False, False,  True, False, False,  True, False, False, False,
            False, False, False, False, False, False, False, False, False,  True,
             True,  True,  True,  True,  True,  True,  True,  True, False, False,
            False, False,  True, False, False,  True,  True, False,  True,  True,
             True, False, False,  True,  True, False, False, False,  True, False,
             True, False, False,  True, False, False, False, False, False, False,
            False, False,  True, False, False, False, False, False,  True, False,
            False, False,  True, False,  True,  True, False,  True,  True, False,
             True,  True, False, False,  True, False, False, False, False,  True,
            False, False, False, False, False, False,  True, False, False,  True,
            False, False,  True, False, False, False, False, False,  True, False,
            False, False, False, False, False,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True, False,  True,  True, False, False, False, False,
            False, False], device='cuda:0')
            """
            data.filter(keep_mask)
            """
            data shape : -> _stats : dict 3 -> masks : (72, 428, 640), iou_preds : (72), points : (72, 2)
            data : <mobile_sam.utils.amg.MaskData object at 0x7f562cb42ee0>
            """

        # Calculate stability score
        """
        self.predictor.model.mask_threshold shape : float
        self.predictor.model.mask_threshold : 0.0
        """
        """
        self.stability_score_offset shape : float
        self.stability_score_offset : 1.0
        """
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        """
        data shape : -> _stats : dict 4 -> masks : (72, 428, 640), iou_preds : (72), points : (72, 2), stability_score : (72)
        data : <mobile_sam.utils.amg.MaskData object at 0x7f562cb42ee0>
        """
        # Filter by stability score
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            """
            keep_mask.shape : (72)
            keep_mask : tensor([False,  True,  True,  True, False, False, False, False, False, False,
            False, False, False, False, False, False, False,  True, False,  True,
             True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
             True,  True,  True, False, False, False, False,  True, False, False,
            False, False, False, False, False, False, False,  True, False, False,
            False, False,  True, False,  True,  True,  True,  True,  True,  True,
             True,  True,  True,  True,  True, False,  True,  True,  True,  True,
             True,  True], device='cuda:0')
            """
            data.filter(keep_mask)
            """
            data shape : -> _stats : dict 4 -> masks : (38, 428, 640), iou_preds : (38), points : (38, 2), stability_score : (38)
            data : <mobile_sam.utils.amg.MaskData object at 0x7f562cb42ee0>
            """

        # Threshold masks and calculate boxes # 기존 모델에서 추출된 mask의 픽셀값이 float 형태임. 이 픽셀들 중에서 0 이상인 경우에만 True로 바꿔줌 !!!
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        """
        data shape : -> _stats : dict 4 -> masks : (38, 428, 640), iou_preds : (38), points : (38, 2), stability_score : (38)
        data : <mobile_sam.utils.amg.MaskData object at 0x7f562cb42ee0>
        """
        data["boxes"] = batched_mask_to_box(data["masks"])
        """
        data shape : -> _stats : dict 5 -> masks : (38, 428, 640), iou_preds : (38), points : (38, 2), stability_score : (38), boxes : (38, 4)
        data : <mobile_sam.utils.amg.MaskData object at 0x7f562cb42ee0>
        """

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        """
        keep_mask.shape (38)
        keep_mask : tensor([True, True, True, True, True, True, True, True, True, True, True, True,
            True, True, True, True, True, True, True, True, True, True, True, True,
            True, True, True, True, True, True, True, True, True, True, True, True,
            True, True], device='cuda:0')
        """
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        """
        data shape : -> _stats : dict 5 -> masks : (38, 428, 640), iou_preds : (38), points : (38, 2), stability_score : (38), boxes : (38, 4)
        data : <mobile_sam.utils.amg.MaskData object at 0x7f562cb42ee0>
        """

        data["rles"] = mask_to_rle_pytorch(data["masks"])
        """
        data shape : -> _stats : dict 6 -> masks : (38, 428, 640), iou_preds : (38), points : (38, 2), stability_score : (38), boxes : (38, 4), rles : (38)
        data : <mobile_sam.utils.amg.MaskData object at 0x7f562cb42ee0>
        """
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        """
        mask_data shape : -> _stats : dict 6 -> iou_preds : (51), points : (51, 2), stability_score : (51), boxes : (51, 4), rles : (51), crop_boxes : (51, 4)
        mask_data : <mobile_sam.utils.amg.MaskData object at 0x7feb468526a0>
        """
        for rle in mask_data["rles"]:
            """
            rle shape : dict 2 -> size : list 2, counts : list 129
            rle : {'size': [428, 640], 'counts': [136056, 4, 421, 8, 417, 12, 415, 13, 414, 14, 413, 15, 412, 17, 410, 18, 410, 18, 409, 19, 408, 20, 407, 22, 404, 24, 403, 25, 403, 25, 402, 26, 402, 27, 400, 28, 399, 29, 398, 30, 397, 31, 397, 31, 396, 32, 396, 33, 394, 34, 394, 34, 393, 35, 393, 35, 393, 35, 393, 36, 392, 36, 391, 37, 391, 37, 391, 37, 391, 37, 391, 37, 391, 37, 391, 38, 390, 38, 390, 38, 390, 38, 390, 38, 391, 37, 391, 37, 391, 37, 391, 37, 391, 37, 392, 36, 392, 37, 392, 36, 392, 36, 393, 35, 393, 35, 394, 34, 395, 33, 396, 32, 397, 31, 398, 30, 399, 29, 400, 27, 403, 23, 412, 14, 416, 10, 421, 5, 110896]}
            """
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            """
            mask.shape : (428, 640)
            mask : [[False False False ... False False False], [False False False ... False False False], [False False False ... False False False], ..., [False False False ... False False False], [False False False ... False False False], [False False False ... False False False]]
            """
            """
            changed : False
            """
            unchanged = not changed
            """
            unchanged : True
            """
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            """
            mask.shape : (428, 640)
            mask : [[False False False ... False False False], [False False False ... False False False], [False False False ... False False False], ..., [False False False ... False False False], [False False False ... False False False], [False False False ... False False 
            """
            unchanged = unchanged and not changed
            """
            unchanged : False
            """

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            """
            new_masks[0].shape : (1, 428, 640)
            new_masks[0] : tensor([[[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]]])
            """
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))
            """
            scores[0] shape : float
            scores[0] : 0.0
            """

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data
