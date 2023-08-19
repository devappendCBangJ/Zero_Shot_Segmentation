# ==============================================================
# 0. 라이브러리 불러오기
# ==============================================================
# --------------------------------------------------------------
# 1) CLIP + MobileSAM
# --------------------------------------------------------------
import clip
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# %%%%%%%%%%%%%%%%%%%%
# Free SOLO ●
# %%%%%%%%%%%%%%%%%%%%
# from detectron2.checkpoint import DetectionCheckpointer
# from freesolo.engine.trainer import BaselineTrainer

# # hacky way to register
# import freesolo.data.datasets.builtin
# from freesolo.modeling.solov2 import PseudoSOLOv2

# %%%%%%%%%%%%%%%%%%%%
# RefCOCO Dataset ●
# %%%%%%%%%%%%%%%%%%%%
# # refer data
# from data.dataset_refer_bert import ReferDataset
# from utils import setup
# %%%%%%%%%%%%%%%%%%%%

from model.backbone import clip_backbone, CLIPViTFM
from utils import default_argument_parser, Compute_IoU, extract_noun_phrase

# --------------------------------------------------------------
# 2) 딥러닝 라이브러리
# --------------------------------------------------------------
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import spacy

# --------------------------------------------------------------
# 3) 기본 라이브러리
# --------------------------------------------------------------
import tqdm
import numpy as np # 이미지 확장자 변환 ●
import os # 파일 불러오기 ●
import PIL # 이미지 불러오기 ●

import matplotlib.pyplot as plt # 시각화 ●
import cv2 # 시각화 ●

# ==============================================================
# 1. 변수 선언
# ==============================================================
args = default_argument_parser().parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================
# 1. 함수 정의
# ==============================================================
# --------------------------------------------------------------
# 1) pil <-> tensor
# --------------------------------------------------------------
def pil_to_tensor(pil_image):
    """
    PIL: [width, height]
    -> NumPy: [width, height, channel]
    -> Tensor: [channel, width, height]
    """
    return torch.as_tensor(np.asarray(pil_image)).permute(2,0,1)

def tensor_to_pltimg(tensor_image): # 시각화 ●
    return tensor_image.permute(1, 2, 0).numpy()

font_style = {'color':  'black', # 시각화 ●
        'style': 'normal',
        'size': 6}

# --------------------------------------------------------------
# 2) 마스크 시각화
# --------------------------------------------------------------
def plot_mask_generator_pred(_image_filename, _image, _mask_generator_pred):
    args.subplot_count = 0
    plt.figure(figsize=(20, 20))
    for j in range(4):
        # --------------------------------------------------------------
        # (1) Plt Subplot Split
        # --------------------------------------------------------------
        plot_idx = args.subplot_count % (args.subplot_rows * args.subplot_columns)
        plt.subplot(args.subplot_rows, args.subplot_columns, plot_idx + 1)

        # (2) Plt Title
        plt.title(f'{args.base_dataset_path}/{_image_filename}', fontsize=5)

        # (3) Plt Label
        # plt.xlabel('x-axis')
        # plt.ylabel('y-axis')
        plt.xticks([])
        plt.yticks([])

        # (4) Subplot에 Image 그리기
        plt.imshow(_image)
        if j == 1:
            show_anns(_mask_generator_pred, mask_ok = True, point_ok = False, bbox_ok = False)
        elif j == 2:
            show_anns(_mask_generator_pred, mask_ok = True, point_ok = True, bbox_ok = False)
        elif j == 3:
            show_anns(_mask_generator_pred, mask_ok = True, point_ok = False, bbox_ok = True)

        # (5) Plt Figure Axis
        plt.axis('off')

        args.subplot_count += 1

    plt.show()

def plot_mask_pred(fig_num, pred_masks):
    # 6] 마스크 후보 시각화 ●
    plt.figure(figsize=(8, 8))
    img_iter = 0
    for pm in pred_masks:
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
            fig_num += 1
    plt.show()
    # plt.waitforbuttonpress(1)
    # plt.close(fig)

def plot_mask_result(image_filename, real_image, sentence, img_iter, sentence_for_spacy, noun_phrase, result_seg, scores):
    # 1] 원본 이미지 시각화 ●
    plot_idx = img_iter % (4 * 4)
    plt.subplot(4, 4, plot_idx + 1)
    plt.title(f'O_{sentence}', fontsize=7)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(real_image)
    plt.axis("off")
    img_iter += 1

    # # 2] 타겟 마스크 시각화 ●
    # plot_idx = img_iter % (4 * 4)
    # plt.subplot(4, 4, plot_idx + 1)
    # plt.title(f'T_{sentence}', fontsize=7)
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(target[0].cpu().numpy() * 255)
    # plt.axis("off")
    # img_iter += 1

    # 3] 결과 마스크 시각화 ●
    plot_idx = img_iter % (4 * 4)
    plt.subplot(4, 4, plot_idx + 1)
    plt.title(f'S_{sentence}', fontsize=7)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(result_seg.cpu().numpy() * 255)
    plt.axis("off")
    img_iter += 1

    # 4] Score 정렬 ●
    sorted_scores = sorted(scores[-1], reverse=True)
    sorted_scores_idx = sorted(range(len(scores[-1])), key=lambda k: scores[-1][k], reverse=True)

    # 5] 사진 경로 + 입력 문장 + 단어 + Score 순위 시각화 ●
    plot_idx = img_iter % (4 * 4)
    plt.subplot(4, 4, plot_idx + 1)
    plt.text(0, 1, f'[sentence] {sentence_for_spacy}', fontdict=font_style)
    plt.text(0, 0.9, f'[noun] {noun_phrase}', fontdict=font_style)
    for ssi in range(len(sorted_scores)):
        plt.text(0, 0.8 - (ssi * 0.1), f'[{sorted_scores_idx[ssi]}] {sorted_scores[ssi]}', fontdict=font_style)
        if 0.8 - (ssi * 0.1) <= 0.1:
            break
    plt.xticks([])
    plt.yticks([])
    plt.axis("off")
    img_iter += 2

    # %%%%%%%%%%%%%%%%%%%%
    # RefCOCO Dataset ●
    # %%%%%%%%%%%%%%%%%%%%
    # this_IoU, m_IoU, cum_I, cum_U = Compute_IoU(result_seg, target, cum_I, cum_U, m_IoU)
    """
    m_IoU : [tensor(0., device='cuda:0')]
    cum_I : tensor(0, device='cuda:0')
    cum_U :tensor(85325, device='cuda:0')
    """
    # plt.text(0, 0, f'[this_IoU] {this_IoU}', fontdict=font_style) # IoU 출력 ●
    # %%%%%%%%%%%%%%%%%%%%

    plt.text(0, -0.1, f"[file_name] {args.base_dataset_path}/{image_filename}", fontdict=font_style)  # 파일경로 출력 ●

def show_anns(anns, mask_ok = True, point_ok = False, bbox_ok = False):
    # --------------------------------------------------------------
    # (1) 면적 큰 순서대로 정렬
    # --------------------------------------------------------------
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    """
    sorted_anns.shape : 마스크 개수
    sorted_anns[0] : {'segmentation': array([[False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       ...,
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False],
       [False, False, False, ..., False, False, False]]), 'area': 1251199, 'bbox': [0, 171, 1855, 870], 'predicted_iou': 1.0059185028076172, 'point_coords': [[29.0, 962.4375]], 'stability_score': 0.9520606994628906, 'crop_box': [0, 0, 1856, 1044]}
    """
    # bbox 역할 ??? : Segmentation 영역에 해당하는 [x, y, w, h] 값
    # predicted_iou 역할 ??? :
    # point_coords 역할 ??? : Segment Everything에서 point를 64x64개 뿌렸을 때, 해당 물체를 잡은 point
    # stability_score 역할 ??? :
    # crop_box 역할 ??? : 이미지 [0, 0, 이미지 너비, 이미지 높이] 값

    # --------------------------------------------------------------
    # (2) 출력 이미지 Scale 세팅
    # --------------------------------------------------------------
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # --------------------------------------------------------------
    # (3) 마스크 씌우기
    # --------------------------------------------------------------
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4)) # 마지막 차원 역할 : 투명도 조절
    """
    img.shape : (이미지 높이, 이미지 너비, 4)
    img : [[[1. 1. 1. 1.],  [1. 1. 1. 1.],  [1. 1. 1. 1.],  ...,  [1. 1. 1. 1.],  [1. 1. 1. 1.],  [1. 1. 1. 1.]],, [[1. 1. 1. 1.],  [1. 1. 1. 1.],  [1. 1. 1. 1.],  ...,  [1. 1. 1. 1.],  [1. 1. 1. 1.],  [1. 1. 1. 1.]],, [[1. 1. 1. 1.],  [1. 1. 1. 1.],  [1. 1. 1. 1.],
    """

    img[:,:,3] = 0
    """
    img.shape : (이미지 높이, 이미지 너비, 4)
    img : [[[1. 1. 1. 0.],  [1. 1. 1. 0.],  [1. 1. 1. 0.],  ...,  [1. 1. 1. 0.],  [1. 1. 1. 0.],  [1. 1. 1. 0.]],, [[1. 1. 1. 0.],  [1. 1. 1. 0.],  [1. 1. 1. 0.],  ...,  [1. 1. 1. 0.],  [1. 1. 1. 0.],  [1. 1. 1. 0.]],, [[1. 1. 1. 0.],  [1. 1. 1. 0.],  [1. 1. 1. 0.],
    """
    for ann in sorted_anns:
        # 1] 마스크 시각화
        if mask_ok == True:
            m = ann['segmentation']
            """
            m.shape : (이미지 높이, 이미지 너비)
            m : [[ True  True  True ... False False False], [ True  True  True ... False False False], [ True  True  True ... False False False], ..., [ True  True  True ... False False False], [ True  True  True ... False False False], [ True  True  True ... False False False]]
            """
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            """
            color_mask.shape : 4
            color_mask : [0.91638865 0.74538557 0.17961054 1      ]
            """
            img[m] = color_mask
            """
            img.shape : (이미지 높이, 이미지 너비, 4)
            img : [[False False False ... False False False], [False False False ... False False False], [False False False ... False False False], ..., [False False False ... False False False], [False False False ... False False False], [False False False ... False False False]]
            """
        # 2] 탐지 포인트 시각화
        if point_ok == True:
            color_mask = np.concatenate([np.random.random(3), [1]])
            cv2.circle(img, (int(ann['point_coords'][0][0]), int(ann['point_coords'][0][1])), 10, color_mask, 5, cv2.LINE_AA)
        # 3] bbox 시각화
        if bbox_ok == True:
            color_mask = [255, 255, 255, 1]
            x, y, w, h = list(map(int, ann['bbox']))
            cv2.rectangle(img, (x, y), (x+w, y+h), color_mask, 10)
    # --------------------------------------------------------------
    # (4) 이미지 시각화
    # --------------------------------------------------------------
    ax.imshow(img)

# ==============================================================
# 2) Main문
# ==============================================================
def main(args, resized_height, resized_width):
    # --------------------------------------------------------------
    # 1) 초기화
    # --------------------------------------------------------------
    assert args.eval_only, 'Only eval_only available!'

    # %%%%%%%%%%%%%%%%%%%%
    # Free SOLO ●
    # %%%%%%%%%%%%%%%%%%%%
    # cfg = setup(args) # 내부 구조 ???

    # %%%%%%%%%%%%%%%%%%%%
    # RefCOCO Dataset ●
    # %%%%%%%%%%%%%%%%%%%%
    # if args.dataset == 'refcocog':
    #     args.splitBy = 'umd'  # umd or google in refcocog
    # else:
    #     args.splitBy = 'unc'  # unc in refcoco, refcoco+,
    # %%%%%%%%%%%%%%%%%%%%

    # --------------------------------------------------------------
    # 2) Dataset
    # --------------------------------------------------------------
    # 커스텀 데이터셋 경로 불러오기 ●
    image_filenames = os.listdir(args.base_dataset_path)
    """
    image_filenames len : 1681
    image_filenames : ['0-9593-276_jpg.rf.cd034c928f33279d7d2153700458b553.jpg', '00114337b89792cc_jpg.rf.cdb7f1d39db989b32e293c87dd7a7adc.jpg', '002ccf685b8e6da2_jpg.rf.caa878083b8477414094e9b1e4900b6c.jpg', '00532d9d29cc7f22_jpg.rf.1d5fc2c150e04e69100cf51e7793734d.jpg', '292aa8c57d0c6fa7_jpg.rf.44df388dfa6b01a474d65ac1c6b3e0a9.jpg', 'images--92-_png.rf.f5f41594c04ddd4c820fbf095ced6b13.jpg', 'images--93-_png.rf.5092c924e246e96cb156548f2096c14c.jpg', 'images--94-_png.rf.66e006c42b7c4350fba443324f7587b5.jpg', 'images--95-_png.rf.e5413274d5de5220a3b2ea9a8ddd0897.jpg', 'images--96-_png.rf.9649d0fea341e2e57cedb745f10ecd54.jpg', 'images--97-_png.rf.7d23d8e96252b4582142df365c329f5e.jpg', 'images--98-_png.rf.dca8b25cfa82d3459aebd80891dac27b.jpg', 'images--99-_png.rf.13ce9fe72a3275fce8ad72ce4956d103.jpg', 'images-1-_png_jpg.rf.c9560b5f228100c7211e697c38b40c78.jpg', 'images-10-_png_jpg.rf.ca70f63b3565e064bed491818cbc1ff5.jpg', 'images-100-_png_jpg.rf.56b8e1aff98023c5cb32da9aa9142b70.jpg', 'images-100_jpeg.rf.461ca57ba1d3e43ea17976ad9c52c5b6.jpg', 'images-101_jpeg.rf.8974e6235dfa9de6120c3f48fc8ab45c.jpg', 'images-102_jpeg.rf.10602a44110a8864c8fc5f2ec56ce612.jpg', 'images-103_jpeg.rf.9a3a84736fa0add66eb1c6640cf89810.jpg', 'images-104_jpeg.rf.ef6d428f77343dff2675b687b401c39d.jpg', '009fa98953c23664_jpg.rf.592a1ce1e3179c1c7a7c73c2ebf9f83e.jpg', '01745cd4dc676e1b_jpg.rf.fb5fd5c7ccb9e20c5534f25bf95bd013.jpg', '05ef5bd6ffb4c0c7_jpg.rf.8d1d75bd02fbf8f53820522e56490daa.jpg', '0d4176f2f6bc151e_jpg.rf.d10358484db2cfb4cd1e5cb6968fbd38.jpg', '102b9825ff44c81a_jpg.rf.169d51f058024de0253e8455270599f6.jpg', '10629-8439-31510_jpg.rf.8c7b4a353b529a1d2f3a3e92f791dd26.jpg', '10883-11134-22470_jpg.rf.ca027037549ab04865c19debc0af625a.jpg', '11078-14028-4505_jpg.rf.15d0cf666a26402bf8ddfbdab133e2de.jpg', '11260-10086-19090_jpg.rf.c020a8e804e895c820bd050658d6a7c2.jpg', '11858-6085-11611_jpg.rf.9fd3d4b7bd9743ce07bdd67e49db00e0.jpg', '12282-31391-20867_jpg.rf.67d8d828974bdbfe2c58d2d8b98fb557.jpg', '12895-17279-16742_jpg.rf.349decebe9ff8cf561113a80aae54b2b.jpg', '13226-12070-15070_jpg.rf.16a0a98000d96a53785bd4fe60abd3d7.jpg', '13525-11863-23362_jpg.rf.e67ff36fa3dcf649ac4d18568dec9cff.jpg', '13a6faa42d691f5c_jpg.rf.9722572ab622e399bb07f01b675c8ac7.jpg', 'images-106_jpeg.rf.475ccd898193402dd3ae220154171927.jpg', 'images-107_jpeg.rf.d27a76873d2e26635b7aba4b04c52ee3.jpg', 'images-10_jpeg.rf.ba49ee0986a13fd2b1d5cc04b66f5b44.jpg', 'images-11-_png_jpg.rf.0f82620554283962b20c8314f6de6c7a.jpg', 'images-11_jpeg.rf.4ac88edee42501260cc246f197307a8f.jpg', 'images-12-_png_jpg.rf.1429944337590c796683b20617138241.jpg', 'images-12_jpeg.rf.d5380c726b16b537f32356bb69ca0c71.jpg', 'images-13-_png_jpg.rf.3ce136126e5cd35b9e0d11fd02b8e2b2.jpg', 'images-13_jpeg.rf.4174b317cb263c13b0447110cd6a69f8.jpg', 'images-14-_png_jpg.rf.06c51a5151862f308d10d4c680358bf9.jpg', 'images-14_jpeg.rf.e5ab28361d9ae529a30dbdc236815645.jpg', 'images-15-_png_jpg.rf.207e425e7a11b12a473d7ca7b0a6f922.jpg', 'images-15_jpeg.rf.fc83ea5fd5202bc4fb07ca950cf18f24.jpg', 'images-16-_png_jpg.rf.082e5ad5b03ab0a0540f1cd81a679182.jpg', 'images-16_jpeg.rf.4567785bfeace310594a8ebdb8410a5d.jpg', 'images-17-_png_jpg.rf.50caf6cec6d76fb94c2bd453f656c335.jpg', 'golf-ball-in-rough2_jpg.rf.cd6938433a4bba46ad4fdd19249c1ac8.jpg', 'golf-ball-in-rough3_jpg.rf.b1fedbf8bdb52d727b67a1f106d44103.jpg', 'golf-ball-in-rough4_jpg.rf.edb2fdfc747c389f495fb000ef456bcb.jpg', 'golf-ball-in-rough5_jpg.rf.6f2ccd126707b491bed35a2be18107e0.jpg', 'golf-ball-in-rough6_jpg.rf.34b3f15041446fd41b22b42b62d4ba75.jpg', 'golf-ball-in-rough7_jpg.rf.93294c257cfeaead8a005f38410b8add.jpg', 'golf-ball-in-rough8_jpg.rf.a7e6c4592acb6620b7140e7d8da2d717.jpg', 'golf-ball-in-rough9_jpg.rf.4ab80cc5a9912470aa126f8905dd1783.jpg', 'golf-ball-in-rough_jpg.rf.87865e1b6329031a9a48a503183fdf87.jpg', 'golf-ball-ROUGH1_jpg.rf.43eaf108ae9d76223d5e49493179e89b.jpg', 'n03445777_3278_JPEG.rf.67f211f65d125ef7352342c45233fbf8.jpg', 'n03445777_3307_JPEG.rf.ce77ac45302c5fb9ad60a93c9873b7b7.jpg', 'n03445777_333_JPEG.rf.681eedd8983188b37aa0cb2f2dd2b5e8.jpg', 'n03445777_3386_JPEG_jpg.rf.bd11ce38e63074a7dba62c2cb703f0ed.jpg', 'n03445777_3413_JPEG_jpg.rf.4264a26cb00b4df3031a75009fb752d4.jpg', 'n03445777_3429_JPEG_jpg.rf.54a99779ac72b4cd9aafaa729acb5173.jpg', 'n03445777_3444_JPEG_jpg.rf.0d0807f5d76f3e229c8eec0018333ced.jpg', 'n03445777_3464_JPEG.rf.07dd63c40d8035a22e22480c3b03956c.jpg', 'n03445777_3595_JPEG_jpg.rf.41d839391f5f42702b961679d21a487d.jpg', 'n03445777_36_JPEG.rf.cb283522a201a50f1427eeef3bd6a2f0.jpg', 'n03445777_6324_JPEG_jpg.rf.83d787c7bccd4dc9a1fce41e33b9b7dd.jpg', 'n03445777_6918_JPEG_jpg.rf.5fe14f4c981e1f10803a9ec38eb7b662.jpg', 'n03445777_6_JPEG.rf.753f7d787937e23b128feb52c2b71f71.jpg', 'n03445777_7015_JPEG.rf.c3002919a73300c33240dcc2f6256701.jpg', 'n03445777_7146_JPEG.rf.c1bc88772197cfecb9818ee08b8f51c5.jpg', 'n03445777_7238_JPEG.rf.9adda9e02afb90a89db7e6047a30a6d1.jpg', 'n03445777_7255_JPEG.rf.c2a55e73b797db63b560df2bd5b904bb.jpg', 'n03445777_7283_JPEG.rf.96b52a3a90e043b97b7f34cf4ddd3329.jpg', 'n03445777_7557_JPEG.rf.58d769350ee02a36921e52f12168af46.jpg', '11269-22883-24966_jpg.rf.ed08e32b2d0d3c4f81d48be37ca04e72.jpg', '11285-26820-17616_jpg.rf.4c46b3f7d0f4bfa84dbcc5cbf86816ca.jpg', '11298-9284-2449_jpg.rf.c3f4c76b03b01177c91e4f2c5531996e.jpg', '11490-7526-20691_jpg.rf.1be67cbb97cc22fd33704f0dae7d1800.jpg', '11601-12432-32206_jpg.rf.b15c0b5ae8532a68df27d52d4c1778a4.jpg', '11701-8618-1471_jpg.rf.ab992921fdfd509f2441f30eadb55a2b.jpg', '11728-18433-26113_jpg.rf.f1aaafa055168705b05b32e360b17e67.jpg', '11728-20503-16813_jpg.rf.35751571f9e7118d1623e866be39967c.jpg', '11738-29657-32694_jpg.rf.0f575a5f06be234b93bfc9e31baa1357.jpg', '11751-2718-7247_jpg.rf.0f10386383e186a05d0229156fc9d129.jpg', '11787-21075-8436_jpg.rf.adb579043578653104cf0142832e1712.jpg', '11787-2668-13770_jpg.rf.9b16eeb060132367e9f1b18c1679ccce.jpg', '11795-7147-31433_jpg.rf.821969e3133c7b4b73b13eea1f70b2b5.jpg', '11853-7824-15188_jpg.rf.152d7643dfb41b0c44cd15e3fefc25b3.jpg', 'B26_jpg.rf.e6db835271f21481b7ea23418e380103.jpg', 'b270f0a8a9367160_jpg.rf.9535d06350b1e2538d6505e853e24ade.jpg', 'B27_jpg.rf.90421696b76b58f0d197723d8caa5467.jpg', 'b285de6acfa65a9c_jpg.rf.4ddce4745929ccd5f57b432b632e5a1a.jpg', 'B28_jpg.rf.10bf4926e1763b131cc3f3b56fd16b0a.jpg'...
    """

    # %%%%%%%%%%%%%%%%%%%%
    # RefCOCO Dataset ●
    # %%%%%%%%%%%%%%%%%%%%
    # dataset = ReferDataset(args,
    #                        image_transforms=None,
    #                        target_transforms=None,
    #                        split=args.split,
    #                        eval_mode=True)
    """
    Cat_dict
    cat_names
    classes
    imgs
    input_ids
    ref_ids
    refer
    sentence_raws
    """
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False) # 커스텀 데이터셋 불러오기 위해 주석 ●
    # %%%%%%%%%%%%%%%%%%%%

    # --------------------------------------------------------------
    # 3) Load Model
    # --------------------------------------------------------------
    # (1) Load Mask Generator + Setting
    # print(f'sam_model_registry.keys() : {sam_model_registry.keys()}') # ['default', 'vit_h', 'vit_l', 'vit_b', 'vit_t']
    mobile_sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_model_path)  # 모델 구조 ???
    mobile_sam = mobile_sam.to(device=device)  # 모델 구조 ???
    mobile_sam.eval() # MobileSAM

    # mask_generator = SamAutomaticMaskGenerator(model=mobile_sam)
    """
    box_nms_thresh = 0.7
    crop_n_layers = 0
    crop_n_points_downscale_factor = 1
    crop_nms_thresh = 0.7
    crop_overlap_ratio = 0.34133333333
    min_mask_region_area = 0
    output_mode = 'binary_mask'
    point_grids = [array([[0.015625, 0.015625],
       [0.046875, 0.015625],
       [0.078125, 0.015625],
       ...,
       [0.921875, 0.984375],
       [0.953125, 0.984375],
       [0.984375, 0.984375]])]
    points_per_batch = 64
    pred_iou_thresh = 0.88
    predictor = <mobile_sam.predictor.SamPredictor object at 0x7fb4b89fbd60>
    stability_score_offset = 1.0
    stability_score_thresh = 0.95
    """
    mask_generator = SamAutomaticMaskGenerator(model = mobile_sam, # 모델 구조 ???
                                               points_per_side = 32,
                                               pred_iou_thresh = 0.95,
                                               stability_score_thresh = 0.95,
                                               crop_n_layers=0,
                                               crop_n_points_downscale_factor=1,
                                               min_mask_region_area=1000)
    predictor = SamPredictor(mobile_sam)  # 모델 구조 ???

    # %%%%%%%%%%%%%%%%%%%%
    # Free SOLO
    # %%%%%%%%%%%%%%%%%%%%
    # Trainer = BaselineTrainer
    # Free_SOLO = Trainer.build_model(cfg) # 모델 구조 ???
    # Free_SOLO.eval()
    #
    # DetectionCheckpointer(Free_SOLO, save_dir=cfg.OUTPUT_DIR).resume_or_load( # 모델 구조 ???
    #     cfg.MODEL.WEIGHTS, resume=args.resume
    # )
    # %%%%%%%%%%%%%%%%%%%%

    # (2) Load CLIP Model
    mode = 'ViT'  # or ViT
    assert (mode == 'Res') or (mode == 'ViT'), 'Specify mode(Res or ViT)'
    Model = clip_backbone(model_name='RN50').to(device) if mode == 'Res' else CLIPViTFM(model_name='ViT-B/32').to(device)

    # (3) Load Spacy
    nlp = spacy.load('en_core_web_lg') # Spacy의 어떤 기능을 사용한건지 ???

    # %%%%%%%%%%%%%%%%%%%%
    # RefCOCO Dataset ●
    # %%%%%%%%%%%%%%%%%%%%
    # cum_I, cum_U =0, 0 # 커스텀 데이터셋 불러오기 위해 주석
    # m_IoU = []
    # %%%%%%%%%%%%%%%%%%%%

    # --------------------------------------------------------------
    # 6) EValuation
    # --------------------------------------------------------------
    v = 0
    r = 0

    # %%%%%%%%%%%%%%%%%%%%
    # RefCOCO ●
    # %%%%%%%%%%%%%%%%%%%%
    # v = 0.85 if args.dataset == 'refcocog' else 0.95
    # %%%%%%%%%%%%%%%%%%%%

    tbar = tqdm.tqdm(image_filenames)
    with torch.no_grad(): # 자동 미분 제거 추가 ●
        # --------------------------------------------------------------
        # (1) 이미지 하나씩 처리
        # --------------------------------------------------------------
        for i, image_filename in enumerate(tbar):
            torch.cuda.empty_cache() # 사용하지 않는 캐시 삭제 추가 ●

            # %%%%%%%%%%%%%%%%%%%%
            # 메모리 최적화 실패 ●
            # %%%%%%%%%%%%%%%%%%%%
            # Garbage Collector 시도 : 실패
            # import gc
            # gc.collect()
            # crop_features = None
            # pred = None
            # Model = clip_backbone(model_name='RN50').to(device) if mode == 'Res' else CLIPViTFM(model_name='ViT-B/32').to(device)
            # del Model
            # Model = clip_backbone(model_name='RN50').to(device) if mode == 'Res' else CLIPViTFM(model_name='ViT-B/32').to(device)
            # %%%%%%%%%%%%%%%%%%%%

            # --------------------------------------------------------------
            # (2) Data Load & Preprocessing
            # --------------------------------------------------------------
            hwc_image = cv2.imread(f'{args.base_dataset_path}/{image_filename}')
            """
            hwc_image shape : (이미지 높이, 이미지 너비, 3)
            hwc_image : [[[ 58  77  11],  [ 61  80  14],  [ 67  86  22],  ...,  [ 45  62  26],  [ 40  56  27],  [ 36  52  25]],, [[ 59  78  12],  [ 62  81  15],  [ 67  86  22],  ...,  [ 43  60  24],  [ 39  55  26],  [ 36  52  25]],, [[ 58  79  12],  [ 61  82  15],  [ 65  86  21],
            """
            hwc_image = cv2.cvtColor(hwc_image, cv2.COLOR_BGR2RGB)
            """
            hwc_image shape : (이미지 높이, 이미지 너비, 3)
            hwc_image : [[[ 58  77  11],  [ 61  80  14],  [ 67  86  22],  ...,  [ 45  62  26],  [ 40  56  27],  [ 36  52  25]],, [[ 59  78  12],  [ 62  81  15],  [ 67  86  22],  ...,  [ 43  60  24],  [ 39  55  26],  [ 36  52  25]],, [[ 58  79  12],  [ 61  82  15],  [ 65  86  21],  ...,  [ 37  52  19],  [ 33  48  19],  [ 29  43  17]],, ...,, [[ 71  94  38],  [ 66  89  33],  [ 60  83  27],  ...,  [ 14  24   0],  [ 15  25   1],  [ 16  26   2]],, [[ 93 118  61],  [ 82 107  50],  [ 66  91  34],  ...,  [ 16  24   1],  [ 17  25   2],  [ 18  26   3]],, [[ 94 119  62],  [ 85 113  55],  [ 76 101  44],  ...,  [ 39  47  24],  [ 41  46  24],  [ 40  48  25]]]
            """
            chw_image = hwc_image.transpose(2, 0, 1)
            """
            chw_image.shape : (3, 이미지 높이, 이미지 너비)
            chw_image : [[[ 58  61  67 ...  45  40  36],  [ 59  62  67 ...  43  39  36],  [ 58  61  65 ...  37  33  29],  ...,  [135 137 141 ... 165 159 156],  [115 121 130 ... 169 165 163],  [101 109 122 ... 160 152 147]],, [[ 77  80  86 ...  62  56  52],  [ 78  81  86 ...  60  
            """
            image = [{'image': torch.from_numpy(np.expand_dims(chw_image, axis=0)), 'height': torch.tensor([chw_image.shape[1]]),'width': torch.tensor([chw_image.shape[2]])}]
            """
            image shape list 1 -> dict 3 -> image (1, 3, 이미지 높이, 이미지 너비) height (1) width (1)
            image : [{'image': tensor([[[[ 58,  61,  67,  ...,  45,  40,  36],
              [ 59,  62,  67,  ...,  43,  39,  36],
              [ 58,  61,  65,  ...,  37,  33,  29],
              ...,
              [135, 137, 141,  ..., 165, 159, 156],
              [115, 121, 130,  ..., 169, 165, 163],
              [101, 109, 122,  ..., 160, 152, 147]],
    
             [[ 77,  80,  86,  ...,  62,  56,  52],
              [ 78,  81,  86,  ...,  60,  55,  52],
              [ 79,  82,  86,  ...,  52,  48,  43],
              ...,
              [131, 133, 137,  ..., 172, 168, 165],
              [111, 117, 126,  ..., 176, 174, 172],
              [ 97, 105, 118,  ..., 167, 161, 156]],
    
             [[ 11,  14,  22,  ...,  26,  27,  25],
              [ 12,  15,  22,  ...,  24,  26,  25],
              [ 12,  15,  21,  ...,  19,  19,  17],
              ...,
              [119, 121, 125,  ..., 156, 151, 148],
              [ 99, 105, 114,  ..., 160, 157, 155],
              [ 85,  93, 106,  ..., 151, 144, 139]]]], dtype=torch.uint8), 'height': tensor([1856]), 'width': tensor([1044])}]
            """

            sentence_raw = [['Large sandy area, brown or tan or white in the picture, not people, not green, not blue, not small area']] # [['Large sandy area, brown or tan or white in the picture']] # [["wide sandy area in the picture"]] # [["white golf ball in the picture"]]

            # %%%%%%%%%%%%%%%%%%%%
            # Custom Dataset + FreeSOLO ●
            # %%%%%%%%%%%%%%%%%%%%
            # pil_img = PIL.Image.open(f'{args.base_dataset_path}/{image_filename}')
            # resized_img = T.Resize(800)(pil_img)
            # tensor_img = T.ToTensor()(resized_img)
            # tensor_normalized_img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor_img)
            # # 골프공 데이터 평균, 분산 : (0.50320652139677, 0.5165877819951724, 0.44484686164115467), (0.1920507651921667, 0.19278384706607293, 0.21140713878626044)
            #
            # image = [{'image': tensor_normalized_img.unsqueeze(dim=0), 'height': torch.tensor([pil_img.size[1]]), 'width': torch.tensor([pil_img.size[0]]), 'file_name': [image_filename]}]
            """
            iamge len : 1 -> 4 -> [[1, 3, 800, 1422], [1], [1], [1]]
            image : [{'image': tensor([[[[-1.0733, -1.0048, -0.8678,  ..., -1.2959, -1.3987, -1.4843],
              [-1.0904, -1.0219, -0.9020,  ..., -1.3987, -1.4672, -1.5357],
              [-1.0904, -1.0390, -0.9363,  ..., -1.4672, -1.6042, -1.7412],
              ...,
              [ 0.3652,  0.3823,  0.3652,  ...,  0.6221,  0.4337,  0.2967],
              [ 0.0398,  0.1426,  0.2624,  ...,  0.7591,  0.6906,  0.6392],
              [-0.2856, -0.0972,  0.1597,  ...,  0.7077,  0.5878,  0.5022]],
    
             [[-0.6702, -0.6001, -0.4601,  ..., -0.9153, -1.0203, -1.1078],
              [-0.6527, -0.5826, -0.4601,  ..., -1.0203, -1.0903, -1.1779],
              [-0.6352, -0.5651, -0.4601,  ..., -1.1253, -1.2829, -1.4230],
              ...,
              [ 0.4328,  0.4503,  0.4328,  ...,  0.9055,  0.7304,  0.5903],
              [ 0.1001,  0.2052,  0.3277,  ...,  1.0280,  0.9930,  0.9405],
              [-0.2325, -0.0399,  0.2227,  ...,  0.9755,  0.8880,  0.8004]],
    
             [[-1.5953, -1.5081, -1.3513,  ..., -1.3339, -1.3339, -1.3513],
              [-1.5779, -1.5081, -1.3513,  ..., -1.4384, -1.4036, -1.4210],
              [-1.5604, -1.5081, -1.3861,  ..., -1.5081, -1.5604, -1.6302],
              ...,
              [ 0.4439,  0.4614,  0.4439,  ...,  0.8274,  0.6531,  0.5136],
              [ 0.1128,  0.2173,  0.3393,  ...,  0.9668,  0.9145,  0.8622],
              [-0.2184, -0.0267,  0.2348,  ...,  0.9145,  0.8099,  0.7228]]]]), 'height': tensor([1044]), 'width': tensor([1856]), 'file_name': ['golf ball in bunker1.jpg']}]
            """
            # sentence_raw = [["sandy area in the picture"]]  # [["golf ball somewhere in the picture"]]

            # %%%%%%%%%%%%%%%%%%%%
            # RefCOCO Dataset + FreeSOLO ●
            # %%%%%%%%%%%%%%%%%%%%
            # image, target, clip_embedding, sentence_raw = data
            """
            image len : 1 -> 9 -> [[1, 3, 이미지의 width, 이미지의 height], [1], [1], [1], [1], [1], [], [], []]
            [{'image': tensor([[[[-0.6281, -0.4397, -0.1143,  ..., -0.4739, -0.4397, -0.4226],
              [-0.7822, -0.5767, -0.2171,  ..., -0.4568, -0.4226, -0.4054],
              [-1.0562, -0.8335, -0.4226,  ..., -0.4397, -0.4054, -0.3712],
              ...,
              [ 0.2967,  0.3138,  0.3309,  ...,  0.3309,  0.3652,  0.3823],
              [ 0.2967,  0.3138,  0.3481,  ...,  0.3481,  0.3823,  0.3994],
              [ 0.2967,  0.3138,  0.3481,  ...,  0.3481,  0.3823,  0.3994]],

             [[-0.9678, -0.7052, -0.2500,  ..., -1.3529, -1.3529, -1.3529],
              [-1.0553, -0.8277, -0.4251,  ..., -1.3354, -1.3354, -1.3354],
              [-1.1954, -1.0203, -0.7402,  ..., -1.3179, -1.3004, -1.3004],
              ...,
              [ 0.6779,  0.6779,  0.6779,  ...,  0.7654,  0.8004,  0.8179],
              [ 0.6779,  0.6779,  0.6954,  ...,  0.7829,  0.8179,  0.8354],
              [ 0.6779,  0.6779,  0.6954,  ...,  0.8004,  0.8354,  0.8529]],

             [[-1.1770, -1.0201, -0.7413,  ..., -1.7347, -1.7522, -1.7522],
              [-1.2467, -1.1073, -0.8633,  ..., -1.7173, -1.7347, -1.7347],
              [-1.3687, -1.2641, -1.0898,  ..., -1.6999, -1.6999, -1.6999],
              ...,
              [ 1.0888,  1.0888,  1.1062,  ...,  1.1759,  1.1934,  1.1934],
              [ 1.0888,  1.1062,  1.1237,  ...,  1.2108,  1.2282,  1.2282],
              [ 1.0888,  1.1062,  1.1237,  ...,  1.2282,  1.2457,  1.2457]]]]), 'height': tensor([428]), 'width': tensor([640]), 'file_name': ['COCO_train2014_000000580957.jpg'], 'cat_name': ['bowl'], 'img_id': [tensor([580957])], 'coco_instance_gt': [], 'coco_instance_gt_box': [], 'coco_instance_cat': []}]
            """
            """
            target.shape : (1, 428, 640)
            tensor([[[0, 0, 0,  ..., 0, 0, 0],
                     [0, 0, 0,  ..., 1, 1, 1],
                     [0, 0, 0,  ..., 1, 1, 1],
                     ...,
                     [0, 0, 0,  ..., 0, 0, 0],
                     [0, 0, 0,  ..., 0, 0, 0],
                     [0, 0, 0,  ..., 0, 0, 0]]], device='cuda:0', dtype=torch.uint8)
            """
            """
            clip_embedding.shape : (1, 77, 4)
            tensor([[[49406, 49406, 49406, 49406],
             [ 3814,  4531,  1579,  1579],
             [ 2403,   530,  4531,  5168],
             [  518,  1253,   530,  7067],
             [ 3326,  1155,   518,  1155],
             [  753,  5253,  1253,  5253],
             [ 1033, 49407,  1155, 49407],
             [  862,     0,  5253,     0],
             [ 1551,     0,   269,     0],
             [49407,     0, 49407,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0],
             [    0,     0,     0,     0]]], device='cuda:0', dtype=torch.int32)
            """
            """
            sentence_raw len : 4
            sentence_raw : [('bowl behind the others can only see part',), ('Dish in top right corner',), ('White dish in the top right corner.',), ('white pot upper right corner',)]
            """
            # clip_embedding, target = clip_embedding.squeeze(1).to(device), target.to(device)
            # %%%%%%%%%%%%%%%%%%%%

            # --------------------------------------------------------------
            # (3) Visual Feature Embedding
            # --------------------------------------------------------------
            # 1] Mask Generator 예측
            mask_generator_pred = mask_generator.generate(hwc_image)
            """
            mask_generator_pred shape : list 마스크 개수 -> dict 7
            mask_generator_pred[0] : {'segmentation': array([[ True,  True,  True, ...,  True,  True,  True],
               [ True,  True,  True, ...,  True,  True,  True],
               [ True,  True,  True, ...,  True,  True,  True],
               ...,
               [False, False, False, ..., False, False, False],
               [False, False, False, ..., False, False, False],
               [False, False, False, ..., False, False, False]]), 'area': 557089, 'bbox': [0, 0, 1855, 466], 'predicted_iou': 1.0176726579666138, 'point_coords': [[377.0, 48.9375]], 'stability_score': 0.9742143750190735, 'crop_box': [0, 0, 1856, 1044]}
            """
            torch.cuda.empty_cache()  # 사용하지 않는 캐시 삭제 추가 ●

            sorted_mask_generator_pred = sorted(mask_generator_pred, key=(lambda x: x['area']), reverse=True)
            """
            sorted_mask_generator_pred.shape : list 마스크 개수 -> dict 7
            sorted_mask_generator_pred[0] : {'segmentation': array([[False, False, False, ..., False, False, False],
               [False, False, False, ..., False, False, False],
               [False, False, False, ..., False, False, False],
               ...,
               [False, False, False, ..., False, False, False],
               [False, False, False, ..., False, False, False],
               [False, False, False, ..., False, False, False]]), 'area': 1251199, 'bbox': [0, 171, 1855, 870], 'predicted_iou': 1.0059185028076172, 'point_coords': [[29.0, 962.4375]], 'stability_score': 0.9520606994628906, 'crop_box': [0, 0, 1856, 1044]}
            """
            pred_masks = torch.from_numpy(np.array([sorted_mask['segmentation'] for sorted_mask in sorted_mask_generator_pred])).to(device)
            """
            pred_masks.shape : (마스크 개수, 이미지 높이, 이미지 너비)
            pred_masks : tensor([[[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            [[ True,  True,  True,  ...,  True,  True,  True],
             [ True,  True,  True,  ...,  True,  True,  True],
             [ True,  True,  True,  ...,  True,  True,  True],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            [[ True,  True,  True,  ...,  True,  True,  True],
             [ True,  True,  True,  ...,  True,  True,  True],
             [ True,  True,  True,  ...,  True,  True,  True],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            ...,
    
            [[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            [[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            [[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]]])
            """

            def xywh_to_xyxy(x,y,w,h):
                return x, y, x+w, y+h
            pred_boxes = torch.from_numpy(np.array([xywh_to_xyxy(*sorted_mask['bbox']) for sorted_mask in sorted_mask_generator_pred]))
            """
            pred_boxes.shape : (마스크 개수, 4)
            pred_boxes : tensor([[   0,  171, 1855,  870],
                [   0,    0, 1855,  669],
                [   0,    0, 1855,  466],
                [  97,  637,   71,   81],
                [ 853,  514,   83,   59],
                [ 264,  472,   94,   53],
                [1800,  221,   25,  202],
                [ 596,  955,   63,   62],
                [ 103,  548,  122,   30],
                [  60,  484,   65,   52],
                [  89,  821,   64,   61],
                [ 335,  766,  126,   85],
                [ 917,  250,  121,   37],
                [  88,  836,   50,   46],
                [ 184,  236,   65,   50],
                [ 614,  806,   68,   35],
                [ 608,  682,   52,   84],
                [   3,  469,   53,   38],
                [1475,  417,   22,   47], 
                ...])
            """

            # +] Mask Generator 예측 시각화
            if args.plot_mask_pred:
                plot_mask_generator_pred(image_filename, hwc_image, mask_generator_pred)

            # %%%%%%%%%%%%%%%%%%%%
            # Free SOLO ●
            # %%%%%%%%%%%%%%%%%%%%
            # mask_generator_pred = Free_SOLO(image)[0] # 모델 구조 ???
            """
            mask_generator_pred len : dict 1 -> instances 36
            mask_generator_pred : {'instances': Instances(num_instances=46, image_height=428, image_width=640, fields=[pred_classes: tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           device='cuda:0'), scores: tensor([0.7973, 0.4304, 0.3678, 0.3214, 0.2933, 0.2672, 0.2518, 0.2480, 0.1854,
            0.1796, 0.1780, 0.1644, 0.1605, 0.1559, 0.1543, 0.1474, 0.1417, 0.1239,
            0.1172, 0.1133, 0.1123, 0.1113, 0.1112, 0.1089, 0.1086, 0.1061, 0.1043,
            0.1026, 0.0921, 0.0850, 0.0819, 0.0802, 0.0778, 0.0759, 0.0749, 0.0690,
            0.0667, 0.0650, 0.0630, 0.0623, 0.0605, 0.0591, 0.0581, 0.0569, 0.0562,
            0.0519], device='cuda:0', grad_fn=<IndexBackward0>), category_scores: tensor([0.8096, 0.4509, 0.4644, 0.3463, 0.3065, 0.2700, 0.2614, 0.2985, 0.5195,
            0.3369, 0.3746, 0.3264, 0.2787, 0.2287, 0.2973, 0.2961, 0.3570, 0.2649,
            0.2310, 0.1592, 0.7809, 0.7646, 0.1135, 0.3872, 0.5991, 0.2411, 0.1090,
            0.1920, 0.6629, 0.1352, 0.5435, 0.2537, 0.4359, 0.3325, 0.2545, 0.2645,
            0.2917, 0.3334, 0.2063, 0.4177, 0.2162, 0.1463, 0.1048, 0.1489, 0.3363,
            0.2370], device='cuda:0'), maskness: tensor([0.9847, 0.9549, 0.9362, 0.9667, 0.9571, 0.9896, 0.9688, 0.9529, 0.9389,
            0.9414, 0.9313, 0.9143, 0.9337, 0.9493, 0.9408, 0.9472, 0.9353, 0.9409,
            0.9393, 0.9260, 0.9837, 0.9848, 0.9797, 0.9519, 0.9793, 0.9470, 0.9570,
            0.8724, 0.9862, 0.9651, 0.9839, 0.9430, 0.9594, 0.9664, 0.9704, 0.9655,
            0.9519, 0.9652, 0.9684, 0.9856, 0.9705, 0.9636, 0.9650, 0.9591, 0.9596,
            0.9703], device='cuda:0', grad_fn=<IndexBackward0>), pred_masks: tensor([[[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            [[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            [[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            ...,
    
            [[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            [[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            [[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]]], device='cuda:0'), pred_embs: tensor([[ 0.0543,  0.0920,  0.0428,  ...,  0.1230,  0.0671, -0.0899],
            [-0.0032, -0.0351, -0.0330,  ...,  0.0042, -0.0007,  0.0998],
            [-0.0156,  0.1707,  0.0830,  ..., -0.1202,  0.0328,  0.1264],
            ...,
            [ 0.1281,  0.0801,  0.0572,  ...,  0.0291,  0.0566,  0.0213],
            [ 0.0404,  0.0154,  0.0156,  ...,  0.0754,  0.0599,  0.0677],
            [ 0.1510,  0.0160,  0.0593,  ...,  0.1745,  0.1200,  0.1113]],
           device='cuda:0'), pred_boxes: Boxes(tensor([[ 1.6413e+02,  1.5438e+02,  5.9413e+02,  3.8738e+02],
            [ 7.6500e+01,  9.3494e+01,  1.9950e+02,  2.5949e+02],
            [ 1.5508e+02,  5.4742e+01,  6.2308e+02,  3.3174e+02],
            [-5.0000e-01,  1.4500e+01,  1.3350e+02,  1.6650e+02],
            [ 1.4950e+02,  2.9500e+01,  4.2650e+02,  1.3150e+02],
            [ 7.9324e+01,  1.5350e+02,  3.5132e+02,  4.2750e+02],
            [ 1.2350e+02,  2.9748e+02,  6.3950e+02,  4.2748e+02],
            [-5.0000e-01,  7.8500e+01,  5.5500e+01,  1.8850e+02],
            [ 1.6097e+02,  5.0231e+01,  5.9797e+02,  3.8123e+02],
            [ 1.6078e+02,  5.0943e+01,  5.8578e+02,  3.5794e+02],
            [ 1.6107e+02,  6.8542e+01,  5.9507e+02,  3.0854e+02],
            [ 1.6046e+02,  5.3622e+01,  5.8146e+02,  3.8062e+02],
            [ 1.6301e+02,  6.9561e+01,  5.7801e+02,  2.9856e+02],
            [ 1.5350e+02,  3.7500e+01,  5.4450e+02,  2.7950e+02],
            [ 1.5694e+02,  5.1816e+01,  6.1894e+02,  3.3382e+02],
            [ 1.5678e+02,  4.8552e+01,  5.6878e+02,  3.3955e+02],
            [ 5.8149e+00,  6.0500e+01,  1.8681e+02,  2.5050e+02],
            [ 1.6019e+02,  5.2176e+01,  5.9419e+02,  3.4218e+02],
            [ 1.6072e+02,  5.3466e+01,  5.8572e+02,  3.4647e+02],
            [ 1.4950e+02,  3.7500e+01,  5.1150e+02,  2.7750e+02],
            [ 1.6425e+02,  1.5050e+02,  5.9425e+02,  3.8750e+02],
            [ 1.6585e+02,  1.5645e+02,  5.8685e+02,  3.8745e+02],
            [ 4.7050e+02,  1.3500e+01,  6.3950e+02,  1.2550e+02],
            [ 1.5962e+02,  6.6234e+01,  5.9962e+02,  3.8223e+02],
            [ 1.6505e+02,  1.0763e+02,  5.9905e+02,  3.8463e+02],
            [ 2.9759e+00,  5.0490e+01,  1.8998e+02,  2.5349e+02],
            [ 5.9350e+02,  1.2750e+02,  6.3950e+02,  2.6350e+02],
            [-5.0000e-01,  2.4500e+01,  9.6500e+01,  1.6850e+02],
            [ 1.6471e+02,  1.6650e+02,  5.8371e+02,  3.8750e+02],
            [-5.0000e-01,  3.2500e+01,  1.9350e+02,  2.5650e+02],
            [ 1.6412e+02,  1.5248e+02,  5.9612e+02,  3.8848e+02],
            [ 2.5130e+01,  6.4489e+01,  1.8713e+02,  2.5349e+02],
            [ 7.0500e+01,  8.4500e+01,  1.9950e+02,  2.5950e+02],
            [ 6.3015e+01,  7.4500e+01,  1.9901e+02,  2.5850e+02],
            [-5.0000e-01,  1.2500e+01,  1.1850e+02,  1.5350e+02],
            [-5.0000e-01,  1.3500e+01,  1.1850e+02,  1.6550e+02],
            [-5.0000e-01,  7.5500e+01,  5.6500e+01,  1.7750e+02],
            [ 7.3500e+01,  8.6489e+01,  2.0350e+02,  2.6149e+02],
            [-5.0000e-01,  1.1500e+01,  1.1650e+02,  1.5750e+02],
            [ 1.6563e+02,  1.6050e+02,  5.8663e+02,  3.8850e+02],
            [-4.9166e-01,  1.2500e+01,  1.1951e+02,  1.5150e+02],
            [ 1.6334e+02,  3.2950e+02,  5.9534e+02,  4.2750e+02],
            [-5.0000e-01,  2.0500e+01,  1.8450e+02,  2.4450e+02],
            [ 1.4550e+02,  3.2029e+01,  4.4250e+02,  2.4003e+02],
            [ 8.0492e+01,  1.0150e+02,  2.0149e+02,  2.6050e+02],
            [ 1.4250e+02,  2.9500e+01,  4.2450e+02,  1.3650e+02]], device='cuda:0'))])}
            """
            # pred_masks = mask_generator_pred['instances'].pred_masks
            """
            pred_masks.shape : tensor (46, 428, 640)
            tensor([[[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            [[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            [[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            ...,
    
            [[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            [[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]],
    
            [[False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             ...,
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False],
             [False, False, False,  ..., False, False, False]]], device='cuda:0')
            """
            # pred_boxes = mask_generator_pred['instances'].pred_boxes
            """
            pred_boxes.shape : Boxes (36) -> tensor (46, 4)
            Boxes(tensor([[ 1.6413e+02,  1.5438e+02,  5.9413e+02,  3.8738e+02],
            [ 7.6500e+01,  9.3494e+01,  1.9950e+02,  2.5949e+02],
            [ 1.5508e+02,  5.4742e+01,  6.2308e+02,  3.3174e+02],
            [-5.0000e-01,  1.4500e+01,  1.3350e+02,  1.6650e+02],
            [ 1.4950e+02,  2.9500e+01,  4.2650e+02,  1.3150e+02],
            [ 7.9324e+01,  1.5350e+02,  3.5132e+02,  4.2750e+02],
            [ 1.2350e+02,  2.9748e+02,  6.3950e+02,  4.2748e+02],
            [-5.0000e-01,  7.8500e+01,  5.5500e+01,  1.8850e+02],
            [ 1.6097e+02,  5.0231e+01,  5.9797e+02,  3.8123e+02],
            [ 1.6078e+02,  5.0943e+01,  5.8578e+02,  3.5794e+02],
            [ 1.6107e+02,  6.8542e+01,  5.9507e+02,  3.0854e+02],
            [ 1.6046e+02,  5.3622e+01,  5.8146e+02,  3.8062e+02],
            [ 1.6301e+02,  6.9561e+01,  5.7801e+02,  2.9856e+02],
            [ 1.5350e+02,  3.7500e+01,  5.4450e+02,  2.7950e+02],
            [ 1.5694e+02,  5.1816e+01,  6.1894e+02,  3.3382e+02],
            [ 1.5678e+02,  4.8552e+01,  5.6878e+02,  3.3955e+02],
            [ 5.8149e+00,  6.0500e+01,  1.8681e+02,  2.5050e+02],
            [ 1.6019e+02,  5.2176e+01,  5.9419e+02,  3.4218e+02],
            [ 1.6072e+02,  5.3466e+01,  5.8572e+02,  3.4647e+02],
            [ 1.4950e+02,  3.7500e+01,  5.1150e+02,  2.7750e+02],
            [ 1.6425e+02,  1.5050e+02,  5.9425e+02,  3.8750e+02],
            [ 1.6585e+02,  1.5645e+02,  5.8685e+02,  3.8745e+02],
            [ 4.7050e+02,  1.3500e+01,  6.3950e+02,  1.2550e+02],
            [ 1.5962e+02,  6.6234e+01,  5.9962e+02,  3.8223e+02],
            [ 1.6505e+02,  1.0763e+02,  5.9905e+02,  3.8463e+02],
            [ 2.9759e+00,  5.0490e+01,  1.8998e+02,  2.5349e+02],
            [ 5.9350e+02,  1.2750e+02,  6.3950e+02,  2.6350e+02],
            [-5.0000e-01,  2.4500e+01,  9.6500e+01,  1.6850e+02],
            [ 1.6471e+02,  1.6650e+02,  5.8371e+02,  3.8750e+02],
            [-5.0000e-01,  3.2500e+01,  1.9350e+02,  2.5650e+02],
            [ 1.6412e+02,  1.5248e+02,  5.9612e+02,  3.8848e+02],
            [ 2.5130e+01,  6.4489e+01,  1.8713e+02,  2.5349e+02],
            [ 7.0500e+01,  8.4500e+01,  1.9950e+02,  2.5950e+02],
            [ 6.3015e+01,  7.4500e+01,  1.9901e+02,  2.5850e+02],
            [-5.0000e-01,  1.2500e+01,  1.1850e+02,  1.5350e+02],
            [-5.0000e-01,  1.3500e+01,  1.1850e+02,  1.6550e+02],
            [-5.0000e-01,  7.5500e+01,  5.6500e+01,  1.7750e+02],
            [ 7.3500e+01,  8.6489e+01,  2.0350e+02,  2.6149e+02],
            [-5.0000e-01,  1.1500e+01,  1.1650e+02,  1.5750e+02],
            [ 1.6563e+02,  1.6050e+02,  5.8663e+02,  3.8850e+02],
            [-4.9166e-01,  1.2500e+01,  1.1951e+02,  1.5150e+02],
            [ 1.6334e+02,  3.2950e+02,  5.9534e+02,  4.2750e+02],
            [-5.0000e-01,  2.0500e+01,  1.8450e+02,  2.4450e+02],
            [ 1.4550e+02,  3.2029e+01,  4.4250e+02,  2.4003e+02],
            [ 8.0492e+01,  1.0150e+02,  2.0149e+02,  2.6050e+02],
            [ 1.4250e+02,  2.9500e+01,  4.2450e+02,  1.3650e+02]], device='cuda:0'))
            """
            # %%%%%%%%%%%%%%%%%%%%

            if len(pred_masks) == 0:
                print('No pred masks')
                continue

            # 2] Global 이미지 Resize + Masking
            original_imgs = image[0]['image'].to(pred_masks.device)
            """
            original_imgs.shape : (1, 3, 이미지 높이, 이미지 너비)
            original_imgs : tensor([[[[ 58,  61,  67,  ...,  45,  40,  36],
              [ 59,  62,  67,  ...,  43,  39,  36],
              [ 58,  61,  65,  ...,  37,  33,  29],
              ...,
              [135, 137, 141,  ..., 165, 159, 156],
              [115, 121, 130,  ..., 169, 165, 163],
              [101, 109, 122,  ..., 160, 152, 147]],
    
             [[ 77,  80,  86,  ...,  62,  56,  52],
              [ 78,  81,  86,  ...,  60,  55,  52],
              [ 79,  82,  86,  ...,  52,  48,  43],
              ...,
              [131, 133, 137,  ..., 172, 168, 165],
              [111, 117, 126,  ..., 176, 174, 172],
              [ 97, 105, 118,  ..., 167, 161, 156]],
    
             [[ 11,  14,  22,  ...,  26,  27,  25],
              [ 12,  15,  22,  ...,  24,  26,  25],
              [ 12,  15,  21,  ...,  19,  19,  17],
              ...,
              [119, 121, 125,  ..., 156, 151, 148],
              [ 99, 105, 114,  ..., 160, 157, 155],
              [ 85,  93, 106,  ..., 151, 144, 139]]]], dtype=torch.uint8)
            """
            resized_imgs = torch.stack([T.Resize((resized_height, resized_width))(img.to(pred_masks.device)) for img in image[0]['image']], dim=0)
            """
            resized_imgs.shape : (1, 3, 224, 224)
            resized_imgs : tensor([[[[ 76,  65,  65,  ...,  71, 101,  40],
              [ 63,  73,  46,  ...,  64,  83, 117],
              [ 30,  78,  73,  ...,  56,  32,  49],
              ...,
              [124, 126, 122,  ..., 106, 180, 147],
              [129, 132, 105,  ..., 145, 151, 164],
              [141, 134,  91,  ..., 170, 140, 168]],
    
             [[ 96,  82,  86,  ...,  96, 110,  56],
              [ 91,  98,  69,  ...,  89, 105, 122],
              [ 70, 114,  97,  ...,  81,  69,  40],
              ...,
              [120, 122, 119,  ..., 108, 182, 149],
              [125, 128, 102,  ..., 147, 153, 166],
              [137, 130,  78,  ..., 172, 126, 172]],
    
             [[ 34,  40,  44,  ...,  32,  79,  14],
              [ 27,  53,  17,  ...,  25,  57,  93],
              [  1,  66,  34,  ...,  17,   0,  27],
              ...,
              [108, 110, 100,  ...,  94, 168, 135],
              [113, 116,  83,  ..., 133, 139, 152],
              [125, 118,  69,  ..., 158, 117, 157]]]], dtype=torch.uint8)
            """

            # %%%%%%%%%%%%%%%%%%%%
            # RefCOCO Dataset ●
            # %%%%%%%%%%%%%%%%%%%%
            # original_imgs = torch.stack([T.Resize((height, width))(img.to(pred_masks.device)) for img, height, width in
            #                              zip(image[0]['image'], image[0]['height'], image[0]['width'])], dim=0)
            """
                        original_imgs.shape : (1, 3, 428, 640)
                        original_imgs : tensor([[[[-0.6100, -0.0581,  0.5194,  ..., -0.5521, -0.4769, -0.4226],
                          [-1.0033, -0.3723,  0.2925,  ..., -0.5059, -0.4485, -0.3861],
                          [-1.2073, -0.7970, -0.0724,  ..., -0.4715, -0.4298, -0.4033],
                          ...,
                          [ 0.3013,  0.3114,  0.3916,  ...,  0.3309,  0.2898,  0.2935],
                          [ 0.3041,  0.3413,  0.4107,  ...,  0.3280,  0.3257,  0.3645],
                          [ 0.3041,  0.3503,  0.3713,  ...,  0.3719,  0.3510,  0.3920]],

                         [[-0.8984, -0.2313,  0.2830,  ..., -1.3658, -1.3453, -1.3453],
                          [-1.1498, -0.6836,  0.0855,  ..., -1.3331, -1.3179, -1.3004],
                          [-1.2871, -1.0627, -0.3478,  ..., -1.3034, -1.3301, -1.3196],
                          ...,
                          [ 0.7100,  0.6876,  0.7544,  ...,  0.7479,  0.7182,  0.7241],
                          [ 0.6809,  0.6885,  0.7463,  ...,  0.7449,  0.7601,  0.7997],
                          [ 0.6779,  0.6954,  0.6918,  ...,  0.7898,  0.7928,  0.8377]],

                         [[-1.1424, -0.7316, -0.3980,  ..., -1.7272, -1.7272, -1.7446],
                          [-1.3302, -1.0387, -0.4891,  ..., -1.6946, -1.6999, -1.6999],
                          [-1.3011, -1.2127, -0.7718,  ..., -1.6620, -1.6937, -1.7173],
                          ...,
                          [ 1.1557,  1.1106,  1.1680,  ...,  1.1934,  1.1189,  1.0901],
                          [ 1.0941,  1.1168,  1.1743,  ...,  1.1904,  1.1670,  1.1775],
                          [ 1.0964,  1.1237,  1.1201,  ...,  1.2351,  1.2207,  1.2381]]]],
                       device='cuda:0')
                        """
            # resized_imgs = torch.stack([T.Resize((resized_height, resized_width))(img.to(pred_masks.device)) for img in image[0]['image']], dim=0)
            """
            resized_imgs.shape : (1, 3, 224, 224)
            resized_imgs : tensor([[[[-0.2160,  0.7043,  0.7062,  ..., -0.3562, -0.4416, -0.4577],
              [-0.9904,  0.6919,  0.6732,  ..., -0.5223, -0.4481, -0.4376],
              [-1.3080,  0.1317,  0.7491,  ..., -0.4178, -0.5389, -0.4886],
              ...,
              [ 0.3786,  0.2962,  0.4088,  ...,  0.3601,  0.3224,  0.3378],
              [ 0.3029,  0.3590,  0.3026,  ...,  0.3356,  0.3763,  0.3093],
              [ 0.3477,  0.3911,  0.3740,  ...,  0.4215,  0.3457,  0.3432]],

             [[-0.4553,  0.4793,  0.4838,  ..., -1.2105, -1.3238, -1.3304],
              [-1.1767,  0.4252,  0.4672,  ..., -1.0678, -1.2915, -1.3325],
              [-1.3109, -0.1478,  0.5350,  ..., -0.8562, -1.3053, -1.3179],
              ...,
              [ 0.7266,  0.6424,  0.7490,  ...,  0.8212,  0.7392,  0.7724],
              [ 0.6813,  0.7329,  0.6539,  ...,  0.7506,  0.8020,  0.7159],
              [ 0.6921,  0.7333,  0.7402,  ...,  0.7948,  0.7304,  0.7779]],

             [[-0.8862, -0.1904, -0.1122,  ..., -1.6325, -1.6210, -1.7123],
              [-1.2567, -0.2788, -0.0498,  ..., -1.5029, -1.5950, -1.6944],
              [-1.2235, -0.6671,  0.0411,  ..., -1.3057, -1.5853, -1.6342],
              ...,
              [ 1.1548,  1.0784,  1.1770,  ...,  1.1908,  1.1847,  1.1685],
              [ 1.1067,  1.1522,  1.1119,  ...,  1.1759,  1.2133,  1.1245],
              [ 1.1204,  1.1614,  1.1656,  ...,  1.2052,  1.1472,  1.2008]]]],
           device='cuda:0')
            """
            # %%%%%%%%%%%%%%%%%%%%

            # 3] Local 이미지 ResizeCrop + Masking
            resize_crop_masked_imgs = []
            pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).to(pred_masks.device) # 용도가 뭐임??? 예측된 마스크 영역 1개만 그대로 남기고, 나머지 영역은 pixel_mean 값으로 만들기!!!
            """
            pixel_mean.shape : (1, 3, 1, 1)
            pixel_mean : tensor([[[[0.4850]],
             [[0.4560]],
             [[0.4060]]]], device='cuda:0')
            """
            for pred_box, pred_mask in zip(pred_boxes.__iter__(), pred_masks):
                pred_mask, pred_box = pred_mask.type(torch.uint8), pred_box.type(torch.int)
                """
                pred_mask.shape : torch.Size([이미지 높이, 이미지 너비])
                pred_mask : tensor([[0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                ...,
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0],
                [0, 0, 0,  ..., 0, 0, 0]], dtype=torch.uint8)
                """
                """
                pred_box.shape : 4
                pred_box : tensor([   0,  171, 1855,  870], dtype=torch.int32)
                """
                masked_img = original_imgs * pred_mask[None, None, ...] + (1 - pred_mask[None, None, ...]) * pixel_mean # 용도가 뭐임??? 예측된 마스크 영역 1개만 그대로 남기고, 나머지 영역은 pixel_mean 값으로 만들기!!!
                """
                masked_img.shape : torch.Size([1, 3, 이미지 높이, 이미지 너비])
                masked_img : tensor([[[[0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
                  [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
                  [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
                  ...,
                  [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
                  [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
                  [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850]],
        
                 [[0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
                  [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
                  [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
                  ...,
                  [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
                  [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
                  [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560]],
        
                 [[0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
                  [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
                  [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
                  ...,
                  [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
                  [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
                  [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060]]]])
                """
                x1, y1, x2, y2 = int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])
                """
                x1 : 164, x2 : 594, y1 : 154, y2 : 387
                """

                resize_crop_masked_img = TF.resized_crop(masked_img.squeeze(0), y1, x1, (y2 - y1), (x2 - x1),(resized_height, resized_width))

                # %%%%%%%%%%%%%%%%%%%%
                # Free SOLO ●
                # %%%%%%%%%%%%%%%%%%%%
                resize_crop_masked_img = TF.resized_crop(masked_img.squeeze(0), y1, x1, (y2 - y1), (x2 - x1), (resized_height, resized_width))
                """
                resize_crop_masked_img.shape : torch.Size([3, 224, 224])
                resize_crop_masked_img : tensor([[[  0.4850,   0.4850,   0.4850,  ...,   0.4850,   0.4850,   0.4850],
                 [  0.4850,   0.4850,   0.4850,  ...,   0.4850,   0.4850,   0.4850],
                 [  0.4850,   0.4850,   0.4850,  ...,   0.4850,   0.4850,   0.4850],
                 ...,
                 [104.3215, 134.5965, 153.4879,  ..., 164.9783, 159.0286, 169.7652],
                 [125.4779, 118.4292, 152.7473,  ..., 156.6215, 136.1776, 121.5470],
                 [120.8953, 124.3950, 177.6726,  ..., 177.1607, 165.4808, 141.5840]],
        
                [[  0.4560,   0.4560,   0.4560,  ...,   0.4560,   0.4560,   0.4560],
                 [  0.4560,   0.4560,   0.4560,  ...,   0.4560,   0.4560,   0.4560],
                 [  0.4560,   0.4560,   0.4560,  ...,   0.4560,   0.4560,   0.4560],
                 ...,
                 [102.3215, 132.5965, 150.4879,  ..., 160.9783, 161.0286, 171.7652],
                 [130.4106, 120.2483, 135.6516,  ..., 152.6215, 138.1776, 123.5470],
                 [126.2546, 126.3950, 164.7193,  ..., 173.1607, 167.4808, 143.5840]],
        
                [[  0.4060,   0.4060,   0.4060,  ...,   0.4060,   0.4060,   0.4060],
                 [  0.4060,   0.4060,   0.4060,  ...,   0.4060,   0.4060,   0.4060],
                 [  0.4060,   0.4060,   0.4060,  ...,   0.4060,   0.4060,   0.4060],
                 ...,
                 [ 77.3215, 107.5965, 131.4879,  ..., 148.9783, 147.0286, 157.7652],
                 [ 99.2839, 108.5495, 121.2899,  ..., 140.6215, 124.1776, 109.5470],
                 [ 94.8171, 115.2388, 148.7674,  ..., 161.1607, 153.4808, 129.5840]]])
                """
                # %%%%%%%%%%%%%%%%%%%%

                resize_crop_masked_imgs.append(resize_crop_masked_img)
                """
                resize_crop_masked_imgs.size : torch.Size([3, 224, 224])
                resize_crop_masked_imgs : [tensor([[[0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
                 [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
                 [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
                 ...,
                 [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
                 [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
                 [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850]],
        
                [[0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
                 [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
                 [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
                 ...,
                 [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
                 [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
                 [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560]],
        
                [[0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
                 [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
                 [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
                 ...,
                 [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
                 [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
                 [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060]]],
               device='cuda:0')]
                """
            resize_crop_masked_imgs = torch.stack(resize_crop_masked_imgs, dim=0)
            """
            resize_crop_masked_imgs.shape : torch.Size([마스크 개수, 3, 224, 224])
            resize_crop_masked_imgs : tensor([[[[0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              ...,
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850]],
    
             [[0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              ...,
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560]],
    
             [[0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              ...,
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060]]],
    
    
            [[[0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              ...,
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850]],
    
             [[0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              ...,
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560]],
    
             [[0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              ...,
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060]]],
    
    
            [[[0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              ...,
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850]],
    
             [[0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              ...,
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560]],
    
             [[0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              ...,
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060]]],
    
    
            ...,
    
    
            [[[0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              ...,
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850]],
    
             [[0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              ...,
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560]],
    
             [[0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              ...,
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060]]],
    
    
            [[[0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              ...,
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850]],
    
             [[0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              ...,
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560]],
    
             [[0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              ...,
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060]]],
    
    
            [[[0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              ...,
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850],
              [0.4850, 0.4850, 0.4850,  ..., 0.4850, 0.4850, 0.4850]],
    
             [[0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              ...,
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560],
              [0.4560, 0.4560, 0.4560,  ..., 0.4560, 0.4560, 0.4560]],
    
             [[0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              ...,
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060],
              [0.4060, 0.4060, 0.4060,  ..., 0.4060, 0.4060, 0.4060]]]],
           device='cuda:0')
            """

            # 4] Visual Feature Embedding = Global + Local 이미지 결합
            global_features = Model.feature_map_masking(resized_imgs, pred_masks) if mode == 'Res' else Model(resized_imgs, pred_masks, masking_type='token_masking', masking_block=9) # 모델 구조 ??? 왜 resized_imgs, pred_masks를 넣어주는거지 ???
            """
            mask_features.shape : torch.Size([마스크 개수, 512])
            mask_features : tensor([[ 0.3418,  0.0722, -0.2132,  ...,  0.4795, -0.1112, -0.1187],
            [ 0.3533,  0.0617, -0.2182,  ...,  0.5211, -0.1144, -0.0929],
            [ 0.3569,  0.0764, -0.2233,  ...,  0.5041, -0.1106, -0.1007],
            ...,
            [ 0.3574,  0.0635, -0.2233,  ...,  0.5246, -0.1170, -0.0881],
            [ 0.3529,  0.0612, -0.2173,  ...,  0.5216, -0.1136, -0.0938],
            [ 0.3545,  0.0594, -0.2211,  ...,  0.5243, -0.1186, -0.0908]],
           device='cuda:0', grad_fn=<MmBackward0>)
            """
            crop_features = Model.get_gloval_vector(resize_crop_masked_imgs) if mode == 'Res' else Model(resize_crop_masked_imgs, pred_masks=None, masking_type='crop') # 모델 구조 ???
            """
            crop_features.shape : torch.Size([마스크 개수, 512])
            crop_features : tensor([[-0.4071,  0.2320, -0.1380,  ...,  0.3936, -0.2222, -0.4506],
            [-0.0564,  0.1969, -0.2230,  ...,  0.8359, -0.0170,  0.1958],
            [ 0.2146,  0.1360, -0.1061,  ...,  0.2914, -0.0773, -0.5755],
            ...,
            [ 0.5168,  0.0441,  0.2411,  ...,  0.0992, -0.2189, -0.4323],
            [-0.1040,  0.2459, -0.0639,  ...,  0.8697,  0.0183,  0.1077],
            [ 0.2693,  0.2223,  0.0175,  ...,  0.1679, -0.1726, -0.1539]],
           device='cuda:0', grad_fn=<SliceBackward0>)
            """
            visual_feature = v * global_features + (1 - v) * crop_features
            """
            visual_feature.shape : torch.Size([마스크 개수, 512])
            visual_feature : tensor([[ 0.3043,  0.0802, -0.2094,  ...,  0.4752, -0.1168, -0.1353],
            [ 0.3328,  0.0684, -0.2184,  ...,  0.5369, -0.1095, -0.0785],
            [ 0.3498,  0.0794, -0.2174,  ...,  0.4935, -0.1089, -0.1245],
            ...,
            [ 0.3654,  0.0625, -0.2001,  ...,  0.5033, -0.1221, -0.1053],
            [ 0.3301,  0.0704, -0.2096,  ...,  0.5390, -0.1070, -0.0837],
            [ 0.3502,  0.0675, -0.2092,  ...,  0.5065, -0.1213, -0.0940]],
           device='cuda:0', grad_fn=<AddBackward0>)
            """

            # --------------------------------------------------------------
            # (4) Visualization Setting ●
            # --------------------------------------------------------------
            if args.plot_mask_result:
                # 실제 이미지 시각화 준비 ●
                fig_num = 0
                img_iter = 0
                scores = [] # score 출력을 위한 배열 ●

                real_image = cv2.imread(f"{args.base_dataset_path}/{image_filename}")
                real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
                plt.figure(f"{args.base_dataset_path}/{image_filename}", figsize=(8, 8))

            # %%%%%%%%%%%%%%%%%%%%
            # Free SOLO ●
            # %%%%%%%%%%%%%%%%%%%%
            # real_image = cv2.imread(f"{args.base_dataset_path}/{image[0]['file_name'][0]}")
            # real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
            # plt.figure(f"{args.base_dataset_path}/{image[0]['file_name'][0]}", figsize=(8, 8))
            # %%%%%%%%%%%%%%%%%%%%

            # --------------------------------------------------------------
            # (5) Text Feature Embedding
            # --------------------------------------------------------------
            for sentence in sentence_raw:
                # 1] Global Text 전처리
                """
                sentence : 'bowl behind the others can only see part'
                """
                sentence = sentence[0].lower()
                doc = nlp(sentence)
                """
                doc : bowl behind the others can only see part
                """

                sentence_for_spacy = []
                for i, token in enumerate(doc):
                    if token.text == ' ':
                        continue
                    sentence_for_spacy.append(token.text)

                # 2] Global Text 토큰화
                sentence_for_spacy = ' '.join(sentence_for_spacy)
                """
                'bowl behind the others can only see part'
                """
                sentence_token = clip.tokenize(sentence_for_spacy).to(device) # 모델 구조 ???
                """
                sentecne_token.shape : torch.Size([1, 77])
                sentence_token : tensor([[49406,  3814,  2403,   518,  3326,   753,  1033,   862,  1551, 49407,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                 0,     0,     0,     0,     0,     0,     0]], device='cuda:0',
               dtype=torch.int32)
                """

                # 3] Local Text 토큰화
                noun_phrase, not_phrase_index, head_noun = extract_noun_phrase(sentence_for_spacy, nlp, need_index=True) # 모델 구조 ???
                """
                noun_phrase : 'bowl'
                not_phrase_index : [1, 2, 3, 4, 5, 6, 7]
                head_noun : 'bowl'
                """
                noun_phrase_token = clip.tokenize(noun_phrase).to(device) # 모델 구조 ???
                """
                noun_phrase_token.shape : torch.Size([1, 512])
                noun_phrase_token : tensor([[-3.1288e-01,  1.9868e-01,  6.3410e-02,  5.9130e-03, -2.5689e-01,
                  1.4357e-01,  2.4914e-02, -1.1847e+00, -1.7176e-01,  2.7721e-01,
                 -4.2232e-01,  3.6275e-01, -8.2750e-02, -2.9226e-01,  1.2504e-01,
                  3.7595e-01,  3.9817e-01,  1.2646e-02, -3.3204e-02, -1.2081e-01,
                  8.2538e-02,  1.9247e-01,  3.2237e-01, -1.3626e-02, -3.2959e-01,
                  6.7287e-02,  8.4734e-03,  7.3723e-02,  8.7372e-02, -2.1483e-01,
                 -1.6705e-01,  3.2392e-02, -1.9547e-01, -4.8685e-01, -8.3071e-02,
                 -1.0352e-01,  1.4988e-01, -6.7656e-02,  1.1041e-01, -5.9326e-02,
                 -3.8006e-03,  1.9275e-01,  2.0609e-01, -3.2231e-02,  1.4242e-01,
                  8.9160e-02, -7.8913e-02,  3.5854e-01,  2.2395e-01,  2.2365e-01,
                 -6.9465e-02, -1.7802e-02,  1.9738e-01, -2.2856e-01, -5.3674e-02,
                 -4.5416e-01, -1.2657e-01,  2.5537e-01, -3.1432e-01,  1.4374e-01,
                  1.3779e-01, -5.3343e-01,  7.9334e-02, -2.7781e-01, -3.9952e-02,
                 -4.7663e-02,  4.7914e-02,  8.3076e-02,  1.2847e-01, -6.1525e-01,
                  8.6440e-02, -2.4401e-01, -2.4107e-01, -7.8068e-02, -2.5077e-01,
                  1.4077e-01,  2.7977e-01, -1.3687e-01,  2.4125e-01, -3.4648e-01,
                 -1.4675e-01, -1.4325e-01,  4.0331e-02,  1.9804e-01, -2.5194e-01,
                  2.2884e-01,  2.4437e-02, -1.4888e-01, -2.3829e-01,  1.3431e-01,
                 -2.2580e-01, -9.6217e-02, -1.5861e+00,  4.9879e-01, -3.3553e-01,
                 -1.5793e-01, -5.7047e-02,  2.2602e-01, -1.2495e-01, -2.6306e-01,
                  3.6522e-02,  1.5620e-02,  1.4671e-01, -1.3330e-01, -4.3593e-01,
                 -1.7359e-01,  1.3334e-01,  6.5376e-03, -5.3005e-01,  2.8766e-02,
                  6.8308e-02, -2.0637e-02,  3.1641e-01,  2.0615e-01, -3.5933e-01,
                 -3.9607e-02,  2.3878e-01,  3.8896e-01,  2.0010e-02, -4.3410e-01,
                 -1.8150e-01, -1.6673e-01, -3.1509e-01, -2.4146e-02, -1.8163e-01,
                  4.6200e-02, -2.0873e-01, -2.4345e-01, -3.0328e-02,  1.9167e-01,
                 -1.2216e-01,  5.5974e-02,  2.5269e-01,  5.9472e+00, -1.6705e-01,
                 -1.1355e-01, -1.8924e-01, -4.4419e-01, -1.2451e-01,  4.6184e-01,
                 -1.1506e-01, -6.9656e-02, -2.1386e-01,  3.1867e-01, -2.2326e-01,
                 -9.5977e-02,  1.1119e-01, -3.9094e-01, -2.7230e-01, -2.0347e-01,
                  2.9436e-01,  3.6822e-02, -1.5815e-02, -3.7122e-01, -1.7764e-01,
                 -6.0197e-02,  3.5643e-02,  9.7910e-02, -1.3553e-01,  1.2496e-01,
                 -2.5128e-01, -3.8761e-01, -1.1345e-01, -9.0660e-02, -1.9797e-01,
                 -8.8419e-02, -1.3574e-01,  1.8968e-02,  2.8267e-01, -4.5066e-02,
                 -1.2961e-01,  7.8770e-02,  1.4701e-01,  2.1145e-01, -3.2591e-01,
                  2.2682e-01, -5.0231e-01,  5.2956e-02, -5.0363e-02,  1.2424e-01,
                  2.1329e-01,  1.7319e-01, -8.2520e-02, -1.2082e-01, -1.4179e-02,
                 -2.3763e-01,  3.6687e-01,  1.7497e-01,  4.1452e-01,  1.4390e-02,
                  2.7573e-01, -1.4020e-02,  1.7483e-01, -1.6012e-01,  1.1034e-01,
                  2.1523e-01,  2.2537e-02,  7.1703e-03, -5.4270e-02, -2.0996e-01,
                 -1.3045e-01,  1.7665e-01,  8.1510e-02, -5.0535e-02,  2.5713e-02,
                 -1.9052e-01,  1.7143e-01, -1.2986e-01, -5.9927e-02, -1.0935e-01,
                  6.0536e-01,  2.6957e-01,  3.9631e-03, -2.3594e-02,  5.4444e-01,
                 -1.5775e-01,  9.0956e-02, -8.8719e-02, -9.4092e-02,  2.4461e-01,
                  1.2247e-01, -4.0090e-02, -5.7478e-02, -1.9884e-01,  3.9187e-01,
                 -8.4505e-02,  1.2765e-01,  1.5925e-01, -1.8334e-01, -4.1102e-01,
                  1.7031e-02,  4.5177e-02,  1.9683e-01, -4.8356e-01,  1.1266e-01,
                  6.8362e-03,  3.0162e-01,  1.6277e-01, -2.2266e-01, -1.2178e-01,
                  2.7420e-01,  9.2792e-02,  3.7061e-01,  1.8657e-01, -4.7485e-01,
                 -1.2420e-01, -9.8386e-02,  7.1552e-02,  3.8503e-02, -2.4881e-01,
                 -1.1710e-01,  2.2547e-01,  3.5309e-02,  2.0920e-02,  2.9762e-01,
                  1.5063e-01,  3.8431e-01,  1.4999e-01,  8.6780e-02,  1.6162e-01,
                  2.4825e-01,  2.4521e-01, -2.3367e-01, -7.3213e-02, -2.1498e-01,
                 -7.2813e-01, -1.9719e-02, -1.7236e-01, -2.2480e-01, -2.0907e-01,
                  2.2137e-02, -5.3531e-01,  4.0348e-01, -3.0235e-02, -3.0204e-01,
                 -1.4148e-01, -1.8870e-02,  8.9582e-02, -1.5384e-01, -3.6393e-02,
                  1.7171e-01,  8.4841e-02,  2.0560e-01,  1.7416e-01, -1.4746e-01,
                  2.3643e-01,  3.5369e-01,  2.3413e-01,  3.2286e-01,  3.1071e-02,
                  2.6131e-01, -1.5344e-01,  2.9477e-01,  4.2552e-01,  9.5060e-03,
                 -2.8329e-02, -1.5531e-01,  7.8966e-02,  4.3568e-03,  2.5510e-01,
                  5.9494e-02, -2.4635e-01,  1.1894e-01,  1.3863e-01, -7.3145e-02,
                  6.4702e-02,  3.2561e-01,  2.1508e-01, -1.0226e-01,  6.6201e-02,
                 -8.6703e-02,  2.6188e-01,  5.9399e+00, -2.4431e-01,  2.9629e-01,
                  2.8289e-01, -6.5784e-02,  1.8922e-02,  2.7636e-02,  1.3596e-01,
                  5.0913e-01,  3.9259e-01, -1.9727e-01,  1.3090e-01, -1.0204e-01,
                  2.7558e-01, -1.3004e-01,  1.1447e-02,  3.2990e-01, -2.2872e+00,
                 -1.4517e-01, -2.6903e-03, -5.1088e-02, -1.8329e-01,  4.1177e-02,
                  9.7603e-02,  2.8070e-01,  1.3112e-01, -3.0705e-01, -1.9555e-01,
                 -1.6648e-01, -1.1660e-01, -2.1350e-01, -2.1952e-01, -1.2589e-01,
                  1.9431e-01, -1.3657e-01,  1.3491e-01, -2.2437e-01,  3.7159e-02,
                  1.4929e-01, -1.2002e-01,  2.0036e-01,  1.4663e-01,  2.5033e-02,
                  6.3788e-02, -2.3389e-01, -6.4804e-02,  1.3218e-01,  1.0229e-01,
                 -3.6847e-01,  2.6623e-03,  1.2738e-01,  1.5848e-01,  1.9998e-01,
                 -3.0747e-01, -1.8236e-01,  2.3896e-01,  1.9095e-02,  1.7856e-01,
                 -3.4052e-01,  3.8021e-01,  3.5367e-01,  1.3776e-01, -1.8270e-01,
                 -3.1200e-01, -1.3092e-01,  2.4485e-02, -1.8635e-01, -1.3082e-01,
                 -2.4332e-01, -2.8921e-01, -3.0383e-01, -6.4549e-01,  2.3153e-01,
                 -3.6648e-01, -2.3368e-01, -4.1021e-03, -4.5348e-01, -1.4946e-01,
                 -4.0948e-01, -1.4884e-01,  9.7284e-02, -5.0248e-01, -1.6031e-01,
                  4.5336e-01,  1.5102e-01, -5.1815e-01,  1.4861e-01,  4.6900e-01,
                  2.1243e-02, -5.0289e-01,  2.7647e-01, -2.0399e-02, -3.5841e-01,
                  2.0661e-01, -3.3865e-02,  3.1167e-01, -7.9390e-02, -1.1525e-01,
                  4.2161e-02,  6.1937e-02, -3.9952e-01,  2.5797e-01, -1.6907e-01,
                  1.8231e-01, -1.7383e-01, -9.8369e-02, -1.1737e-01, -1.6136e-01,
                 -1.7716e-01, -1.1577e-01, -3.1500e-01, -3.3365e-01, -3.8970e-01,
                 -1.8288e-02, -3.7344e-01, -2.6820e-01,  1.2445e-01,  4.0507e-01,
                  2.4568e-01, -2.2345e-02,  1.2151e-01,  1.1989e-01,  6.7317e-02,
                 -2.0575e-01, -6.0335e-01,  5.3096e-02, -5.1035e-01,  5.3115e-01,
                 -4.0060e-01, -2.0013e-01, -2.4608e-01, -4.6972e-01,  2.3396e-01,
                  1.3228e-01, -3.8635e-01, -2.2659e-01, -1.8002e-01,  5.9459e-01,
                  3.5092e-01,  1.3363e-01, -3.3972e-01, -2.1212e-01, -1.3756e-01,
                  5.1313e-01,  1.6997e-01,  1.0936e-01,  2.5592e-01, -1.8614e-01,
                  7.4150e-02,  1.6125e-01,  1.4336e-01,  3.7730e-01,  2.7649e-01,
                  1.8804e-01,  1.2836e-01, -5.1168e-01, -1.7294e-01, -1.9409e-01,
                 -2.2978e-01, -2.0757e-01, -1.2454e-01, -2.6939e-01, -2.0909e-01,
                  1.2017e-01,  2.1435e-02,  8.0776e-02,  4.5147e-01,  4.2465e-01,
                 -8.3474e-02, -5.4937e-01,  1.6122e-01,  1.9284e-01,  3.4774e-02,
                  4.9528e-02, -7.8889e-03, -3.8514e-04,  4.8247e-01,  3.1967e-02,
                 -3.0196e-01,  4.9058e-02,  4.1116e-03,  6.4413e-01, -8.9781e-02,
                 -3.2805e-02,  5.0292e-01,  2.2077e-02, -3.4828e-01, -6.1042e-03,
                  3.8133e-02, -8.4256e-02,  2.8812e-01,  3.2640e-01, -5.7216e-02,
                 -3.6502e-01, -2.4135e-01, -1.8629e-01, -2.8301e-01, -2.9029e-01,
                 -8.4889e-02,  3.9403e-02]], device='cuda:0', grad_fn=<MmBackward0>)
                """

                # 4] Visual Feature Embedding = Global + Local Text 결합
                sentence_features = Model.get_text_feature(sentence_token) if mode == 'Res' else Model.model.encode_text(sentence_token) # 모델 구조 ???
                """
                sentence_features.shape : torch.Size([1, 512])
                sentence_features : tensor([[-2.9142e-01,  2.2443e-01,  9.3369e-02,  4.4079e-02, -3.0768e-01,
                  2.5683e-01, -2.1920e-02, -8.4727e-01,  1.7823e-01, -1.3852e-01,
                 -7.2470e-02,  5.2332e-01, -8.7746e-02, -4.4824e-01,  1.7622e-01,
                  3.6364e-02,  9.1988e-02, -5.3039e-02, -1.7804e-02,  2.4014e-01,
                  5.6870e-03,  3.0132e-01, -1.0937e-01,  6.1947e-02, -1.0131e-01,
                 -1.9841e-01, -8.2029e-03,  6.9492e-02,  3.0149e-01, -1.7617e-01,
                 -2.2489e-01,  1.6554e-01, -1.1646e-01, -2.1286e-01, -2.3961e-01,
                 -2.7453e-01,  5.8963e-02, -1.2962e-01,  5.9250e-01,  8.9433e-02,
                  1.8409e-01,  2.0138e-01,  2.9291e-01,  6.1309e-02, -7.5402e-02,
                 -1.5232e-01,  1.4825e-01,  2.7628e-01, -1.0555e-01, -7.9670e-02,
                 -3.1267e-01,  1.2304e-02, -2.1519e-01,  8.1166e-02, -2.6203e-01,
                 -2.5853e-01, -6.9524e-01, -6.2007e-02, -1.9145e-02,  1.9979e-01,
                  2.2954e-01, -2.2341e-01,  6.5801e-02, -1.6396e-01,  1.3766e-01,
                 -1.1543e-01, -3.3014e-02,  1.0655e-01,  1.3970e-01, -3.1312e-01,
                 -5.1395e-03, -2.1567e-01, -1.7758e-01, -3.7235e-01, -1.0070e-01,
                  2.9599e-01, -3.2611e-02,  1.0856e-01,  1.1643e-01, -3.3029e-01,
                  1.4241e-01,  1.5789e-01, -9.4035e-02,  1.9495e-01, -2.1103e-01,
                  1.7153e-01,  1.3929e-01,  1.3426e-02, -9.1485e-02,  1.1912e-01,
                 -1.8030e-02,  1.9896e-01, -1.4071e+00,  5.6097e-01, -5.2306e-01,
                 -3.3753e-02, -4.0628e-01, -7.7282e-02, -7.6944e-02,  2.3998e-02,
                 -8.5015e-02, -1.6432e-01,  1.0993e-01, -8.6850e-02, -4.6135e-01,
                 -4.1916e-03, -1.0773e-01, -1.4269e-01, -3.7869e-01,  1.5458e-01,
                 -2.4381e-01, -1.8311e-01,  2.3984e-01,  3.4596e-02, -2.1676e-01,
                  9.1979e-02,  7.6689e-02, -9.5344e-02, -4.2881e-01, -2.8806e-01,
                 -3.6834e-01, -5.6135e-01, -1.8599e-01, -1.5775e-02, -1.6376e-02,
                  2.2791e-01, -3.0112e-01, -1.5974e-02,  6.0281e-02,  2.1586e-01,
                 -1.8887e-01, -3.7623e-01,  2.3025e-01,  5.9560e+00, -1.0115e-01,
                 -2.4105e-01, -2.1104e-01, -5.4412e-01, -5.8422e-01,  5.2251e-01,
                 -2.0204e-01, -6.9033e-02, -9.8291e-02,  7.1463e-02, -1.9644e-01,
                 -1.7725e-01, -7.1285e-02, -8.0635e-01, -5.2972e-01, -2.1713e-03,
                 -1.0845e-01,  2.7923e-01,  4.1792e-01,  3.2399e-02, -1.3295e-01,
                 -3.3503e-01, -7.8842e-02, -2.2001e-01,  3.7012e-01, -1.5281e-01,
                 -7.6925e-02, -1.6145e-01, -1.3396e-02, -4.3757e-02,  1.5141e-01,
                  7.3999e-02, -2.7453e-02,  1.4384e-01,  2.0729e-01, -2.1015e-01,
                 -4.7958e-02,  9.3142e-02,  3.9641e-01,  1.3407e-01, -3.2103e-01,
                  1.3330e-01, -4.7816e-01, -1.3882e-02, -1.7103e-01,  4.1709e-01,
                 -9.9381e-02, -8.8482e-02, -3.0031e-01, -6.5812e-02, -2.3166e-02,
                 -5.7109e-02,  1.9086e-01, -1.0484e-01,  5.1555e-01, -1.6165e-02,
                  1.6427e-01,  1.6495e-01,  2.4768e-01, -1.2583e-01,  3.5419e-01,
                  4.9166e-01, -1.4919e-01,  1.5799e-01, -3.1561e-02, -1.7302e-01,
                 -5.2588e-03,  3.1557e-01,  1.2805e-01,  9.5292e-02, -2.4941e-01,
                 -5.8580e-02, -1.1055e-02,  5.3883e-02, -2.2561e-01, -1.6644e-01,
                  2.0660e-01,  2.3044e-01,  4.3719e-02,  9.1895e-02,  3.7058e-01,
                 -1.3457e-01, -1.2007e-01, -4.7984e-01,  1.0801e-01,  2.5206e-01,
                  1.5559e-01, -3.4058e-02, -2.7214e-02,  2.5134e-01, -1.0629e-01,
                 -2.8345e-01, -8.4512e-02,  3.0506e-02, -4.0436e-03, -2.3406e-01,
                 -2.3454e-01,  2.2906e-01,  4.1951e-02, -1.9460e-01,  5.4900e-02,
                  5.3750e-02,  3.4535e-01, -4.9017e-02, -3.3586e-01,  8.7726e-02,
                  3.1723e-01,  1.1314e-01,  8.0431e-02,  1.1334e-01, -5.9689e-01,
                  5.3810e-01, -1.9796e-02, -4.4747e-02, -1.3938e-01,  9.8059e-02,
                  3.5418e-02,  4.4159e-02,  2.3737e-01, -3.4982e-01,  8.7847e-02,
                  1.5216e-02, -2.3137e-01,  1.0284e-01,  1.6518e-01,  2.4273e-01,
                  1.4820e-02,  1.4290e-01, -1.9911e-01, -2.0179e-02, -8.2811e-02,
                 -6.5647e-01,  2.4723e-01,  2.1711e-01, -3.3967e-01,  2.9298e-02,
                  1.8811e-01, -4.1490e-01,  1.4696e-01, -2.1423e-01, -2.6462e-01,
                  1.2744e-01, -3.9182e-02,  2.5321e-01,  2.7453e-02, -6.8268e-02,
                 -1.6430e-01,  2.5548e-01,  2.1450e-01,  2.8094e-01, -7.4572e-02,
                  1.2294e-01,  1.8421e-01,  2.0405e-01,  3.1564e-01,  2.7504e-01,
                 -1.5038e-01, -1.1428e-01,  3.3977e-01,  2.0209e-01,  1.5012e-01,
                 -3.4406e-02, -2.9812e-01,  7.0028e-02, -2.1794e-01,  8.2461e-02,
                 -1.4313e-01, -9.1752e-02,  7.4258e-02,  7.6239e-02, -4.8832e-03,
                  8.1165e-02,  2.8160e-01,  5.8857e-02, -1.7141e-01, -2.7394e-01,
                 -9.8507e-02,  5.6074e-01,  5.9500e+00, -2.4412e-01,  5.7694e-01,
                  4.0686e-01, -9.2283e-02,  3.7015e-01, -2.1331e-01,  6.6175e-01,
                 -6.9490e-04,  3.6321e-01, -2.1200e-01,  6.7201e-02, -1.6645e-01,
                  5.4205e-01,  2.1893e-01, -6.3558e-02,  1.6420e-01, -2.3079e+00,
                 -3.3211e-02, -3.1879e-01, -8.3841e-03, -2.6831e-01,  1.3922e-02,
                 -2.0043e-01,  6.0339e-02,  2.2847e-01, -2.7800e-01,  1.7961e-01,
                 -1.8427e-01,  1.9433e-01, -2.4398e-01, -4.0899e-01, -1.1082e-01,
                  3.3354e-01,  9.6186e-03,  1.6392e-01, -4.5035e-01,  2.4354e-01,
                  3.6725e-01, -1.2428e-01, -4.7281e-02,  2.8713e-01, -1.3585e-01,
                 -1.6992e-01, -1.9461e-01, -6.3078e-02,  1.6810e-01,  4.3954e-02,
                 -4.1516e-01, -6.5464e-02,  1.4753e-01,  1.8815e-02, -1.8950e-01,
                  4.4111e-01, -2.8524e-01,  2.7609e-01,  5.9427e-02,  2.2090e-01,
                 -3.0422e-01,  3.2670e-01,  1.5264e-01,  1.2111e-02,  1.5186e-01,
                 -4.4561e-01,  6.5381e-02, -4.0110e-02,  7.2288e-02, -1.5973e-01,
                  1.4955e-01, -1.2550e-01, -9.2839e-02, -5.3443e-01,  2.1378e-01,
                  2.7992e-01,  1.2326e-01,  1.6482e-01, -3.7513e-01, -1.7574e-01,
                 -8.1564e-01, -1.2543e-01,  4.7621e-02, -2.4425e-01,  1.9490e-02,
                  3.4347e-01, -2.2942e-01, -3.4757e-01,  1.6796e-01,  7.0423e-01,
                  3.1299e-01, -1.4233e-01,  4.8142e-01, -7.2903e-02, -2.7156e-01,
                  2.3589e-01, -2.8693e-01,  3.0750e-01, -2.4924e-01,  2.2697e-01,
                  1.8569e-01, -5.7867e-02, -3.2315e-01, -8.0274e-02, -1.5714e-01,
                 -4.0000e-01, -1.5662e-01, -7.8469e-02,  6.0323e-02, -2.3535e-01,
                 -7.7039e-02, -5.7856e-02,  1.0100e-01, -1.7041e-01, -7.7837e-02,
                 -4.4627e-01, -1.9666e-01, -6.7220e-02,  2.2015e-01,  3.6701e-01,
                  1.6861e-01,  2.5266e-01,  2.9867e-02,  7.3276e-02, -5.0574e-02,
                 -6.3717e-02, -4.9752e-01, -1.7732e-01, -1.8444e-01,  2.8131e-01,
                 -4.3745e-01, -3.3886e-01, -1.6700e-01, -2.9536e-01, -7.8970e-02,
                 -8.8233e-02, -2.0447e-01, -2.7975e-02, -1.2767e-01,  1.8495e-01,
                  1.8559e-01, -4.7652e-02, -2.7816e-01, -4.4630e-01, -1.0082e-01,
                 -4.1604e-02, -8.4402e-02, -8.3976e-02, -1.7541e-01, -3.0040e-01,
                 -1.0465e-01, -4.5345e-02,  1.4118e-01,  1.0087e-01,  4.3190e-01,
                 -4.4737e-02, -2.0366e-01, -3.4147e-01, -1.6879e-02,  4.7507e-02,
                 -2.3256e-01,  5.1867e-02,  2.9508e-01, -1.8425e-01, -1.8528e-01,
                  1.7646e-01, -1.3902e-01, -7.3132e-02,  3.0933e-01,  3.3163e-01,
                  1.2413e-02, -8.3822e-01,  2.0002e-01,  2.2591e-01,  2.9225e-02,
                 -1.3518e-01, -8.3727e-02,  6.9648e-02,  2.0951e-01, -6.3939e-02,
                 -2.4219e-01, -2.4148e-01, -1.0175e-01,  9.3865e-01,  9.6632e-02,
                 -1.7744e-01,  3.1003e-01,  7.8363e-02, -2.7525e-01,  3.8351e-02,
                  8.4488e-02, -6.7476e-02,  8.6994e-02,  1.0528e-01,  1.8795e-01,
                 -5.7046e-02, -1.3731e-01,  5.2807e-02, -2.0157e-01, -6.5436e-01,
                 -2.0181e-03, -9.9033e-02]], device='cuda:0', grad_fn=<MmBackward0>)
                """
                noun_phrase_features = Model.get_text_feature(noun_phrase_token) if mode == 'Res' else Model.model.encode_text(noun_phrase_token) # 모델 구조 ???
                """
                noun_phrase_features.shape : torch.Size([1, 512])
                noun_phrase_features : tensor([[-3.1288e-01,  1.9868e-01,  6.3410e-02,  5.9130e-03, -2.5689e-01,
                  1.4357e-01,  2.4914e-02, -1.1847e+00, -1.7176e-01,  2.7721e-01,
                 -4.2232e-01,  3.6275e-01, -8.2750e-02, -2.9226e-01,  1.2504e-01,
                  3.7595e-01,  3.9817e-01,  1.2646e-02, -3.3204e-02, -1.2081e-01,
                  8.2538e-02,  1.9247e-01,  3.2237e-01, -1.3626e-02, -3.2959e-01,
                  6.7287e-02,  8.4734e-03,  7.3723e-02,  8.7372e-02, -2.1483e-01,
                 -1.6705e-01,  3.2392e-02, -1.9547e-01, -4.8685e-01, -8.3071e-02,
                 -1.0352e-01,  1.4988e-01, -6.7656e-02,  1.1041e-01, -5.9326e-02,
                 -3.8006e-03,  1.9275e-01,  2.0609e-01, -3.2231e-02,  1.4242e-01,
                  8.9160e-02, -7.8913e-02,  3.5854e-01,  2.2395e-01,  2.2365e-01,
                 -6.9465e-02, -1.7802e-02,  1.9738e-01, -2.2856e-01, -5.3674e-02,
                 -4.5416e-01, -1.2657e-01,  2.5537e-01, -3.1432e-01,  1.4374e-01,
                  1.3779e-01, -5.3343e-01,  7.9334e-02, -2.7781e-01, -3.9952e-02,
                 -4.7663e-02,  4.7914e-02,  8.3076e-02,  1.2847e-01, -6.1525e-01,
                  8.6440e-02, -2.4401e-01, -2.4107e-01, -7.8068e-02, -2.5077e-01,
                  1.4077e-01,  2.7977e-01, -1.3687e-01,  2.4125e-01, -3.4648e-01,
                 -1.4675e-01, -1.4325e-01,  4.0331e-02,  1.9804e-01, -2.5194e-01,
                  2.2884e-01,  2.4437e-02, -1.4888e-01, -2.3829e-01,  1.3431e-01,
                 -2.2580e-01, -9.6217e-02, -1.5861e+00,  4.9879e-01, -3.3553e-01,
                 -1.5793e-01, -5.7047e-02,  2.2602e-01, -1.2495e-01, -2.6306e-01,
                  3.6522e-02,  1.5620e-02,  1.4671e-01, -1.3330e-01, -4.3593e-01,
                 -1.7359e-01,  1.3334e-01,  6.5376e-03, -5.3005e-01,  2.8766e-02,
                  6.8308e-02, -2.0637e-02,  3.1641e-01,  2.0615e-01, -3.5933e-01,
                 -3.9607e-02,  2.3878e-01,  3.8896e-01,  2.0010e-02, -4.3410e-01,
                 -1.8150e-01, -1.6673e-01, -3.1509e-01, -2.4146e-02, -1.8163e-01,
                  4.6200e-02, -2.0873e-01, -2.4345e-01, -3.0328e-02,  1.9167e-01,
                 -1.2216e-01,  5.5974e-02,  2.5269e-01,  5.9472e+00, -1.6705e-01,
                 -1.1355e-01, -1.8924e-01, -4.4419e-01, -1.2451e-01,  4.6184e-01,
                 -1.1506e-01, -6.9656e-02, -2.1386e-01,  3.1867e-01, -2.2326e-01,
                 -9.5977e-02,  1.1119e-01, -3.9094e-01, -2.7230e-01, -2.0347e-01,
                  2.9436e-01,  3.6822e-02, -1.5815e-02, -3.7122e-01, -1.7764e-01,
                 -6.0197e-02,  3.5643e-02,  9.7910e-02, -1.3553e-01,  1.2496e-01,
                 -2.5128e-01, -3.8761e-01, -1.1345e-01, -9.0660e-02, -1.9797e-01,
                 -8.8419e-02, -1.3574e-01,  1.8968e-02,  2.8267e-01, -4.5066e-02,
                 -1.2961e-01,  7.8770e-02,  1.4701e-01,  2.1145e-01, -3.2591e-01,
                  2.2682e-01, -5.0231e-01,  5.2956e-02, -5.0363e-02,  1.2424e-01,
                  2.1329e-01,  1.7319e-01, -8.2520e-02, -1.2082e-01, -1.4179e-02,
                 -2.3763e-01,  3.6687e-01,  1.7497e-01,  4.1452e-01,  1.4390e-02,
                  2.7573e-01, -1.4020e-02,  1.7483e-01, -1.6012e-01,  1.1034e-01,
                  2.1523e-01,  2.2537e-02,  7.1703e-03, -5.4270e-02, -2.0996e-01,
                 -1.3045e-01,  1.7665e-01,  8.1510e-02, -5.0535e-02,  2.5713e-02,
                 -1.9052e-01,  1.7143e-01, -1.2986e-01, -5.9927e-02, -1.0935e-01,
                  6.0536e-01,  2.6957e-01,  3.9631e-03, -2.3594e-02,  5.4444e-01,
                 -1.5775e-01,  9.0956e-02, -8.8719e-02, -9.4092e-02,  2.4461e-01,
                  1.2247e-01, -4.0090e-02, -5.7478e-02, -1.9884e-01,  3.9187e-01,
                 -8.4505e-02,  1.2765e-01,  1.5925e-01, -1.8334e-01, -4.1102e-01,
                  1.7031e-02,  4.5177e-02,  1.9683e-01, -4.8356e-01,  1.1266e-01,
                  6.8362e-03,  3.0162e-01,  1.6277e-01, -2.2266e-01, -1.2178e-01,
                  2.7420e-01,  9.2792e-02,  3.7061e-01,  1.8657e-01, -4.7485e-01,
                 -1.2420e-01, -9.8386e-02,  7.1552e-02,  3.8503e-02, -2.4881e-01,
                 -1.1710e-01,  2.2547e-01,  3.5309e-02,  2.0920e-02,  2.9762e-01,
                  1.5063e-01,  3.8431e-01,  1.4999e-01,  8.6780e-02,  1.6162e-01,
                  2.4825e-01,  2.4521e-01, -2.3367e-01, -7.3213e-02, -2.1498e-01,
                 -7.2813e-01, -1.9719e-02, -1.7236e-01, -2.2480e-01, -2.0907e-01,
                  2.2137e-02, -5.3531e-01,  4.0348e-01, -3.0235e-02, -3.0204e-01,
                 -1.4148e-01, -1.8870e-02,  8.9582e-02, -1.5384e-01, -3.6393e-02,
                  1.7171e-01,  8.4841e-02,  2.0560e-01,  1.7416e-01, -1.4746e-01,
                  2.3643e-01,  3.5369e-01,  2.3413e-01,  3.2286e-01,  3.1071e-02,
                  2.6131e-01, -1.5344e-01,  2.9477e-01,  4.2552e-01,  9.5060e-03,
                 -2.8329e-02, -1.5531e-01,  7.8966e-02,  4.3568e-03,  2.5510e-01,
                  5.9494e-02, -2.4635e-01,  1.1894e-01,  1.3863e-01, -7.3145e-02,
                  6.4702e-02,  3.2561e-01,  2.1508e-01, -1.0226e-01,  6.6201e-02,
                 -8.6703e-02,  2.6188e-01,  5.9399e+00, -2.4431e-01,  2.9629e-01,
                  2.8289e-01, -6.5784e-02,  1.8922e-02,  2.7636e-02,  1.3596e-01,
                  5.0913e-01,  3.9259e-01, -1.9727e-01,  1.3090e-01, -1.0204e-01,
                  2.7558e-01, -1.3004e-01,  1.1447e-02,  3.2990e-01, -2.2872e+00,
                 -1.4517e-01, -2.6903e-03, -5.1088e-02, -1.8329e-01,  4.1177e-02,
                  9.7603e-02,  2.8070e-01,  1.3112e-01, -3.0705e-01, -1.9555e-01,
                 -1.6648e-01, -1.1660e-01, -2.1350e-01, -2.1952e-01, -1.2589e-01,
                  1.9431e-01, -1.3657e-01,  1.3491e-01, -2.2437e-01,  3.7159e-02,
                  1.4929e-01, -1.2002e-01,  2.0036e-01,  1.4663e-01,  2.5033e-02,
                  6.3788e-02, -2.3389e-01, -6.4804e-02,  1.3218e-01,  1.0229e-01,
                 -3.6847e-01,  2.6623e-03,  1.2738e-01,  1.5848e-01,  1.9998e-01,
                 -3.0747e-01, -1.8236e-01,  2.3896e-01,  1.9095e-02,  1.7856e-01,
                 -3.4052e-01,  3.8021e-01,  3.5367e-01,  1.3776e-01, -1.8270e-01,
                 -3.1200e-01, -1.3092e-01,  2.4485e-02, -1.8635e-01, -1.3082e-01,
                 -2.4332e-01, -2.8921e-01, -3.0383e-01, -6.4549e-01,  2.3153e-01,
                 -3.6648e-01, -2.3368e-01, -4.1021e-03, -4.5348e-01, -1.4946e-01,
                 -4.0948e-01, -1.4884e-01,  9.7284e-02, -5.0248e-01, -1.6031e-01,
                  4.5336e-01,  1.5102e-01, -5.1815e-01,  1.4861e-01,  4.6900e-01,
                  2.1243e-02, -5.0289e-01,  2.7647e-01, -2.0399e-02, -3.5841e-01,
                  2.0661e-01, -3.3865e-02,  3.1167e-01, -7.9390e-02, -1.1525e-01,
                  4.2161e-02,  6.1937e-02, -3.9952e-01,  2.5797e-01, -1.6907e-01,
                  1.8231e-01, -1.7383e-01, -9.8369e-02, -1.1737e-01, -1.6136e-01,
                 -1.7716e-01, -1.1577e-01, -3.1500e-01, -3.3365e-01, -3.8970e-01,
                 -1.8288e-02, -3.7344e-01, -2.6820e-01,  1.2445e-01,  4.0507e-01,
                  2.4568e-01, -2.2345e-02,  1.2151e-01,  1.1989e-01,  6.7317e-02,
                 -2.0575e-01, -6.0335e-01,  5.3096e-02, -5.1035e-01,  5.3115e-01,
                 -4.0060e-01, -2.0013e-01, -2.4608e-01, -4.6972e-01,  2.3396e-01,
                  1.3228e-01, -3.8635e-01, -2.2659e-01, -1.8002e-01,  5.9459e-01,
                  3.5092e-01,  1.3363e-01, -3.3972e-01, -2.1212e-01, -1.3756e-01,
                  5.1313e-01,  1.6997e-01,  1.0936e-01,  2.5592e-01, -1.8614e-01,
                  7.4150e-02,  1.6125e-01,  1.4336e-01,  3.7730e-01,  2.7649e-01,
                  1.8804e-01,  1.2836e-01, -5.1168e-01, -1.7294e-01, -1.9409e-01,
                 -2.2978e-01, -2.0757e-01, -1.2454e-01, -2.6939e-01, -2.0909e-01,
                  1.2017e-01,  2.1435e-02,  8.0776e-02,  4.5147e-01,  4.2465e-01,
                 -8.3474e-02, -5.4937e-01,  1.6122e-01,  1.9284e-01,  3.4774e-02,
                  4.9528e-02, -7.8889e-03, -3.8514e-04,  4.8247e-01,  3.1967e-02,
                 -3.0196e-01,  4.9058e-02,  4.1116e-03,  6.4413e-01, -8.9781e-02,
                 -3.2805e-02,  5.0292e-01,  2.2077e-02, -3.4828e-01, -6.1042e-03,
                  3.8133e-02, -8.4256e-02,  2.8812e-01,  3.2640e-01, -5.7216e-02,
                 -3.6502e-01, -2.4135e-01, -1.8629e-01, -2.8301e-01, -2.9029e-01,
                 -8.4889e-02,  3.9403e-02]], device='cuda:0', grad_fn=<MmBackward0>)
                """
                text_feature = r * sentence_features + (1-r) * noun_phrase_features
                """
                text_ensemble.shape : torch.Size([1, 512])
                text_ensemble : tensor([[-3.0215e-01,  2.1155e-01,  7.8389e-02,  2.4996e-02, -2.8228e-01,
                  2.0020e-01,  1.4969e-03, -1.0160e+00,  3.2318e-03,  6.9347e-02,
                 -2.4740e-01,  4.4304e-01, -8.5248e-02, -3.7025e-01,  1.5063e-01,
                  2.0616e-01,  2.4508e-01, -2.0197e-02, -2.5504e-02,  5.9666e-02,
                  4.4113e-02,  2.4690e-01,  1.0650e-01,  2.4161e-02, -2.1545e-01,
                 -6.5559e-02,  1.3524e-04,  7.1607e-02,  1.9443e-01, -1.9550e-01,
                 -1.9597e-01,  9.8967e-02, -1.5597e-01, -3.4986e-01, -1.6134e-01,
                 -1.8902e-01,  1.0442e-01, -9.8638e-02,  3.5145e-01,  1.5054e-02,
                  9.0144e-02,  1.9707e-01,  2.4950e-01,  1.4539e-02,  3.3510e-02,
                 -3.1580e-02,  3.4670e-02,  3.1741e-01,  5.9200e-02,  7.1988e-02,
                 -1.9107e-01, -2.7491e-03, -8.9083e-03, -7.3695e-02, -1.5785e-01,
                 -3.5635e-01, -4.1091e-01,  9.6681e-02, -1.6673e-01,  1.7176e-01,
                  1.8367e-01, -3.7842e-01,  7.2567e-02, -2.2089e-01,  4.8854e-02,
                 -8.1544e-02,  7.4502e-03,  9.4811e-02,  1.3408e-01, -4.6419e-01,
                  4.0650e-02, -2.2984e-01, -2.0933e-01, -2.2521e-01, -1.7573e-01,
                  2.1838e-01,  1.2358e-01, -1.4153e-02,  1.7884e-01, -3.3838e-01,
                 -2.1658e-03,  7.3186e-03, -2.6852e-02,  1.9650e-01, -2.3148e-01,
                  2.0019e-01,  8.1863e-02, -6.7725e-02, -1.6489e-01,  1.2672e-01,
                 -1.2191e-01,  5.1374e-02, -1.4966e+00,  5.2988e-01, -4.2929e-01,
                 -9.5842e-02, -2.3166e-01,  7.4371e-02, -1.0095e-01, -1.1953e-01,
                 -2.4246e-02, -7.4351e-02,  1.2832e-01, -1.1008e-01, -4.4864e-01,
                 -8.8891e-02,  1.2808e-02, -6.8078e-02, -4.5437e-01,  9.1675e-02,
                 -8.7753e-02, -1.0187e-01,  2.7812e-01,  1.2037e-01, -2.8804e-01,
                  2.6186e-02,  1.5773e-01,  1.4681e-01, -2.0440e-01, -3.6108e-01,
                 -2.7492e-01, -3.6404e-01, -2.5054e-01, -1.9961e-02, -9.9004e-02,
                  1.3706e-01, -2.5492e-01, -1.2971e-01,  1.4976e-02,  2.0377e-01,
                 -1.5552e-01, -1.6013e-01,  2.4147e-01,  5.9516e+00, -1.3410e-01,
                 -1.7730e-01, -2.0014e-01, -4.9416e-01, -3.5437e-01,  4.9218e-01,
                 -1.5855e-01, -6.9345e-02, -1.5607e-01,  1.9506e-01, -2.0985e-01,
                 -1.3661e-01,  1.9955e-02, -5.9865e-01, -4.0101e-01, -1.0282e-01,
                  9.2956e-02,  1.5802e-01,  2.0105e-01, -1.6941e-01, -1.5530e-01,
                 -1.9761e-01, -2.1599e-02, -6.1053e-02,  1.1730e-01, -1.3926e-02,
                 -1.6410e-01, -2.7453e-01, -6.3422e-02, -6.7208e-02, -2.3282e-02,
                 -7.2098e-03, -8.1595e-02,  8.1403e-02,  2.4498e-01, -1.2761e-01,
                 -8.8784e-02,  8.5956e-02,  2.7171e-01,  1.7276e-01, -3.2347e-01,
                  1.8006e-01, -4.9023e-01,  1.9537e-02, -1.1070e-01,  2.7066e-01,
                  5.6957e-02,  4.2355e-02, -1.9141e-01, -9.3316e-02, -1.8673e-02,
                 -1.4737e-01,  2.7886e-01,  3.5065e-02,  4.6504e-01, -8.8719e-04,
                  2.2000e-01,  7.5463e-02,  2.1126e-01, -1.4298e-01,  2.3227e-01,
                  3.5344e-01, -6.3327e-02,  8.2578e-02, -4.2915e-02, -1.9149e-01,
                 -6.7855e-02,  2.4611e-01,  1.0478e-01,  2.2379e-02, -1.1185e-01,
                 -1.2455e-01,  8.0185e-02, -3.7987e-02, -1.4277e-01, -1.3790e-01,
                  4.0598e-01,  2.5001e-01,  2.3841e-02,  3.4151e-02,  4.5751e-01,
                 -1.4616e-01, -1.4556e-02, -2.8428e-01,  6.9606e-03,  2.4834e-01,
                  1.3903e-01, -3.7074e-02, -4.2346e-02,  2.6251e-02,  1.4279e-01,
                 -1.8398e-01,  2.1569e-02,  9.4877e-02, -9.3692e-02, -3.2254e-01,
                 -1.0875e-01,  1.3712e-01,  1.1939e-01, -3.3908e-01,  8.3781e-02,
                  3.0293e-02,  3.2349e-01,  5.6877e-02, -2.7926e-01, -1.7026e-02,
                  2.9572e-01,  1.0297e-01,  2.2552e-01,  1.4996e-01, -5.3587e-01,
                  2.0695e-01, -5.9091e-02,  1.3402e-02, -5.0438e-02, -7.5376e-02,
                 -4.0840e-02,  1.3481e-01,  1.3634e-01, -1.6445e-01,  1.9273e-01,
                  8.2924e-02,  7.6467e-02,  1.2642e-01,  1.2598e-01,  2.0217e-01,
                  1.3153e-01,  1.9405e-01, -2.1639e-01, -4.6696e-02, -1.4890e-01,
                 -6.9230e-01,  1.1376e-01,  2.2374e-02, -2.8223e-01, -8.9888e-02,
                  1.0512e-01, -4.7511e-01,  2.7522e-01, -1.2223e-01, -2.8333e-01,
                 -7.0171e-03, -2.9026e-02,  1.7140e-01, -6.3195e-02, -5.2330e-02,
                  3.7013e-03,  1.7016e-01,  2.1005e-01,  2.2755e-01, -1.1102e-01,
                  1.7968e-01,  2.6895e-01,  2.1909e-01,  3.1925e-01,  1.5306e-01,
                  5.5464e-02, -1.3386e-01,  3.1727e-01,  3.1380e-01,  7.9811e-02,
                 -3.1368e-02, -2.2671e-01,  7.4497e-02, -1.0679e-01,  1.6878e-01,
                 -4.1819e-02, -1.6905e-01,  9.6599e-02,  1.0744e-01, -3.9014e-02,
                  7.2933e-02,  3.0360e-01,  1.3697e-01, -1.3683e-01, -1.0387e-01,
                 -9.2605e-02,  4.1131e-01,  5.9450e+00, -2.4422e-01,  4.3661e-01,
                  3.4488e-01, -7.9033e-02,  1.9453e-01, -9.2836e-02,  3.9885e-01,
                  2.5422e-01,  3.7790e-01, -2.0463e-01,  9.9051e-02, -1.3424e-01,
                  4.0881e-01,  4.4445e-02, -2.6056e-02,  2.4705e-01, -2.2975e+00,
                 -8.9188e-02, -1.6074e-01, -2.9736e-02, -2.2580e-01,  2.7549e-02,
                 -5.1415e-02,  1.7052e-01,  1.7979e-01, -2.9252e-01, -7.9689e-03,
                 -1.7537e-01,  3.8869e-02, -2.2874e-01, -3.1426e-01, -1.1836e-01,
                  2.6392e-01, -6.3478e-02,  1.4942e-01, -3.3736e-01,  1.4035e-01,
                  2.5827e-01, -1.2215e-01,  7.6540e-02,  2.1688e-01, -5.5408e-02,
                 -5.3064e-02, -2.1425e-01, -6.3941e-02,  1.5014e-01,  7.3123e-02,
                 -3.9181e-01, -3.1401e-02,  1.3745e-01,  8.8649e-02,  5.2358e-03,
                  6.6824e-02, -2.3380e-01,  2.5752e-01,  3.9261e-02,  1.9973e-01,
                 -3.2237e-01,  3.5345e-01,  2.5316e-01,  7.4935e-02, -1.5420e-02,
                 -3.7880e-01, -3.2768e-02, -7.8125e-03, -5.7034e-02, -1.4528e-01,
                 -4.6889e-02, -2.0736e-01, -1.9833e-01, -5.8996e-01,  2.2266e-01,
                 -4.3280e-02, -5.5206e-02,  8.0358e-02, -4.1430e-01, -1.6260e-01,
                 -6.1256e-01, -1.3713e-01,  7.2453e-02, -3.7336e-01, -7.0411e-02,
                  3.9842e-01, -3.9197e-02, -4.3286e-01,  1.5828e-01,  5.8661e-01,
                  1.6712e-01, -3.2261e-01,  3.7894e-01, -4.6651e-02, -3.1498e-01,
                  2.2125e-01, -1.6040e-01,  3.0958e-01, -1.6431e-01,  5.5858e-02,
                  1.1393e-01,  2.0350e-03, -3.6134e-01,  8.8847e-02, -1.6310e-01,
                 -1.0885e-01, -1.6522e-01, -8.8419e-02, -2.8521e-02, -1.9836e-01,
                 -1.2710e-01, -8.6815e-02, -1.0700e-01, -2.5203e-01, -2.3377e-01,
                 -2.3228e-01, -2.8505e-01, -1.6771e-01,  1.7230e-01,  3.8604e-01,
                  2.0714e-01,  1.1516e-01,  7.5689e-02,  9.6584e-02,  8.3716e-03,
                 -1.3473e-01, -5.5043e-01, -6.2111e-02, -3.4740e-01,  4.0623e-01,
                 -4.1902e-01, -2.6949e-01, -2.0654e-01, -3.8254e-01,  7.7495e-02,
                  2.2021e-02, -2.9541e-01, -1.2728e-01, -1.5385e-01,  3.8977e-01,
                  2.6826e-01,  4.2988e-02, -3.0894e-01, -3.2921e-01, -1.1919e-01,
                  2.3576e-01,  4.2784e-02,  1.2691e-02,  4.0255e-02, -2.4327e-01,
                 -1.5251e-02,  5.7954e-02,  1.4227e-01,  2.3908e-01,  3.5419e-01,
                  7.1653e-02, -3.7651e-02, -4.2658e-01, -9.4908e-02, -7.3294e-02,
                 -2.3117e-01, -7.7854e-02,  8.5269e-02, -2.2682e-01, -1.9718e-01,
                  1.4832e-01, -5.8794e-02,  3.8217e-03,  3.8040e-01,  3.7814e-01,
                 -3.5530e-02, -6.9379e-01,  1.8062e-01,  2.0937e-01,  3.1999e-02,
                 -4.2826e-02, -4.5808e-02,  3.4631e-02,  3.4599e-01, -1.5986e-02,
                 -2.7207e-01, -9.6212e-02, -4.8819e-02,  7.9139e-01,  3.4254e-03,
                 -1.0512e-01,  4.0648e-01,  5.0220e-02, -3.1177e-01,  1.6123e-02,
                  6.1310e-02, -7.5866e-02,  1.8756e-01,  2.1584e-01,  6.5365e-02,
                 -2.1103e-01, -1.8933e-01, -6.6741e-02, -2.4229e-01, -4.7233e-01,
                 -4.3454e-02, -2.9815e-02]], device='cuda:0', grad_fn=<AddBackward0>)
                """

                # --------------------------------------------------------------
                # (6) Visual-Text Similarity 계산
                # --------------------------------------------------------------
                score =  Model.calculate_similarity_score(visual_feature, text_feature) if mode == 'Res' else Model.calculate_score(visual_feature, text_feature) # 모델 구조 ???
                """
                score.shape : torch.Size([마크스 개수, 1])
                score : tensor([[22.4040],
                [21.9637],
                [22.5527],
                [22.0154],
                [22.4180],
                [22.4116],
                [22.4031],
                [21.9437],
                [22.4650],
                [22.4457],
                [22.4893],
                [22.4945],
                [22.2598],
                [22.4233],
                [22.3713],
                [22.3032],
                [22.1083],
                [22.3312],
                [22.4050],
                [22.5024],
                [22.3216],
                [22.3790],
                [22.4147],
                [22.5872],
                [22.3360],
                [22.0431],
                [22.2664],
                [22.0926],
                [22.3382],
                [22.1375],
                [22.4245],
                [22.0147],
                [21.9917],
                [22.0193],
                [21.9351],
                [21.9293],
                [22.1202],
                [21.9651],
                [21.9603],
                [22.3413],
                [21.9561],
                [22.4483],
                [22.0577],
                [22.5733],
                [21.9639],
                [22.4015]], device='cuda:0', grad_fn=<MmBackward0>)
                """
                max_index = torch.argmax(score)
                """
                max_index : tensor(23, device='cuda:0')
                """

                # --------------------------------------------------------------
                # (7) 최종 mask 출력
                # --------------------------------------------------------------
                result_seg = pred_masks[max_index]
                """
                result_seg.shape : torch.Size([428, 640])
                result_seg : tensor([[False, False, False,  ..., False, False, False],
                [False, False, False,  ..., False, False, False],
                [False, False, False,  ..., False, False, False],
                ...,
                [False, False, False,  ..., False, False, False],
                [False, False, False,  ..., False, False, False],
                [False, False, False,  ..., False, False, False]], device='cuda:0')
                """

                # --------------------------------------------------------------
                # (8) 최종 bbox 출력
                # --------------------------------------------------------------
                result_bbox = sorted_mask_generator_pred[max_index]['bbox']
                print(result_bbox)
                print(image[0]['width'])
                bbox_x, bbox_y, bbox_w, bbox_h = (result_bbox[0] + result_bbox[2]) / 2, (result_bbox[1] + result_bbox[3]) / 2, result_bbox[2], result_bbox[3]
                bbox_x, bbox_y, bbox_w, bbox_h = bbox_x / image[0]['width'], bbox_y / image[0]['height'], bbox_w / image[0]['width'], bbox_h / image[0]['height']
                print(bbox_x, bbox_y, bbox_w, bbox_h)

                # --------------------------------------------------------------
                # 1) labels, images 각각 YOLO 폴더 생성
                # --------------------------------------------------------------
                if not os.path.exists(f"{args.json_mother_abs_path}/labels"):
                    os.makedirs(f"{args.json_mother_abs_path}/labels")
                    os.makedirs(f"{args.json_mother_abs_path}/images")

                # --------------------------------------------------------------
                # 2) labes, images 각각 train_val_test_dir 폴더 생성
                # --------------------------------------------------------------
                for train_val_dir in ['train', 'val', 'test']:
                    if not os.path.exists(f"{args.json_mother_abs_path}/labels/{train_val_dir}"):
                        os.makedirs(f"{args.json_mother_abs_path}/labels/{train_val_dir}")
                        os.makedirs(f"{args.json_mother_abs_path}/images/{train_val_dir}")

                set_txt_save_abs_path = f"{args.json_mother_abs_path}/labels/saved_{train_val_dir}.txt"
                with open(set_txt_save_abs_path, 'w') as set_txt:
                    for image_abs_path in images_abs_path:
                        set_txt.write('%s\n' % image_abs_path)

                # --------------------------------------------------------------
                # (7) Visualization ●
                # --------------------------------------------------------------
                if args.plot_mask_result:
                    scores.append(score.cpu().squeeze().numpy()) # score 출력을 위한 배열 ●
                    plot_mask_result(image_filename, real_image, sentence, img_iter, sentence_for_spacy, noun_phrase, result_seg, scores)

            plot_mask_pred(fig_num, pred_masks)

    # %%%%%%%%%%%%%%%%%%%%
    # RefCOCO Dataset ●
    # %%%%%%%%%%%%%%%%%%%%
    # overall = cum_I * 100.0 / cum_U
    # mean_IoU = torch.mean(torch.tensor(m_IoU)) * 100.0
    #
    # save_log_dir = './result_log' # 기존에 경로가 없는 경로일 경우를 대비한 코드 ●
    # save_txt_filename = 'our_method_with_free_solo.txt'
    # if not os.path.exists(save_log_dir):
    #     os.makedirs(save_log_dir)
    # with open(f'{save_log_dir}/{save_txt_filename}', 'w') as txt:
    #     txt.write(f'\n\n CLIP Model: {mode}'
    #         f'\nDataset: {args.dataset} / {args.split} / {args.splitBy}'
    #         f'\nOverall IoU / mean IoU')
    #     txt.write(f'\n{overall:.2f} / {mean_IoU:.2f}')
    # %%%%%%%%%%%%%%%%%%%%

# ==============================================================
# 1. 라이브러리 불러오기
# ==============================================================
if __name__ == "__main__":
    # %%%%%%%%%%%%%%%%%%%%
    # FreeSOLO ●
    # %%%%%%%%%%%%%%%%%%%%
    # opts = ['OUTPUT_DIR', 'training_dir/FreeSOLO_pl', 'MODEL.WEIGHTS', 'checkpoints/FreeSOLO_R101_30k_pl.pth']
    # args.opts = opts
    # print(f"args.opts : {args.opts}")
    # %%%%%%%%%%%%%%%%%%%%

    # --------------------------------------------------------------
    # 1) main문
    # --------------------------------------------------------------
    resized_height, resized_width = 224, 224
    main(args, resized_height, resized_width)