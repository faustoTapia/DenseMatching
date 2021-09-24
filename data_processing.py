import cv2
import numpy as np
from model_selection import select_model
from utils_flow.util_optical_flow import flow_to_image
from pathlib import Path
import torch

from utils_flow.visualization_utils import overlay_semantic_mask
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from test_models import boolean_string, pad_to_same_shape

def warp_data_depths(args, data_path: Path, depth_size=(480, 640), confidence_threshold = 0.5, store_dir_suffix=""):
    data_lists = get_lists_from_day_datasplit(data_path)

    with torch.no_grad():
        network, estimate_uncertainty = select_model(
            args.model, args.pre_trained_model, args, args.optim_iter, None,
            path_to_pre_trained_models=args.pre_trained_models_dir)

        for i in range(len(data_lists['zed_left']['imgs'])):
            zed_left_img = data_lists['zed_left']['imgs'][i]
            zed_left_depth = data_lists['zed_left']['depths'][i]
            zed_right_img = data_lists['zed_right']['imgs'][i]
            zed_right_depth = data_lists['zed_right']['depths'][i]

            pxl_img = data_lists['pxl']['imgs'][i]
            hua_img = data_lists['hua']['imgs'][i]

            parent_dir = zed_left_img.parent.parent
            pxl_out_dir = parent_dir / ('pxl_projected' + store_dir_suffix)
            hua_out_dir = parent_dir / ('hua_projected' + store_dir_suffix)
            pxl_conf_dir = parent_dir / ('pxl_projected_conf' + store_dir_suffix)
            hua_conf_dir = parent_dir / ('hua_projected_conf' + store_dir_suffix)

            out_dirs = [hua_out_dir, pxl_out_dir]
            out_confidence_dirs = [hua_conf_dir, pxl_conf_dir]
            for out_dir in out_dirs + out_confidence_dirs:
                if not out_dir.is_dir():
                    out_dir.mkdir(exist_ok=True)

            imgs_sets = [(zed_left_img, zed_left_depth, hua_img), (zed_right_img, zed_right_depth, pxl_img)]

            for i, imgs_set in enumerate(imgs_sets):
                query_img = cv2.imread(imgs_set[0].__str__())
                query_depth = cv2.imread(imgs_set[1].__str__())
                ref_img = cv2.imread(imgs_set[2].__str__())
                ref_img = cv2.resize(ref_img, depth_size)

                reference_image_shape = ref_img.shape[:2]

                query_img_, reference_image_ = pad_to_same_shape(query_img, ref_img)

                query_img_ = torch.from_numpy(query_img).permute(2, 0, 1).unsqueeze(0)
                reference_image_ = torch.from_numpy(reference_image_).permute(2, 0, 1).unsqueeze(0)

                if estimate_uncertainty:
                    if args.flipping_condition:
                        raise NotImplementedError('No flipping condition with PDC-Net for now')
                    estimated_flow, uncertainty_components = network.estimate_flow_and_confidence_map(query_img_,
                                                                                                  reference_image_,
                                                                                                  mode='channel_first')
                    confidence_map = uncertainty_components['p_r'].squeeze().detach().cpu().numpy()
                    confidence_map = confidence_map[:ref_img.shape[0], :ref_img.shape[1]]
                else:
                    if args.flipping_condition and 'GLUNet' in args.model:
                        estimated_flow = network.estimate_flow_with_flipping_condition(query_img_, reference_image_,
                                                                                       mode='channel_first')
                    else:
                        estimated_flow = network.estimate_flow(query_img_, reference_image_, mode='channel_first')
                estimated_flow_numpy = estimated_flow.squeeze().permute(1, 2, 0).cpu().numpy()
                estimated_flow_numpy = estimated_flow_numpy[:ref_img.shape[0], :ref_img.shape[1]]
                # removes the padding

                warped_query_image = remap_using_flow_fields(query_img, estimated_flow_numpy[:, :, 0],
                                                             estimated_flow_numpy[:, :, 1]).as_type(np.uint8)

                warped_query_depth = remap_using_flow_fields(query_depth, estimated_flow_numpy[:, :, 0],
                                                             estimated_flow_numpy[:, :, 1]).as_type(np.uint16)

                if estimate_uncertainty:
                    color = [255, 102, 51]

                    confident_mask = (confidence_map > confidence_threshold).astype(np.uint8)
                    confident_warped = overlay_semantic_mask(warped_query_image, ann=255 - confident_mask * 255,
                                                             color=color)


def get_lists_from_day_datasplit(data_path: Path):

    pxl_imgs_dir = data_path / "pxl_img"
    pxl_imgs_list = sorted(list(pxl_imgs_dir.glob("*.jpg")))
    pxl_depths_dir = data_path / "pxl_depth"
    pxl_depths_list = sorted(list(pxl_depths_dir.glob("*.jpg")))

    hua_imgs_dir = data_path / "hua_img"
    hua_imgs_list = sorted(list(hua_imgs_dir.glob("*.jpg")))
    hua_depths_dir = data_path / "hua_depth"
    hua_depths_list = sorted(list(hua_depths_dir.glob("*.jpg")))

    zed_left_imgs_dir = data_day_path / "zed_left_img"
    zed_left_imgs_list = sorted(list(zed_left_imgs_dir.glob("*")))
    zed_left_depths_dir = data_day_path / "zed_left_depth"
    zed_left_depths_list = sorted(list(zed_left_depths_dir.glob("*")))

    zed_right_imgs_dir = data_day_path / "zed_right_img"
    zed_right_imgs_list = sorted(list(zed_right_imgs_dir.glob("*")))
    zed_right_depths_dir = data_day_path / "zed_right_depth"
    zed_right_depths_list = sorted(list(zed_right_depths_dir.glob("*")))

    imgs_lists = {
        "pxl": {"imgs": pxl_imgs_list, "depths": pxl_depths_list},
        "hua": {"imgs": hua_imgs_list, "depths": hua_depths_list},
        "zed_left": {"imgs": zed_left_imgs_list, "depths": zed_left_depths_list},
        "zed_right": {"imgs": zed_right_imgs_list, "depths": zed_right_depths_list},
    }
    return imgs_lists

if __name__=="__main__":
    data_day_path = Path("D:\\thesisData\\20210817_for_testing")


