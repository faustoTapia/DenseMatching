import cv2
import numpy as np
from model_selection import select_model
from utils_flow.util_optical_flow import flow_to_image
from pathlib import Path
import torch
import argparse
import time

from utils_flow.visualization_utils import overlay_semantic_mask
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from test_models import boolean_string, pad_to_same_shape
from validation.test_parser import define_pdcnet_parser

def warp_data_depths(args, data_path: Path, input_size=(1024, 768), depth_size=(640, 480), confidence_threshold = 0.5, store_dir_suffix=""):
    data_lists = get_lists_from_day_datasplit(data_path)
    with torch.no_grad():
        # print("Initial memory occupied: {}MB".format(torch.cuda.memory_allocated(device)/1e6))
        network, estimate_uncertainty = select_model(
            args.model, args.pre_trained_model, args, args.optim_iter, local_optim_iter,
            path_to_pre_trained_models=args.pre_trained_models_dir)
        # print("Memory occupied after model loading: {}MB".format(torch.cuda.memory_allocated(device)/1e6))

        prev_time = time.time()
        for img_indx in range(len(data_lists['zed_left']['imgs'])):
            torch.cuda.empty_cache()
            zed_left_img = data_lists['zed_left']['imgs'][img_indx]
            zed_left_depth = data_lists['zed_left']['depths'][img_indx]
            zed_right_img = data_lists['zed_right']['imgs'][img_indx]
            zed_right_depth = data_lists['zed_right']['depths'][img_indx]

            pxl_img = data_lists['pxl']['imgs'][img_indx]
            hua_img = data_lists['hua']['imgs'][img_indx]

            parent_dir = zed_left_img.parent.parent
            pxl_out_dir = parent_dir / ('pxl_projected' + store_dir_suffix)
            hua_out_dir = parent_dir / ('hua_projected' + store_dir_suffix)
            pxl_conf_dir = parent_dir / ('pxl_projected_conf' + store_dir_suffix)
            pxl_confviz_dir = parent_dir / ('pxl_projected_confviz' + store_dir_suffix)
            hua_conf_dir = parent_dir / ('hua_projected_conf' + store_dir_suffix)
            hua_confviz_dir = parent_dir / ('hua_projected_confviz' + store_dir_suffix)

            out_dirs = [hua_out_dir, pxl_out_dir]
            out_confidence_dirs = [hua_conf_dir, pxl_conf_dir]
            out_viz_dirs = [hua_confviz_dir, pxl_confviz_dir]
            for out_dir in out_dirs + out_confidence_dirs + out_viz_dirs:
                if not out_dir.is_dir():
                    out_dir.mkdir(exist_ok=True)

            imgs_sets = [(zed_left_img, zed_left_depth, hua_img), (zed_right_img, zed_right_depth, pxl_img)]
            img_sets_names = ['Huawei', 'Pxl']

            for img_set_indx, imgs_set in enumerate(imgs_sets):
                # print("Memory begining loop: {}MB".format(torch.cuda.memory_allocated(device)/1e6))
                print("Print Sample {}: {}".format(img_indx, img_sets_names[img_set_indx]))
                query_img = cv2.imread(imgs_set[0].__str__())
                query_depth = cv2.imread(imgs_set[1].__str__(), cv2.IMREAD_UNCHANGED)
                ref_img = cv2.imread(imgs_set[2].__str__())
                if depth_size is not None:
                    ref_img = cv2.resize(ref_img, depth_size)
                if input_size is not None:
                    query_img = cv2.resize(query_img, input_size)
                    query_depth = cv2.resize(query_depth, input_size)

                reference_image_shape = ref_img.shape[:2]

                query_img_, reference_image_ = pad_to_same_shape(query_img, ref_img)

                query_img_ = torch.from_numpy(query_img).permute(2, 0, 1).unsqueeze(0)
                reference_image_ = torch.from_numpy(reference_image_).permute(2, 0, 1).unsqueeze(0)

                if estimate_uncertainty:
                    if args.flipping_condition:
                        raise NotImplementedError('No flipping condition with PDC-Net for now')
                    
                    # print("Model input shapes: q/r: {}/{}".format(query_img_.shape, reference_image_.shape))
                    estimated_flow, uncertainty_components = network.estimate_flow_and_confidence_map(query_img_,
                                                                                                  reference_image_,
                                                                                                  mode='channel_first')
                    confidence_map = uncertainty_components['p_r'].squeeze().detach().cpu().numpy()
                    confidence_map = confidence_map[:ref_img.shape[0], :ref_img.shape[1]]
                    confidence_map_img = (confidence_map * 255).astype(np.uint8)
                    print("Coinfidence map info: min={}, max={}, type={}".format(np.min(confidence_map), np.max(confidence_map), confidence_map.dtype))
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
                                                             estimated_flow_numpy[:, :, 1]).astype(np.uint8)

                warped_query_depth = remap_using_flow_fields(query_depth, estimated_flow_numpy[:, :, 0],
                                                             estimated_flow_numpy[:, :, 1]).astype(np.uint16)

                depth_path = out_dirs[img_set_indx] / "depth_{:04d}.png".format(img_indx)
                cv2.imwrite(depth_path.__str__(), warped_query_depth)
                print("Stored img: {}".format(depth_path.__str__()))

                if estimate_uncertainty:
                    color = [255, 102, 51]

                    confident_mask = (confidence_map > confidence_threshold).astype(np.uint8)
                    confident_warped = overlay_semantic_mask(warped_query_image, ann=255 - confident_mask * 255,
                                                             color=color)
                    
                    confidence_img_path = out_confidence_dirs[img_set_indx] / "imgc_{:04}.png".format(img_indx)
                    cv2.imwrite(confidence_img_path.__str__(), confidence_map_img)
                    print("Stored img: {}".format(confidence_img_path.__str__()))

                else:
                    confident_warped = warped_query_image
                
                warped_img_path = out_viz_dirs[img_set_indx] / "imgw_{:04}.png".format(img_indx)
                cv2.imwrite(warped_img_path.__str__(), confident_warped)
                print("Stored img: {}".format(warped_img_path.__str__()))

                curr_time = time.time()
                delta_time = curr_time - prev_time
                prev_time = curr_time

                print("Sample processing time: {}s".format(delta_time))
                
                # print("Memory occupied end inference: {}MB".format(torch.cuda.memory_allocated(device)/1e6))
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
def get_lists_from_day_datasplit(data_path: Path):

    pxl_imgs_dir = data_path / "pxl_img"
    pxl_imgs_list = sorted(list(pxl_imgs_dir.glob("*.jpg")))
    pxl_depths_dir = data_path / "pxl_depth"
    pxl_depths_list = sorted(list(pxl_depths_dir.glob("*.jpg")))

    hua_imgs_dir = data_path / "hua_img"
    hua_imgs_list = sorted(list(hua_imgs_dir.glob("*.jpg")))
    hua_depths_dir = data_path / "hua_depth"
    hua_depths_list = sorted(list(hua_depths_dir.glob("*.jpg")))

    zed_left_imgs_dir = data_path / "zed_left_img"
    zed_left_imgs_list = sorted(list(zed_left_imgs_dir.glob("*")))
    zed_left_depths_dir = data_path / "zed_left_depth"
    zed_left_depths_list = sorted(list(zed_left_depths_dir.glob("*")))

    zed_right_imgs_dir = data_path / "zed_right_img"
    zed_right_imgs_list = sorted(list(zed_right_imgs_dir.glob("*")))
    zed_right_depths_dir = data_path / "zed_right_depth"
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

    parser = argparse.ArgumentParser(description='Test models on a pair of images')
    parser.add_argument('--model', type=str, help='Model to use', required=True)
    parser.add_argument('--flipping_condition', dest='flipping_condition',  default=False, type=boolean_string,
                        help='Apply flipping condition for semantic data and GLU-Net-based networks ? ')
    parser.add_argument('--optim_iter', type=int, default=3,
                        help='Number of optim iter for Global GOCor, if applicable')
    parser.add_argument('--local_optim_iter', dest='local_optim_iter', default=None,
                        help='Number of optim iter for Local GOCor, if applicable')
    parser.add_argument('--pre_trained_models_dir', type=str, default='pre_trained_models/',
                        help='Directory containing the pre-trained-models.')
    parser.add_argument('--pre_trained_model', type=str, help='Name of the pre-trained-model', required=True)
    parser.add_argument('--data_path', type=str,
                        help='Path to dataset file structure.', required=True)
    subparsers = parser.add_subparsers(dest='network_type')
    define_pdcnet_parser(subparsers)
    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # either gpu or cpu

    local_optim_iter = args.optim_iter if not args.local_optim_iter else int(args.local_optim_iter)

    warp_data_depths(args, Path(args.data_path), input_size=(1024, 768), depth_size=(960,720), store_dir_suffix="")
    # warp_data_depths(args, Path(args.data_path), depth_size=(640,480), store_dir_suffix="640x480")
