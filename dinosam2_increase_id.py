import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionatyModel, ObjectInfo
import json
import copy
'''
做好之前请看好几个参数的设置
'''
def one_frame_append_to_evl_txt(frame_index, frame_mask_info:MaskDictionatyModel, output_txt_path):
    with open(output_txt_path, 'a') as f:
        for object_info in frame_mask_info.labels.values():
            obj_id = object_info.instance_id
            x1 = object_info.x1
            y1 = object_info.y1
            w = object_info.x2 - object_info.x1
            h = object_info.y2 - object_info.y1
            # x1, y1, w, h = extract_bbox(mask[0])  # [1,h,w]
            # # Assume a constant score of 0.9 for this example
            score = 0.9
            line = f"{frame_index},{obj_id},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{score},-1,-1,-1\n"
            f.write(line)
"""
Step 1: Environment settings and model initialization
"""
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)


# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
which_dataset = "dancetrack"
whether_use_real_labels = True
detection_model = "grounding-dino"  # or "yolox"
if which_dataset == "dancetrack":
    text = "person."

    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  
    base_video_dir_of_dataset = "/data/zly/mot_data/DanceTrack/"
    split = "val/"
    # sequence_num = "dancetrack0000/"
    sequence_num_list = os.listdir(os.path.join(base_video_dir_of_dataset, split))
    sequence_num_list.sort()
    new_sequence_num_list = copy.deepcopy(sequence_num_list)
    for seq in sequence_num_list:
        if "58" not in seq:
            new_sequence_num_list.remove(seq)
        else:
            break
for sequence_num in new_sequence_num_list:
    if which_dataset == "dancetrack":
        video_dir = os.path.join(base_video_dir_of_dataset, split, sequence_num,"img1")
    else:
        raise NotImplementedError("Only support dancetrack dataset")
    # 'output_dir' is the directory to save the annotated frames
    if whether_use_real_labels == True:
        output_dir = os.path.join( base_video_dir_of_dataset , "outputs_real_labels_sam2", split, sequence_num)
        trackEval_txt_path = os.path.join(base_video_dir_of_dataset,"outputs_real_labels_sam2", split, "track_results", sequence_num.split("/")[0] + ".txt") # 保存成trackeval的格式
    elif detection_model == "grounding-dino":
        output_dir = os.path.join( base_video_dir_of_dataset , "outputs_gd1.0", split, sequence_num)
        trackEval_txt_path = os.path.join(base_video_dir_of_dataset,"outputs_real_labels_sam2", split, "track_results", sequence_num.split("/")[0] + ".txt") # 保存成trackeval的格式
    else:
        raise NotImplementedError("Only support grounding-dino model")
    
    if os.path.exists(trackEval_txt_path):
        os.remove(trackEval_txt_path)
    # create the output directory
    CommonUtils.creat_dirs(output_dir)
    os.makedirs(os.path.dirname(trackEval_txt_path), exist_ok=True)
    with open(trackEval_txt_path, 'w') as f:
        pass
    if whether_use_real_labels:
        input_boxes_list= {}
        txt_path = os.path.join(base_video_dir_of_dataset, split, sequence_num, "gt", "gt.txt")
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                img_id = linelist[0]
                frame_index_correspond_to_array_index = int(img_id) - 1 # MOT数据集遵循MOT17,MOT20,DANCETRACK的规则,img从1开始
                obj_id = linelist[1]
                bbox = [float(linelist[2]), float(linelist[3]), 
                        float(linelist[2]) + float(linelist[4]), 
                        float(linelist[3]) + float(linelist[5])]
                if input_boxes_list.get(frame_index_correspond_to_array_index) is None:
                    input_boxes_list[frame_index_correspond_to_array_index] = []
                input_boxes_list[frame_index_correspond_to_array_index].append(bbox)
    output_video_path = os.path.join(output_dir, "output.mp4")
    mask_data_dir = os.path.join(output_dir, "mask_data")
    json_data_dir = os.path.join(output_dir, "json_data")
    result_dir = os.path.join(output_dir, "result")
    CommonUtils.creat_dirs(mask_data_dir)
    CommonUtils.creat_dirs(json_data_dir)
    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=video_dir)
    step = 5 # the step to sample frames for Grounding DINO predictor

    sam2_masks = MaskDictionatyModel()
    PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
    objects_count = 0

                    
    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for all frames
    """
    print("Total frames:", len(frame_names))
    for start_frame_idx in range(0, len(frame_names), step):
    # prompt grounding dino to get the box coordinates on specific frame
        print("start_frame_idx", start_frame_idx)
        # continue
        img_path = os.path.join(video_dir, frame_names[start_frame_idx])
        image = Image.open(img_path)
        image_base_name = frame_names[start_frame_idx].split(".")[0]
        mask_dict = MaskDictionatyModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")
        if whether_use_real_labels:
            input_boxes = torch.tensor(input_boxes_list[start_frame_idx])
            OBJECTS = ["person" for i in range(len(input_boxes_list))]
        else:
            # run Grounding DINO on the image
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = grounding_model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.25,
                text_threshold=0.25,
                target_sizes=[image.size[::-1]]
            )
            
            # process the detection results
            input_boxes = results[0]["boxes"] # .cpu().numpy()
            # print("results[0]",results[0])
            OBJECTS = results[0]["labels"]

        # prompt SAM image predictor to get the mask for the object
        image_predictor.set_image(np.array(image.convert("RGB")))


        # prompt SAM 2 image predictor to get the mask for the object
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,  # 这里也不多任务预测
        )
        # convert the mask shape to (n, H, W)
        if masks.ndim == 2:
            masks = masks[None]
            scores = scores[None]
            logits = logits[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        """
        Step 3: Register each object's positive points to video predictor
        """

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if mask_dict.promote_type == "mask":
            mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)
        else:
            raise NotImplementedError("SAM 2 video predictor only support mask prompts")


        """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        """
        objects_count = mask_dict.update_masks(tracking_annotation_dict=sam2_masks, iou_threshold=0.8, objects_count=objects_count) # 在这里和之前的进行关联
        print("objects_count", objects_count)
        video_predictor.reset_state(inference_state)
        if len(mask_dict.labels) == 0:
            print("No object detected in the frame, skip the frame {}".format(start_frame_idx))
            continue
        video_predictor.reset_state(inference_state)
        # 这里相当于在每一帧的每一个物体都是添加mask
        for object_id, object_info in mask_dict.labels.items():
            frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                    inference_state,
                    start_frame_idx,
                    object_id,
                    object_info.mask,
                )
        
        video_segments = {}  # output the following {step} frames tracking masks
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
            frame_masks = MaskDictionatyModel()
            
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
                object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id))
                object_info.update_box() # mask 转换成 box
                frame_masks.labels[out_obj_id] = object_info
                image_base_name = frame_names[out_frame_idx].split(".")[0]
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]

            video_segments[out_frame_idx] = frame_masks
            sam2_masks = copy.deepcopy(frame_masks)

        print("video_segments:", len(video_segments))
        """
        Step 5: save the tracking masks and json files
        """
        for frame_idx, frame_masks_info in video_segments.items():
            if frame_idx >= start_frame_idx + step: # 理论上跳过batch最后一帧
                break
            mask = frame_masks_info.labels
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
            for obj_id, obj_info in mask.items():
                mask_img[obj_info.mask == True] = obj_id

            mask_img = mask_img.numpy().astype(np.uint16)
            np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

            json_data = frame_masks_info.to_dict()
            json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
            with open(json_data_path, "w") as f:
                json.dump(json_data, f)
            one_frame_append_to_evl_txt(frame_idx, frame_masks_info, trackEval_txt_path)


    """
    Step 6: Draw the results and save the video
    """
    CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)

    create_video_from_images(result_dir, output_video_path, frame_rate=30)