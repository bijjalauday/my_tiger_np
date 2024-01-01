import torch

from src.ta_pet_id.yolo.utils.datasets import Custom_LoadImages
from src.ta_pet_id.yolo.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
)
from src.ta_pet_id.yolo.utils.torch_utils import time_sync


def yolo_detect(
    model,
    device,
    img_array,
    imgsz=[256, 256],
    conf_thres=0.4,
    iou_thres=0.4,
    filter_class=None,
    max_det=1000,
    half=False,
):
    """
    Helper function for predict function.
    This is where the images are passed through yolo model and nms stage.

    Parameters
    ----------
    model : loaded yolo model
     loaded yolo model
    device : object
      yolo loaded device type object  for model loading
    img_array : list
       list of numpy array images read in BGR format
    imgsz : list
      image size at which we want to do yolo predictions,[256,256]
      gives best results because training was done at 256.
    conf_thres : int
       consider predictions with confidence score only above this value.
    iou_thres : int
       threshold for doing non-max-suppression
    filter_class : int
       pass the class label that you want to detect alone,if None detects all trained classes
    max_det : int
       Max no.of detections you want to allow per image
    half : Boolean
       use FP16 half-precision inference

    Returns
    -------
     list of predictions for the input param img_array

       list of (list of class labels) , (list of bb co-ordinates) ,(list of confidence scores) and (list of cropped faces)
       If no detection or multiple detections are made it returns None for that image

    """
    stride, names, pt, jit, onnx, engine = (
        model.stride,
        model.names,
        model.pt,
        model.jit,
        model.onnx,
        model.engine,
    )
    imgsz = check_img_size(imgsz, s=stride)
    device = device
    half &= (
        pt or jit or engine
    ) and device.type != "cpu"  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    final_preds = []
    for img in img_array:
        im, im_orig = Custom_LoadImages(
            numpy_img=img, img_size=imgsz, stride=stride, auto=pt
        )
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        t2 = time_sync()
        dt[0] += t2 - t1

        # print(im.shape)
        # print("im : ", im)

        # Inference
        pred = model(im, augment=False, visualize=False)
        if type(pred) == list:
            pred = pred[0]

        t3 = time_sync()
        dt[1] += t3 - t2
        # print(len(pred))
        # print("pred : ", pred)
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=filter_class,
            agnostic=False,
            max_det=max_det,
        )
        dt[2] += time_sync() - t3

        # print("pred  : ", pred)

        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], im_orig.shape
                ).round()
                det = det.tolist()
                final_preds.append(det)
            else:
                final_preds.append([])

        # print("final preds : ", final_preds)
    output = get_yolo_outputs(img_array, final_preds)

    return output


def get_yolo_outputs(img_array, yolo_pred):
    conf_scores = []
    pet_types = []
    pet_faces_loc = []
    pet_faces = []
    for i in range(0, len(yolo_pred)):
        if len(yolo_pred[i]) == 1:
            # get pet type
            pred_pet_type = yolo_pred[i][0][-1]
            conf_scores.append(yolo_pred[i][0][-2])
            if pred_pet_type == 0.0:
                pred_pet_type = "cat"
            elif pred_pet_type == 1.0:
                pred_pet_type = "dog"
            else:
                raise Exception(f"Invalid {pred_pet_type} predicted class!")
            pet_types.append(pred_pet_type)
            # get pet's face location
            x_min, x_max = int(yolo_pred[i][0][0]), int(yolo_pred[i][0][2])
            y_min, y_max = int(yolo_pred[i][0][1]), int(yolo_pred[i][0][3])
            pet_faces_loc.append((x_min, x_max, y_min, y_max))
            # get cropped face
            cropped_img = img_array[i][y_min:y_max, x_min:x_max, :]
            pet_faces.append(cropped_img)
            # output = [pet_types, pet_faces_loc, conf_scores, pet_faces]
        elif len(yolo_pred[i]) == 0:
            pred_pet_type = "other"
            pet_types.append(pred_pet_type)
            pet_faces_loc.append(None)
            pet_faces.append(None)
            conf_scores.append(None)
            # output = [pet_types, pet_faces_loc, conf_scores, pet_faces]
        else:
            conf_scores_mul = []
            pet_types_mul = []
            pet_faces_loc_mul = []
            pet_faces_mul = []
            for row in range(0, len(yolo_pred[i])):
                # class name
                pred_pet_type = yolo_pred[i][row][-1]
                if pred_pet_type == 0.0:
                    pred_pet_type = "cat"
                elif pred_pet_type == 1.0:
                    pred_pet_type = "dog"
                else:
                    raise Exception(f"Invalid {pred_pet_type} predicted class!")
                pet_types_mul.append(pred_pet_type)
                # face loaction or bounding box coordinates
                x_min, x_max = int(yolo_pred[i][row][0]), int(yolo_pred[i][row][2])
                y_min, y_max = int(yolo_pred[i][row][1]), int(yolo_pred[i][row][3])
                pet_faces_loc_mul.append((x_min, x_max, y_min, y_max))
                # get cropped face
                cropped_img = img_array[i][y_min:y_max, x_min:x_max, :]
                # print(cropped_img)
                pet_faces_mul.append(cropped_img)
                # get confidence score
                conf_scores_mul.append(yolo_pred[i][row][-2])
                # output
            pet_types.append(pet_types_mul)
            conf_scores.append(conf_scores_mul)
            pet_faces_loc.append(pet_faces_loc_mul)
            pet_faces.append(pet_faces_mul)
    output = [pet_types, pet_faces_loc, conf_scores, pet_faces]
    return output
