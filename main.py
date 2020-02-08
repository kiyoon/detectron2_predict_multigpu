#!/usr/bin/env python3
import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Extract bounding boxes using Detectron2",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config-file",
        default="/home/appuser/detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--videos-input-dir", help="A directory of input videos. It also extracts features from the Faster R-CNN object detector.")
    parser.add_argument("--images-input-dir", type=str, help="A directory of input images with extension *.jpg. The file names should be the frame number (e.g. 00000000001.jpg)")
    parser.add_argument("--model-weights", type=str, default="detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl", help="Detectron2 object detection model.")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument("--visualise-bbox", action='store_true', help="Save bounding box visualisation.")
#    parser.add_argument("--visualise-feature", action='store_true', help="Save ROI pooled feature visualisation (but only the first frame of a video).")
    parser.add_argument("--divide-job-count", type=int, default=1, help="If there are too many files to process, you may want to divide the job into many processes. This is the number of processes you want to split but the programme doesn't run multiprocess for you. It merely splits the file lists into the number and uses --divide-job-index to assign files to process. Only effective when --videos-input-dir is set.")
    parser.add_argument("--divide-job-index", type=int, default=0, help="If there are too many files to process, you may want to divide the job into many processes. This is the index of process.")

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.1,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

parser = get_parser()
args = parser.parse_args()

if bool(args.video_input) + bool(args.images_input_dir) + bool(args.videos_input_dir) != 1:
    parser.error("--video-input, --video-input-dir and --images-input-dir can't come together and only one must be specified.")

# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger


# import some common libraries
import numpy as np
import cv2
import random
import tqdm
import os
import glob
import pickle
#import multiprocessing as mp


# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from predictor import VisualizationDemo
from detectron2.utils.video_visualizer import VideoVisualizer


# For extracting NMS kept indices

import custom_roi_head
import custom_rcnn


def setup_cfg(args):
    # load config from file and command-line arguments{{{
    cfg = get_cfg()
    # default weights
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.WEIGHTS = args.model_weights
#    cfg.freeze()
    return cfg#}}}

if __name__ == '__main__':
    #mp.set_start_method("spawn", force=True)
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    #im = cv2.imread("./input.jpg")

    cfg = setup_cfg(args)
    # add project-specific config (e.g., TensorMask) if you're not running a model in detectron2's core library


    #predictor = DefaultPredictor(cfg)
    #outputs = predictor(im)

    # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
    # outputs["instances"].pred_classes
    # outputs["instances"].pred_boxes

    # We can use `Visualizer` to draw the predictions on the image.
    #v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    #cv2_imshow(v.get_image()[:, :, ::-1])
    #cv2.imwrite('output.png',v.get_image()[:, :, ::-1])

    if args.images_input_dir:
        all_detection_outputs = {}

        jpgs = sorted(glob.glob(os.path.join(args.images_input_dir, "*.jpg")))
        num_frames = len(jpgs)

        predictor = DefaultPredictor(cfg)
        video_visualiser = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))

        os.makedirs(os.path.join(args.output, 'detection'), exist_ok=True)

        predictions_save_path = os.path.join(args.output, "all_detection_outputs.pkl")
        assert not os.path.isfile(predictions_save_path), predictions_save_path

        for jpg in tqdm.tqdm(jpgs):
            image_basename = os.path.basename(jpg)
            frame_num = int(os.path.splitext(image_basename)[0])

            frame = cv2.imread(jpg)

            visualised_jpg_path = os.path.join(args.output, 'detection', image_basename)
            assert not os.path.isfile(visualised_jpg_path), visualised_jpg_path

            predictions = predictor(frame)["instances"].to("cpu")
            output_dict = {'num_detections': len(predictions), 'detection_boxes': predictions.pred_boxes.tensor.numpy(), 'detection_classes': predictions.pred_classes.numpy(), 'detection_score': predictions.scores.numpy()}
            all_detection_outputs[frame_num] = output_dict

            vis_frame = video_visualiser.draw_instance_predictions(frame[:, :, ::-1], predictions)
            cv2.imwrite(visualised_jpg_path, vis_frame.get_image()[:, :, ::-1])

        with open(predictions_save_path, 'wb') as handle:
            pickle.dump(all_detection_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)


    elif args.video_input:
        demo = VisualizationDemo(cfg)

        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                #fourcc=cv2.VideoWriter_fourcc(*"avc1"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for predictions, vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            print(predictions.pred_classes)
            print(predictions.pred_boxes)
            print(predictions.scores)
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()

    elif args.videos_input_dir:
        # Test

        # Use customised model so that we also get NMS kept indices
        cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeadsCustom"
        cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNNCustom"

#        predictor = TestPredictor(cfg)
#        im = cv2.imread('input.jpg')
#        predictions, kept_indices, roipool_feature = predictor(im)
#        print(predictions[0])
#        print(kept_indices[0].shape)
#        print(roipool_feature[kept_indices[0]].shape)

        demo = VisualizationDemo(cfg, visualise = args.visualise_bbox)

        mp4s = sorted(glob.glob(os.path.join(args.videos_input_dir, "*.mp4")))
        num_files = len(mp4s)
        num_files_per_process = round(num_files / args.divide_job_count)

        if args.divide_job_index == args.divide_job_count -1:       # last process
            mp4s = mp4s[args.divide_job_index * num_files_per_process:]     # to the end
        else:
            mp4s = mp4s[args.divide_job_index * num_files_per_process:(args.divide_job_index+1) * num_files_per_process]

        print("Process from %s to %s" % (mp4s[0], mp4s[-1]))

        for mp4 in tqdm.tqdm(mp4s):
            all_detection_outputs = {}

            video = cv2.VideoCapture(mp4)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames_per_second = video.get(cv2.CAP_PROP_FPS)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            basename = os.path.basename(mp4)

            os.makedirs(args.output, exist_ok=True)
            output_fname = os.path.join(args.output, basename)
            predictions_save_path = os.path.splitext(output_fname)[0] + ".pkl"
            assert not os.path.isfile(predictions_save_path), predictions_save_path

            if args.visualise_bbox:
                assert not os.path.isfile(output_fname), output_fname
                output_file = cv2.VideoWriter(
                    filename=output_fname,
                    # some installation of opencv may not support x264 (due to its license),
                    # you can try other format (e.g. MPEG)
                    #fourcc=cv2.VideoWriter_fourcc(*"x264"),
                    fourcc=cv2.VideoWriter_fourcc(*"avc1"),
                    fps=float(frames_per_second),
                    frameSize=(width, height),
                    isColor=True,
                )

            for frame_num, (predictions, features, vis_frame, frame) in enumerate(demo.run_on_video(video), 1):
                #print(predictions.pred_classes)
                #print(predictions.pred_boxes)
                #print(predictions.scores)
                # predictions visualisation
                if args.visualise_bbox:
                    output_file.write(vis_frame)

                # predictions pickle
                detection_boxes = predictions.pred_boxes.tensor.numpy()
                # XYXY to YXYX
                detection_boxes[:, [0,1]] = detection_boxes[:, [1,0]]
                detection_boxes[:, [2,3]] = detection_boxes[:, [3,2]]

#                # Visualise feature
#                if args.visualise_feature and frame_num == 1:
#                    vis_dir = os.path.splitext(output_fname)[0]
#                    os.makedirs(vis_dir, exist_ok=True)
#                    cv2.imwrite(os.path.join(vis_dir, 'frame%04d.jpg' % (frame_num)), frame)
#                    for i, (detection_box, feature) in enumerate(zip(detection_boxes, features.numpy())):
#                        detection_crop = frame[int(round(detection_box[0])):int(round(detection_box[2])), int(round(detection_box[1])):int(round(detection_box[3]))]
#                        detection_crop = cv2.resize(detection_crop, (256,256), interpolation=cv2.INTER_CUBIC)
#                        cv2.imwrite(os.path.join(vis_dir, 'frame%04d-obj%02d.jpg' % (frame_num,i)), detection_crop)
#
#                        for f, channel in enumerate(feature):
#                            channel += 10
#                            channel /= 20
#                            channel *= 255
#                            vis_img = cv2.resize(channel, (256,256), interpolation=cv2.INTER_CUBIC)
#                            cv2.imwrite(os.path.join(vis_dir, 'frame%04d-obj%02d-channel%03d.jpg' % (frame_num,i,f)), vis_img)


                # YXYX to YXHW
                detection_boxes[:, 2] -= detection_boxes[:, 0]
                detection_boxes[:, 3] -= detection_boxes[:, 1]

                output_dict = {'num_detections': len(predictions), 'detection_boxes': detection_boxes, 'detection_classes': predictions.pred_classes.numpy(), 'detection_score': predictions.scores.numpy(), 'feature': features.numpy()}
                all_detection_outputs[frame_num] = output_dict

            video.release()
            if args.visualise_bbox:
                output_file.release()

            with open(predictions_save_path, 'wb') as handle:
                pickle.dump(all_detection_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
