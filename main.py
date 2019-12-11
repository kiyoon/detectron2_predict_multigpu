import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="/detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--no-parallel", action="store_false", dest='parallel', help="Disable parallel processing using multiple GPUs.")
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

args = get_parser().parse_args()

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
import multiprocessing as mp


# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from predictor import VisualizationDemo

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # default weights
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    cfg.freeze()
    return cfg

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    im = cv2.imread("./input.jpg")

    cfg = setup_cfg(args)
    # add project-specific config (e.g., TensorMask) if you're not running a model in detectron2's core library

    demo = VisualizationDemo(cfg, parallel = args.parallel)

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
            #fourcc=cv2.VideoWriter_fourcc(*"x264"),
            fourcc=cv2.VideoWriter_fourcc(*"avc1"),
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
