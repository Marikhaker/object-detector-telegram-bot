# Imports

import numpy as np
import ultralytics
from ultralytics import YOLO
import torch
import os
import shutil
import subprocess
import json
from io import StringIO
import sys
import cv2
import torch
import datetime
import zipfile

from typing import Dict, List, Tuple, Any


def play_video(filepath):
    # importing libraries
    cap = cv2.VideoCapture(filepath)
    if (cap.isOpened() == False):
        print("Error opening video file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv2.imshow('Frame', frame)
            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available")
        return 1
    else:
        print("No GPU, only CPU ")
        return 0


class Metadata:
    def __init__(self, filepath):
        # https://ffmpeg.org/ffprobe.html
        import re
        self.filepath = filepath
        self.metadata = self.extract_metadata(filepath)
        # print(self.metadata)
        i = 0
        while self.metadata['streams'][i]['codec_type'] == "audio":
            i += 1
        self.codec_name = self.metadata['streams'][i]['codec_name']
        self.duration = float(self.metadata['streams'][i]['duration'])
        self.frames = int(self.metadata['streams'][i]['nb_frames'])
        self.width = self.metadata['streams'][i]['width']
        self.height = self.metadata['streams'][i]['height']
        # self.frame_rate = int(re.search(r'\d+', self.metadata['streams'][0]['avg_frame_rate']).group())
        self.frame_rate = float("{:.3f}".format(self.frames / self.duration))

    def get_metadata_str(self) -> str:
        buffer = StringIO()
        sys.stdout = buffer
        print('Codec_name:', self.codec_name)
        print('Duration:', self.duration)
        print('Frames:', self.frames)
        print('Resolution (Width x Height):', self.width, 'x', self.height)
        print('Frame rate:', self.frame_rate)  # bug framerate = 0
        print_output = buffer.getvalue()
        sys.stdout = sys.__stdout__

        print(f"->{print_output}<-")
        return str(print_output)

    def extract_metadata(self, filepath):
        command = ['ffprobe', '-v', 'error', '-show_entries', 'stream', '-of', 'json', filepath]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            metadata = json.loads(result.stdout.decode('utf-8'))
            return metadata
        else:
            error_message = result.stderr.decode('utf-8').strip()
            raise Exception(f'Error running ffprobe: {error_message}')


class Detection:
    def __init__(self, model_name='yolov8s.pt'):
        self.model_name = model_name
        print(f"model_name = {model_name}")
        self.gpu_available = check_gpu()

        ultralytics.checks()
        # model = YOLO('yolov8n.yaml').load('yolov8n.pt')
        self.model = YOLO(model_name)
        self.detect_path = "runs/detect/"
        if os.path.exists(self.detect_path):
            shutil.rmtree(self.detect_path)

        # model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
        # n - smallest, s - balance, m - good

        self.obj365_model = None

    def get_dir_filename_ext(self, filepath):
        directory, file_name = os.path.split(self.filepath)
        file_name, extension = os.path.splitext(file_name)
        return directory, file_name, extension

    def get_classes_names(self) -> [str]:
        self.available_classes_names: Dict[int, str] = self.model.names
        self.available_classes_names_rev = dict((v, k) for k, v in self.available_classes_names.items())
        # result = str([str(i) + '\n' for i in self.available_classes_names.values()])

        result = ''.join([str(elem) + ' - ' + str(self.available_classes_names[elem]) + '\n' for elem in
                          self.available_classes_names])
        return result

    def extract_result_file(self, input_path="runs/detect/", destination_folder_path=""):

        # Split file path into directory, base file name, and extension
        directory, file_name = os.path.split(self.filepath)
        self.filename = file_name
        file_name, extension = os.path.splitext(file_name)

        predict_file_path = self.detect_path + "predict/" + self.filename

        # Add "_ppc" to file name and reassemble file path
        self.result_filepath = os.path.join(directory, file_name + "_ppc" + extension)  # preprocessed

        if os.path.exists(self.detect_path):
            shutil.copy(predict_file_path, os.path.join(destination_folder_path, self.result_filepath))
            # Delete folder and its contents
            shutil.rmtree(self.detect_path)

    def get_metadata(self, filepath):
        self.metadata = Metadata(filepath)
        return self.metadata

    def compress_video(self, filepath):
        tmp_output_path = filepath
        directory, file_name, extension = self.get_dir_filename_ext(filepath)
        if directory != "":
            output_path = directory + "/" + file_name + "_comp" + extension
        else:
            output_path = file_name + "_comp" + extension  # added to stop saving to absolute path when no dir name
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_output_path,
                "-crf",
                "20",
                "-preset",
                "veryfast",
                "-vcodec",
                "libx264",
                output_path,
                '-loglevel',
                'quiet'
            ]
        )
        return output_path

    def set_ppc_params(self, filepath="tonylife2.mp4", vid_stride=5, imgsz=32 * 8, conf=0.2, iou=0.5, half=False):
        self.filepath = filepath
        self.imgsz = imgsz
        self.vid_stride = vid_stride
        self.conf = conf
        self.iou = iou
        self.half = half
        self.get_metadata(self.filepath)

    def preprocess_video(self, compress_video=False):

        import os
        if os.path.exists(self.filepath):
            self.results = self.model.predict(source=self.filepath,
                                              save=True,
                                              vid_stride=self.vid_stride,
                                              # imgsz= max(width, height),
                                              imgsz=self.imgsz,
                                              conf=self.conf,
                                              iou=self.iou,
                                              half=self.half,
                                              # save_txt=True,
                                              save_txt=False,
                                              save_conf=True,
                                              )
        print(len(self.results))

        # Results = ultralytics.yolo.engine.results.Results
        # results: Results = model_results[0].cpu()

        self.get_classes_names()
        classes: Dict[int, str] = self.available_classes_names
        # classes: Dict[int, str] = self.results[0].names
        # self.available_classes_names = classes

        classes_names_list = [[classes[cls] for cls in result.boxes.cls.cpu().numpy()] for result in self.results]
        classes_idx_list = [result.boxes.cls.cpu().numpy() for result in self.results]
        conf_list = [result.boxes.conf.cpu().numpy() for result in self.results]

        self.box_coords_list = [result.boxes.xyxy.cpu().numpy() for result in self.results]

        self.cls_and_conf_each_frame = list(zip(classes_names_list, conf_list, classes_idx_list))
        print(self.cls_and_conf_each_frame)
        self.unique_classes = self.get_unique_classes()
        print(self.unique_classes)

        directory, file_name = os.path.split(self.filepath)
        self.extract_result_file(directory)

        if compress_video:
            self.result_filepath = self.compress_video(self.result_filepath)

        # From the boxes get the predictions
        # results_with_probs: List[Tuple[Results, Tuple[str]]] = [
        #     (result, tuple[[(classes[cls]) for cls in result.boxes.cls.numpy()], result.boxes.conf.numpy()]) for result
        #     in self.results]
        # print(results_with_probs)

    def get_unique_classes(self) -> (dict, str):
        unique_items = {}

        for row in self.cls_and_conf_each_frame:
            for item in row[0]:
                if item not in unique_items:
                    unique_items[item] = 1
                else:
                    unique_items[item] += 1

        sorted_items = sorted(unique_items.items(), key=lambda x: x[1], reverse=True)

        buffer = StringIO()
        sys.stdout = buffer
        # for item, count in unique_items.items():
        #    print(f"{str(self.available_classes_names_rev[item]) + '.':<4}{item:<18} {count}")
        for item in sorted_items:
            print(f"{str(self.available_classes_names_rev[item[0]]) + '.':<4}{item[0]:<18} {item[1]}")
        print_output = buffer.getvalue()
        sys.stdout = sys.__stdout__

        print(f"->{print_output}<-")
        return unique_items, print_output

    def show_result(self):
        import cv2, imutils

        # from google.colab.patches import cv2_imshow
        img_S = imutils.resize(self.results[0][0].plot(probs=True, boxes=True), width=320)
        cv2.imshow('', img_S)
        cv2.waitKey(0)
        print("here")
        print(self.cls_and_conf_each_frame[0][0][0], self.cls_and_conf_each_frame[0][1][0])

        import moviepy.editor
        result_filepath = self.result_filepath
        play_video(result_filepath)

    def extract_class_frames_indexes_list(self, class_name: str) -> List:
        """Returns indexes list"""
        indexes = []
        for frame in self.cls_and_conf_each_frame:
            indexes.append([idx for idx, value in enumerate(frame[0]) if value == class_name])
        return indexes

    def extract_class_frames_to_folder(self, indexes: List, class_name: str, without_det_boxes=False) -> (str, List):
        """Returns filepath to zip with images where this class is shown"""

        directory, file_name = os.path.split(self.filepath)
        file_name, extension = os.path.splitext(file_name)

        folder_path = os.path.join(directory, file_name + "_" + class_name + "_" + self.model_name)

        if without_det_boxes:
            folder_path = folder_path + "_" + "no_boxes"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        frames_indexes = []

        for i, frame in enumerate(zip(self.results, indexes)):
            # cv2_imshow(frame[idx].plot(probs=True, boxes=True))
            frame_idx_list = frame[1]
            if len(frame_idx_list):
                frame_name = os.path.join(folder_path, f"{i}.jpg")
                frames_indexes.append(i)
                if not without_det_boxes:
                    cv2.imwrite(frame_name, frame[0][frame_idx_list].plot(probs=True, boxes=True))

                else:
                    cv2.imwrite(frame_name, frame[0].plot(probs=False, boxes=False))

        return folder_path, frames_indexes

    def crop_class_img_by_coords_and_save_to_folder(self, image_path, coords):
        """
        Function to crop an image.
        :param coords: list of lists of int, coordinates (x1, y1, x2, y2)
        :return:
        """

        directory, file_name = os.path.split(image_path)
        file_name, extension = os.path.splitext(file_name)

        result_folder_path = os.path.join(directory, file_name + "_crop")
        if not os.path.exists(result_folder_path):
            os.makedirs(result_folder_path)

        # Load the image
        img = cv2.imread(image_path)

        # Check if image loading is successful
        if img is None:
            print(f"Failed to load image at {image_path}")
            return None
        cropped_img = []
        # Remember OpenCV uses y first (height) and then x (width)
        for idx, obj in enumerate(coords):
            obj_name = os.path.join(result_folder_path, f"{idx}.jpg")
            x1, y1, x2, y2 = [int(i) for i in obj]
            cropped_img.append(img[y1:y2, x1:x2])
            cv2.imwrite(obj_name, img[y1:y2, x1:x2])

        return cropped_img

    def zip_folder(self, folder_path):
        # Compress the folder to a zip file
        # zip_path = folder_path + ".zip"
        zip_path = folder_path
        shutil.make_archive(zip_path, 'zip', folder_path)

        return zip_path + ".zip"

    def get_frames_intervals(self, frames) -> str:
        intervals = []
        i = 0
        while i < len(frames):
            start = frames[i]
            end = start
            while i < len(frames) - 1 and frames[i + 1] == end + 1:
                end = frames[i + 1]
                i += 1
            i += 1
            if start == end:
                intervals.append(str(start))
            else:
                intervals.append(str(start) + '-' + str(end))
        return f"[{', '.join(intervals)}]"

    def classify_img(self, filepath, save_folder_path):
        if self.obj365_model is None:
            self.obj365_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5m_Objects365.pt')
        result = self.obj365_model(filepath,
                        #save=True,
                        #vid_stride=self.vid_stride,
                        # imgsz= max(width, height),
                        # imgsz=640,
                        # # save_txt=True,
                        # save_txt=False,
                        # save_conf=True,
                        )
        directory, file_name = os.path.split(filepath)
        file_name, extension = os.path.splitext(file_name)
        result_filepath = os.path.join(save_folder_path, file_name + extension)

        # id_cls_conf =
        #
        # classes_names_list = [[classes[cls] for cls in result.boxes.cls.cpu().numpy()] for result in self.results]
        # classes_idx_list = [result.boxes.cls.cpu().numpy() for result in self.results]
        # conf_list = [result.boxes.conf.cpu().numpy() for result in self.results]
        #
        # self.box_coords_list = [result.boxes.xyxy.cpu().numpy() for result in self.results]
        #
        # self.cls_and_conf_each_frame = list(zip(classes_names_list, conf_list, classes_idx_list))
        # print(self.cls_and_conf_each_frame)
        # self.unique_classes = self.get_unique_classes()
        # print(self.unique_classes)
        #
        # directory, file_name = os.path.split(self.filepath)
        # self.extract_result_file(directory)
        result.print()
        result.save()
        #result.show()
        # cv2.imshow('', result.ims[0][:, :, [2, 1, 0]])
        # cv2.waitKey(0)
        cv2.imwrite(result_filepath, result.ims[0][:, :, [2, 1, 0]])

    def classify_folder(self, folder_path):

        files_list = os.listdir(folder_path)
        files_list = [f"{folder_path}/{file}" for file in files_list]
        result_folder_path = folder_path + "_obj365"

        if not os.path.exists(result_folder_path):
            os.makedirs(result_folder_path)

        for file in files_list:
            self.classify_img(file, result_folder_path)

        return result_folder_path


if __name__ == "__main__":
    #detect = Detection(model_name='yolov8s.pt')
    detect = Detection(model_name='v8s_14cls.pt')
    # detect = Detection(model_name='yolov5m_Objects365.pt')
    detect.set_ppc_params(filepath="tonylife2.mp4", vid_stride=15, imgsz=32 * 10)
    #
    # time1 = datetime.datetime.now()
    detect.preprocess_video(compress_video=True)
    # time2 = datetime.datetime.now()
    # elapsedTime = time2 - time1
    #
    # print(f"Time taken: [{divmod(elapsedTime.total_seconds(), 60)}] min, sec")

    # detect.classify_img("tonylife2_person_no_det_boxes/0.jpg")

    detect.classify_folder("tonylife2_person_no_det_boxes")

    # #play_video(detect.result_filepath)
    # cls_name = "person"
    # indexes = detect.extract_class_frames_indexes_list(class_name=cls_name)
    # folder_path, found_frames_idx = detect.extract_class_frames_to_folder(indexes, class_name=cls_name,
    #                                                                       without_det_boxes = False)

    # detect.crop_class_img_by_coords_and_save_to_folder("tonylife2_person/0.jpg", detect.box_coords_list[0])

    # classification_model = YOLO('yolov8l-cls.pt')
    # result_cls = classification_model(source="tonylife2_person_no_det_boxes/6.jpg",
    #                                   save=True,
    #                                   # imgsz= max(width, height),
    #                                   # half = True,
    #                                   save_txt=False,
    #                                   save_conf=True,
    #                                   )
    #
    # cls_list = [(float(el.cpu()), result_cls[0].names[i]) for i, el in enumerate(result_cls[0].probs)]
    # cls_list.sort(key=lambda x: x[0], reverse=True)
    #
    # print("here")

# buffer = StringIO()
# sys.stdout = buffer
#
# detect = Detection(model_name='yolov8n.pt')
# detect.preprocess_video(filepath="tonylife2.mp4", vid_stride=10, imgsz=32*20)
# detect.preprocess_video(filepath="tonylife2.mp4", vid_stride=10, imgsz=32*15)
# detect.preprocess_video(filepath="tonylife2.mp4", vid_stride=10, imgsz=32*10)
# detect.preprocess_video(filepath="tonylife2.mp4", vid_stride=10, imgsz=32*8)
#
# detect = Detection(model_name='yolov8s.pt')
# detect.preprocess_video(filepath="tonylife2.mp4", vid_stride=10, imgsz=32*20)
# detect.preprocess_video(filepath="tonylife2.mp4", vid_stride=10, imgsz=32*15)
# detect.preprocess_video(filepath="tonylife2.mp4", vid_stride=10, imgsz=32*10)
# detect.preprocess_video(filepath="tonylife2.mp4", vid_stride=10, imgsz=32*8)
#
# detect = Detection(model_name='yolov8m.pt')
# detect.preprocess_video(filepath="tonylife2.mp4", vid_stride=10, imgsz=32*20)
# detect.preprocess_video(filepath="tonylife2.mp4", vid_stride=10, imgsz=32*15)
# detect.preprocess_video(filepath="tonylife2.mp4", vid_stride=10, imgsz=32*10)
# detect.preprocess_video(filepath="tonylife2.mp4", vid_stride=10, imgsz=32*8)
# #detect.show_result()
#
# print_output = buffer.getvalue()
# sys.stdout = sys.__stdout__
# print(f"->{print_output}<-")
# f = open("nsm_32x20-8_test_results.txt", "w")
# f.write(print_output)
# f.close()
