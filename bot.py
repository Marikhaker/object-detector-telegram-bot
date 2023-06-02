import logging
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import os
import datetime
from dataclasses import dataclass
# version 13.7
# https://github.com/python-telegram-bot/v13.x-wiki/wiki
# https://docs.python-telegram-bot.org/en/v13.7/index.html
from object_detection import Detection, play_video, Metadata, check_gpu


class LocalUser:
    pass


# Replace TOKEN with your own Telegram bot token
TOKEN = ''

# Enable logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


@dataclass
class ModelParams:
    default_test_params = {
        "model_name": "yolov8s.pt",
        "imgsz": 32 * 8,
        "vid_stride": 10
    }

    fastest_params = {
        "model_name": "yolov8n.pt",
        "imgsz": 32 * 8,
        "vid_stride": 10
    }

    normal_working_params = {
        "model_name": "yolov8s.pt",
        "imgsz": 32 * 10,
        "vid_stride": 1
    }

    best_but_slow_params = {
        "model_name": "yolov8m.pt",
        "imgsz": 32 * 20,
        "vid_stride": 1
    }

    slow_test_params = {
        "model_name": "yolov8m.pt",
        "imgsz": 32 * 20,
        "vid_stride": 10
    }

    custom_trained = {
        "model_name": "v8s_14lcs_22ep.pt",
        "imgsz": 32 * 10,
        "vid_stride": 10
    }



def init_model(update: Update, context: CallbackContext):
    msg = update.message.reply_text(f'model_params: \n{context.user_data["local_model_params"]}')
    # msg.message_id
    context.user_data["detect"] = Detection(model_name=context.user_data["local_model_params"]["model_name"])
    update.message.reply_text(f'Upload your video', reply_markup=keyboard_start())
    context.user_data["menu"] = "init_model"


def keyboard_start():
    kb = [[KeyboardButton('change_params')], [KeyboardButton('cancel')], [KeyboardButton('available_classes')]]
    reply_markup = ReplyKeyboardMarkup(kb, one_time_keyboard=True, resize_keyboard=True)
    return reply_markup


def start(update: Update, context: CallbackContext):
    update.message.reply_text('I can detect objects\nFrom COCO dataset (80 classes) using YOLOv8',
                              reply_markup=keyboard_start())
    context.user_data["menu"] = "start"
    context.user_data["local_model_params"] = ModelParams.default_test_params
    if check_gpu():
        context.user_data["local_model_params"] = ModelParams.best_but_slow_params
    init_model(update, context)
    return


def sent_to_sleep(update: Update, context: CallbackContext):
    print("Pc go to sleep!")
    if str(update.message.from_user.id) == "802493197":
        print("User is Marikhaker")
        # os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    pass


def handle_change_params(update: Update, context: CallbackContext):
    update.message.reply_text('Choose one of presets: ')

    presets_str = f"\n```\n1. default_test_params{ModelParams.default_test_params}\n\n2. custom_trained{ModelParams.custom_trained}\n" \
                  f"\n3. fastest_params{ModelParams.fastest_params}\n\n4. normal_working_params{ModelParams.normal_working_params}\n" \
                  f"\n5. best_but_slow_params{ModelParams.best_but_slow_params}\n\n6. slow_test_params{ModelParams.slow_test_params}\n" \
                  f"```"

    kb = [[KeyboardButton('fastest')], [KeyboardButton('default_test')], [KeyboardButton('normal_working')],
          [KeyboardButton('best_but_slow')], [KeyboardButton('slow_test')], [KeyboardButton('custom_trained')], [KeyboardButton('cancel')]]

    reply_markup = ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=True)

    update.message.reply_text(presets_str, reply_markup=reply_markup, parse_mode="Markdown")
    context.user_data["menu"] = "change_params"

    return


def preset_change(update: Update, context: CallbackContext):
    print(update.message.text)

    presets = ["fastest", "default_test", "normal_working", "best_but_slow"]

    if update.message.text == "fastest":
        context.user_data["local_model_params"] = ModelParams.fastest_params
    if update.message.text == "default_test":
        context.user_data["local_model_params"] = ModelParams.default_test_params
    if update.message.text == "normal_working":
        context.user_data["local_model_params"] = ModelParams.normal_working_params
    if update.message.text == "best_but_slow":
        context.user_data["local_model_params"] = ModelParams.best_but_slow_params
    if update.message.text == "slow_test":
        context.user_data["local_model_params"] = ModelParams.slow_test_params
    if update.message.text == "custom_trained":
        context.user_data["local_model_params"] = ModelParams.custom_trained

    update.message.reply_text(f'Model changed')

    init_model(update, context)

    return


def create_userid_folder(update: Update, context: CallbackContext) -> str:
    user_id = update.message.from_user.id
    print(f"user_id = {user_id}")
    print(f"username = ", update.message.from_user.username)
    dir_path = "users/" + str(user_id)
    if not os.path.exists("users/"):
        os.mkdir("users/")
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    return dir_path


def calculate_processing_time(update: Update, context: CallbackContext):
    model_coef = {
        'yolov8n.pt': 1,
        'yolov8s.pt': 2.3,
        'yolov8m.pt': 2.3 * 2.3,
        'yolov8l.pt': 2.3 * 2.3 * 2.3
    }

    imgsz_speed_ms = {
        32 * 20: 140,
        32 * 15: 85,
        32 * 10: 40,
        32 * 8: 30
    }

    if context.user_data["detect"].model_name in model_coef.keys():
        speed_ms = imgsz_speed_ms[context.user_data["detect"].imgsz] * model_coef[
            context.user_data["detect"].model_name]
    else:
        speed_ms = imgsz_speed_ms[context.user_data["detect"].imgsz] * model_coef["yolov8s.pt"]
        print(f"each frame ~speed_ms = {speed_ms}")

    time_to_process_seconds = context.user_data["detect"].metadata.frames * speed_ms / 1000 / context.user_data[
        "detect"].vid_stride

    if context.user_data["detect"].gpu_available:
        time_to_process_seconds /= 10
    print(f"time_to_process_seconds = {time_to_process_seconds}")

    return float(time_to_process_seconds)


def show_available_classes_names(update: Update, context: CallbackContext):
    # print(context.user_data["detect"].get_classes_names())
    update.message.reply_text(f'Classes: \n```\n{context.user_data["detect"].get_classes_names()}```',
                              parse_mode="Markdown")
    return


def keyboard_select_class(update: Update, context: CallbackContext):
    kb = [[KeyboardButton('select_class')]]
    reply_markup = ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=True)
    update.message.reply_text(f"Select certain class / Upload next video", reply_markup=reply_markup)
    context.user_data["menu"] = "select_class"
    return reply_markup


def keyboard_show_frame(update: Update, context: CallbackContext):
    kb = [[KeyboardButton('select_class')], [KeyboardButton('show_frame')], [KeyboardButton('cancel')]]
    reply_markup = ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=True)
    update.message.reply_text(f"Show certain frame / Select certain class / Upload next video", reply_markup=reply_markup)
    return reply_markup


def handle_select_class(update: Update, context: CallbackContext):
    update.message.reply_text("Write a class name from unique classes above to extract zip" +
                              "with frames where only this class appears")
    context.user_data["menu"] = "input_class_name"
    return


def handle_show_frame(update: Update, context: CallbackContext):
    update.message.reply_text("Write frame index from intervals above:")
    context.user_data["menu"] = "show_frame"
    return


def handle_text_input(update: Update, context: CallbackContext):
    if context.user_data["menu"] == "input_class_name":
        text = update.message.text
        if text not in context.user_data['detect'].unique_classes[0]:
            update.message.reply_text(f"Wrong class name")
            return
        indexes = context.user_data["detect"].extract_class_frames_indexes_list(class_name=str(text))
        folder_path_no_boxes, _ = context.user_data["detect"].extract_class_frames_to_folder(indexes,
                                                                                          class_name=str(text),
                                                                                          without_det_boxes=True)
        zip_path_no_boxes = context.user_data["detect"].zip_folder(folder_path_no_boxes)

        folder_path, found_frames_idx = context.user_data["detect"].extract_class_frames_to_folder(indexes,
                                                                                                   class_name=str(text),
                                                                                                   without_det_boxes=False)
        zip_path = context.user_data["detect"].zip_folder(folder_path)

        context.user_data["found_frames_idx"] = found_frames_idx
        context.user_data["last_folder_path"] = folder_path

        with open(zip_path_no_boxes, 'rb') as file:
            context.bot.send_document(chat_id=update.message.chat_id, document=file)
        with open(zip_path, 'rb') as file:
            context.bot.send_document(chat_id=update.message.chat_id, document=file)
        update.message.reply_text(f"Found [{len(found_frames_idx)}] frames with object:" +
                                  f"\n```\n{context.user_data['detect'].get_frames_intervals(found_frames_idx)}```",
                                  parse_mode="Markdown")
        keyboard_show_frame(update, context)

    if context.user_data["menu"] == "show_frame":
        text = int(update.message.text)
        if text not in context.user_data["found_frames_idx"]:
            update.message.reply_text(f"Wrong frame index")
            return
        frame_path = context.user_data["last_folder_path"] + "/" + str(text) + ".jpg"
        with open(frame_path, 'rb') as file:
            context.bot.send_photo(chat_id=update.message.chat_id, photo=file)

    return


def preprocess_video_and_upload(update: Update, context: CallbackContext):
    first_time = datetime.datetime.now()
    print(f"first_time={first_time}")

    context.user_data["detect"].preprocess_video(compress_video=True)

    pps_video_filepath = context.user_data["detect"].result_filepath

    later_time = datetime.datetime.now()
    print(f"later_time={later_time}")
    time_difference = later_time - first_time

    print(f"Real time taken in seconds = {time_difference.seconds}")
    update.message.reply_text(f"Real time taken: {time_difference.seconds} sec")

    # local_file_path = f'./{video.file_id}_preprocessed.mp4'
    # local_file_path = 'Tonylife2_preprocessed.mp4'

    with open(pps_video_filepath, 'rb') as file:
        context.bot.send_video(chat_id=update.message.chat_id, video=file)


    local_unique_classes = context.user_data['detect'].unique_classes[1]
    update.message.reply_text(f"Unique classes times found:\n```\n{local_unique_classes}```", parse_mode="Markdown")

    keyboard_select_class(update, context)
    # play_video(context.user_data["detect"].result_filepath)


def handle_video(update: Update, context: CallbackContext):
    if update.message.video is not None:
        video = update.message.video
        print("video")
    else:
        if update.message.document.mime_type == "video/mp4":
            print("mp4 file or gif")
            video = update.message.document
        else:
            print("Not video!")
            return

    video_file = video.get_file()

    local_file_path = create_userid_folder(update, context) + "/"

    local_file_path += f'{video.file_unique_id}.mp4'
    context.user_data["local_file_path"] = local_file_path
    video_file.download(context.user_data["local_file_path"])

    local_metadata = Metadata(context.user_data["local_file_path"])

    ###########

    # https://stackoverflow.com/questions/1345827/how-do-i-find-the-time-difference-between-two-datetime-objects-in-python
    print("Sending metadata")
    update.message.reply_text(local_metadata.get_metadata_str())

    context.user_data["detect"].set_ppc_params(filepath=context.user_data["local_file_path"],
                                               imgsz=context.user_data["local_model_params"]["imgsz"],
                                               vid_stride=context.user_data["local_model_params"]["vid_stride"])

    update.message.reply_text(f"Approx time to process: ~{calculate_processing_time(update, context):.3f} sec")

    detect_menu_keyboard(update, context)

    return


def handle_cancel(update: Update, context: CallbackContext):
    update.message.reply_text('Returned to start menu', reply_markup=keyboard_start())
    print("Returned start menu\nUpload your video")
    context.user_data["menu"] = "cancel"
    return
    #return start(update, context)


def detect_menu_keyboard(update: Update, context: CallbackContext):
    kb = [[KeyboardButton('detect')], [KeyboardButton('change_params')]]
    reply_markup = ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=True)
    update.message.reply_text("Detect / change_params / upload another video", reply_markup=reply_markup)


def main():
    updater = Updater(TOKEN, use_context=True)

    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('change_params', handle_change_params))
    dp.add_handler(CommandHandler('sent_to_sleep', sent_to_sleep))
    dp.add_handler(MessageHandler(Filters.regex('^change_params'), handle_change_params))
    dp.add_handler(MessageHandler(Filters.regex('^fastest'), preset_change))
    dp.add_handler(MessageHandler(Filters.regex('^default_test'), preset_change))
    dp.add_handler(MessageHandler(Filters.regex('^normal_working'), preset_change))
    dp.add_handler(MessageHandler(Filters.regex('^best_but_slow'), preset_change))
    dp.add_handler(MessageHandler(Filters.regex('^slow_test'), preset_change))
    dp.add_handler(MessageHandler(Filters.regex('^custom_trained'), preset_change))

    dp.add_handler(MessageHandler(Filters.regex('^detect'), preprocess_video_and_upload))
    #dp.add_handler(MessageHandler(Filters.regex('^detect_obj365'), preprocess_video_and_upload))
    dp.add_handler(MessageHandler(Filters.regex('^select_class'), handle_select_class))
    dp.add_handler(MessageHandler(Filters.regex('^show_frame'), handle_show_frame))
    # dp.add_handler(MessageHandler(Filters.regex('^upload_next'), preprocess_video_and_upload))

    dp.add_handler(MessageHandler(Filters.regex('^available_classes'), show_available_classes_names))
    dp.add_handler(MessageHandler(Filters.video, handle_video))
    dp.add_handler(MessageHandler(Filters.document, handle_video))
    dp.add_handler(MessageHandler(Filters.animation, handle_video))
    dp.add_handler(MessageHandler(Filters.regex('^cancel$'), handle_cancel))
    dp.add_handler(MessageHandler(Filters.text, handle_text_input))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
