import torch
import imageio
import cv2
from PIL import Image
from moviepy.editor import VideoFileClip
from torchvision import transforms as T

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

# gif -> (channels, frame, height, width) tensor
def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

def crop_video_frames(input_path, left_margin, right_margin, top_margin, bottom_margin):
    video_clip = VideoFileClip(input_path)
    cropped_frames = []

    for frame in video_clip.iter_frames(fps=30):
        cropped_frame = frame[top_margin:-bottom_margin, left_margin:-right_margin]
        cropped_frames.append(cropped_frame)

    video_clip.reader.close()
    return cropped_frames

def resize_frames(frames, new_width, new_height):
    resized_frames = []

    for frame in frames:
        resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_frames.append(resized_frame)

    return resized_frames

def reduce_frames(frames, target_duration, target_frame_count):
    if target_frame_count <= 0:
        raise ValueError("Target frame count must be a positive integer")

    if target_duration <= 0:
        raise ValueError("Target duration must be a positive number")

    frame_count = len(frames)
    frame_rate = frame_count / target_duration
    interval = frame_count / target_frame_count

    reduced_frames = []
    for i in range(target_frame_count):
        index = int(i * interval)
        reduced_frames.append(frames[index])

    return reduced_frames

# Create gif with frames
def create_gif(frames, output_gif_path, frame_duration=0.1, frame_resize_factor=0.5):
    resized_frames = [Image.fromarray(frame).resize(
        (int(frame.shape[1] * frame_resize_factor), int(frame.shape[0] * frame_resize_factor)))
        for frame in frames]

    imageio.mimsave(output_gif_path, resized_frames, duration=frame_duration)

def videoToGIF(video_path, output_gif_path ):
  video_reader = imageio.get_reader(video_path)
  frames = []

  for frame in video_reader:
      frames.append(frame)

  # Adjust these parameters as needed
  frame_duration = 0.05  # Controls the speed of the GIF
  frame_resize_factor = 0.5  # Resize frames to 50% of the original dimensions

  resized_frames = [Image.fromarray(frame).resize(
      (int(frame.shape[1] * frame_resize_factor), int(frame.shape[0] * frame_resize_factor)))
      for frame in frames]

  imageio.mimsave(output_gif_path, resized_frames, duration=frame_duration)
  