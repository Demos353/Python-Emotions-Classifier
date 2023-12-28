import cv2
import os

def convert_images_to_video(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Sort images in ascending order

    if len(images) == 0:
        print("No images found in the specified folder.")
        return

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    video_path = os.path.join(image_folder, video_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Choose the video codec (e.g., mp4v, XVID, etc.)
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Example usage
image_folder = "D://Time/meow"
video_name = "30.mp4"
fps = 30

convert_images_to_video(image_folder, video_name, fps)