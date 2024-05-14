import yt_dlp
import mediapipe as mp
import cv2
import os


# BUNDESTAG SLR content can be found by video ID
# on https://dbtg.tv/cvid/7605085 where 7605085 is the video ID
#
# by running curl https://webtv.bundestag.de/player/macros/_x_s-144277506/shareData.json?contentId=7605085
# all download links can be found.


def download_playlist(playlist_url, download_path='./downloads'):
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio",
        'outtmpl': f'{download_path}/%(title)s.%(ext)s',
        'noplaylist': False,
        "format-sort": "vcodec",
        "merge-output-format": "mp4",
        "sub-langs": "all",
        "write-subs": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([playlist_url])


def process_videos(directory):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    for filename in os.listdir(directory):
        if filename.endswith((".mp4", ".avi", ".mov")):  # Add other video formats if needed
            video_path = os.path.join(directory, filename)
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                # Convert the BGR image to RGB, flip the image for later selfie-view display, and process with MediaPipe Pose.
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                # Draw pose landmarks on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp.solutions.drawing_utils.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Show the image
                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()
    cv2.destroyAllWindows()


# Example usage:
playlist_url = 'https://www.youtube.com/playlist?list=PLfRDp3S7rLdtIau6cObj_Q8PIjTWUXrhq'
download_playlist(playlist_url)
process_videos('./downloads')

api = HfApi()

future = api.upload_folder(  # Upload in the background (non-blocking action)
    repo_id="lukasbraach/bundestag_slr",
    folder_path="checkpoints-001",
    run_as_future=True,
)
