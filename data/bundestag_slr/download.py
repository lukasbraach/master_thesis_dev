from huggingface_hub import HfApi
from yt_dlp import YoutubeDL


api = HfApi()

future = api.upload_folder( # Upload in the background (non-blocking action)
    repo_id="lukasbraach/bundestag_slr",
    folder_path="checkpoints-001",
    run_as_future=True,
)


URLS = ['https://www.youtube.com/playlist?list=PLfRDp3S7rLdtIau6cObj_Q8PIjTWUXrhq']
ydl_opts = {
    "o": "%(id)s/video.%(ext)s",
    "format-sort": "vcodec",
    "f": "bestvideo[ext=mp4]+bestaudio",
    "merge-output-format": "mp4",
    "sub-langs": "all",
    "write-subs": True,
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download(URLS)