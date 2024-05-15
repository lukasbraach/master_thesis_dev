# BUNDESTAG SLR content can be found by video ID
# on https://dbtg.tv/cvid/7605085 where 7605085 is the video ID
#
# by running curl https://webtv.bundestag.de/player/macros/_x_s-144277506/shareData.json?contentId=7605085
# all download links can be found. The JSON document has the following format:
#
# {
#     "audioUrlMono": "",
#     "audioUrlStereo": "https://cldf-od.r53.cdn.tv1.eu/1000153copo/ondemand/app144277506/145293313/7605085/7605085_mp3_128kb_stereo_de_128.mp3?fdl=1",
#     "downloadUrl": "https://cldf-od.r53.cdn.tv1.eu/1000153copo/ondemand/app144277506/145293313/7605085/7605085_h264_1920_1080_5000kb_baseline_de_5000.mp4?fdl=1",
#     "downloadUrlMedium": "https://cldf-od.r53.cdn.tv1.eu/1000153copo/ondemand/app144277506/145293313/7605085/7605085_h264_720_400_2000kb_baseline_de_2192.mp4?fdl=1",
#     "downloadUrlLow": "https://cldf-od.r53.cdn.tv1.eu/1000153copo/ondemand/app144277506/145293313/7605085/7605085_h264_512_288_514kb_baseline_de_514.mp4?fdl=1",
#     "downloadUrlSRT": "https://cldf-od.r53.cdn.tv1.eu/1000153copo/ondemand/app144277506/145293313/7605085/7605085.srt",
#     "rssRubric": "https://webtv.bundestag.de/player/macros/bttv/podcast/video/gebaerdensprache_plenarsitzungen.xml",
#     "itunesRubric": "itpc://webtv.bundestag.de/player/macros/bttv/podcast/audio/gebaerdensprache_plenarsitzungen.xml",
#     "rubricName": "Gebärdensprache-Plenarsitzungen",
#     "itunes": "itpc://webtv.bundestag.de/player/macros/bttv/podcast/video/auswahl.xml?contentIds=7605085",
#     "terms_de": "https://www.bundestag.de/resource/blob/296016/b2b8e3ed04b91bbfb235cfed975f1a69/nutzungsbedingungen_de-data.pdf",
#     "terms_en": "https://www.bundestag.de/resource/blob/296018/062266394066e7bc6a1a1a92f9d3358e/nutzungsbedingungen_en-data.pdf",
#     "shareDisabled": false,
#     "embedDisabled": false,
#     "status": {
#         "code": 1,
#         "message": "ok"
#     }
# }

import requests
import csv
import os
from tqdm import tqdm


def get_download_links(video_id):
    url = f"https://webtv.bundestag.de/player/macros/_x_s-144277506/shareData.json?contentId={video_id}"
    response = requests.get(url)
    data = response.json()
    return data['downloadUrl'], data['downloadUrlSRT']


def download_file(url, dest_path):
    headers = {}
    existing_file_size = 0

    if os.path.exists(dest_path):
        existing_file_size = os.path.getsize(dest_path)
        headers['Range'] = f'bytes={existing_file_size}-'

    response = requests.get(url, headers=headers, stream=True)
    total_size = int(response.headers.get('content-length', 0)) + existing_file_size

    mode = 'ab' if 'Range' in headers else 'wb'

    with open(dest_path, mode) as file, tqdm(
            desc=dest_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            initial=existing_file_size,
            ncols=80
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))


def download_videos_from_csv(csv_file, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_id = row['VideoID']
            print(f"Processing video ID: {video_id}")

            try:
                video_url, srt_url = get_download_links(video_id)
                video_path = os.path.join(dest_dir, f"{video_id}.mp4")
                srt_path = os.path.join(dest_dir, f"{video_id}.srt")

                print(f"Downloading video to {video_path}")
                download_file(video_url, video_path)

                print(f"Downloading subtitles to {srt_path}")
                download_file(srt_url, srt_path)
            except Exception as e:
                print(f"Failed to process video ID {video_id}: {e}")

            break


if __name__ == "__main__":
    csv_file = 'video_ids.csv'  # Path to your CSV file with video IDs
    dest_dir = 'raw'  # Directory to save downloaded videos and subtitles
    download_videos_from_csv(csv_file, dest_dir)
    print("Download process completed.")
