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
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_download_links(video_id):
    url = f"https://webtv.bundestag.de/player/macros/_x_s-144277506/shareData.json?contentId={video_id}"
    response = requests.get(url)
    data = response.json()

    download_urls = {
        'medium': data.get('downloadUrlMedium', ''),
        'high': data.get('downloadUrl', ''),
        'srt': data.get('downloadUrlSRT', '')
    }

    return download_urls


def download_file(url, dest_path):
    headers = {}
    existing_file_size = 0

    # Check if the file already exists
    if os.path.exists(dest_path):
        existing_file_size = os.path.getsize(dest_path)
        headers['Range'] = f'bytes={existing_file_size}-'

    # Make the HTTP request with range headers if applicable
    response = requests.get(url, headers=headers, stream=True)

    # Get the total file size from response headers, if available
    total_size = int(response.headers.get('content-length', 0)) + existing_file_size

    if response.status_code == 416:
        # Allow 416 Range Not Satisfiable responses (already fully downloaded)
        print(f"File already fully downloaded: {dest_path}")
        return

    # Check if the response indicates a client or server error
    if response.status_code >= 400:
        raise Exception(f"Failed to download file. HTTP status code: {response.status_code}")

    # Check for small video files based on content-type and size
    if response.headers.get("content-type") == 'video/mp4' and total_size < 5_000_000:
        raise Exception(f"Stopping download. File size too small: {total_size} (less than 5 MB)")

    # Open the file in append mode if resuming, otherwise in write mode
    mode = 'ab' if 'Range' in headers else 'wb'

    # Download the file in chunks and update the progress bar
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


def download_individual_id(video_id, dest_dir):
    print(f"Processing video ID: {video_id}")

    try:
        download_urls = get_download_links(video_id)

        video_path = os.path.join(dest_dir, f"{video_id}.mp4")
        srt_path = os.path.join(dest_dir, f"{video_id}.srt")

        video_url = download_urls['medium'] or download_urls['high']
        srt_url = download_urls['srt']

        if not video_url or not srt_url:
            print(f"Skipping video ID {video_id} due to missing video or subtitles.")
            return

        print(f"Downloading subtitles to {srt_path}")
        download_file(srt_url, srt_path)

        try:
            if download_urls['medium']:
                print(f"Downloading medium quality video to {video_path}")
                download_file(download_urls['medium'], video_path)
        except Exception as e:
            print(f"Downloading medium quality video failed. Falling back to high quality.")

            if download_urls['high']:
                print(f"Downloading high quality video to {video_path}")
                download_file(download_urls['high'], video_path)

    except Exception as e:
        print(f"Failed to process video ID {video_id}: {e}")

        # Clean up partially downloaded files
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(srt_path):
            os.remove(srt_path)


def download_videos_from_csv(csv_file, dest_dir, max_workers=4):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        video_ids = [row['VideoID'] for row in reader]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_individual_id, video_id, dest_dir) for video_id in video_ids]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Exception occurred: {e}")


if __name__ == "__main__":
    csv_file = 'video_ids.csv'  # Path to your CSV file with video IDs
    dest_dir = 'raw'  # Directory to save downloaded videos and subtitles
    max_workers = 4  # Number of threads to use for downloading
    download_videos_from_csv(csv_file, dest_dir, max_workers)
    print("Download process completed.")
