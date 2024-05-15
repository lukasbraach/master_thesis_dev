# BUNDESTAG SLR - Discover Video IDs
#
# by running curl 'https://www.bundestag.de/ajax/filterlist/gebaerdensprache/440438-440438?limit=70&offset=70&mediaCategory=440432%23Geb%C3%A4rdensprache-Plenarsitzungen&noFilterSet=false'
# and adapting the offset all video IDs can be found. The limit of 70 is the upper bound.
# Once there are no more results, the HTML contains the string "Leider keine Ergebnisse gefunden!".
# Python regex (?<=\/gebaerdensprache\?videoid=)[^#]+ finds all video IDs within the HTML.

import requests
import re
import csv


def fetch_video_ids(base_url, limit, max_offset):
    video_ids = []
    offset = 0

    while True:
        url = f"{base_url}&limit={limit}&offset={offset}"
        print(f"Fetching video IDs from {url}")

        response = requests.get(url)
        html = response.text

        # Check if no more results
        if "Leider keine Ergebnisse gefunden!" in html:
            break

        # Extract video IDs using regex
        ids = re.findall(r'(?<=\/gebaerdensprache\?videoid=)[^#]+', html)
        if not ids:
            break

        video_ids.extend(ids)
        offset += limit

        # Break loop if max_offset is reached
        if offset >= max_offset:
            break

    return video_ids


def save_video_ids_to_csv(video_ids, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['VideoID'])
        for video_id in video_ids:
            writer.writerow([video_id])


if __name__ == "__main__":
    base_url = "https://www.bundestag.de/ajax/filterlist/gebaerdensprache/440438-440438?mediaCategory=440432%23Geb%C3%A4rdensprache-Plenarsitzungen&noFilterSet=false"
    limit = 70
    max_offset = 3000  # Adjust this value as needed to cover the full range
    video_ids = fetch_video_ids(base_url, limit, max_offset)
    save_video_ids_to_csv(video_ids, 'video_ids.csv')

    print()
    print(f"Saved {len(video_ids)} video IDs to video_ids.csv")
