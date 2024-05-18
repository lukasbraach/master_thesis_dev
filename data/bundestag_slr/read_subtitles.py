from typing import List

import pysrt
from pysrt import SubRipFile, SubRipItem


def get_subtitle_items(subs: SubRipFile) -> List[SubRipItem]:
    items: List[SubRipItem] = []

    for sub in subs:
        sub: SubRipItem = sub

        sub.text = sub.text.replace('\n', ' ')
        items.append(sub)

    return items


if __name__ == "__main__":
    input_file = '7610784.srt'

    subs = pysrt.open(input_file)
    subs_items = get_subtitle_items(subs)

    for i, sub in enumerate(subs_items):
        start_ms = sub.start.ordinal
        end_ms = sub.end.ordinal

        print(f"{start_ms} --> {end_ms}: {sub.text}")
