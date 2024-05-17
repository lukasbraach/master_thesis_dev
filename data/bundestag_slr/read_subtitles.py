import pysrt


def merge_subtitles(subs):
    merged_subs = pysrt.SubRipFile()
    current_text = ""
    start_time = None

    for sub in subs:
        if not start_time:
            start_time = sub.start

        current_text += " " + sub.text

        if sub.text.endswith(('.', '!', '?')):
            end_time = sub.end
            merged_subs.append(
                pysrt.SubRipItem(index=len(merged_subs) + 1, start=start_time, end=end_time, text=current_text.strip()))
            current_text = ""
            start_time = None

    # Add any remaining text as a new subtitle item
    if current_text:
        merged_subs.append(
            pysrt.SubRipItem(index=len(merged_subs) + 1, start=start_time, end=subs[-1].end, text=current_text.strip()))

    return merged_subs


if __name__ == "__main__":
    input_file = '7610784.srt'
    output_file = 'output.srt'

    subs = pysrt.open(input_file)
    merged_subs = merge_subtitles(subs)
    merged_subs.save(output_file, encoding='utf-8')
    print(f"Merged subtitles have been written to {output_file}")
