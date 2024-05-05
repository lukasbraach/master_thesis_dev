from lsfb_dataset import Downloader

downloader = Downloader(dataset='cont', destination="./dataset", landmarks=[], include_videos=True,
                        max_parallel_connections=2, skip_existing_files=True)
downloader.download()
