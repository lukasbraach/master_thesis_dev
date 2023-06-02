from nvidia.dali.pipeline import Pipeline

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

image_dir = "/home/lbraach/datasets/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train/01April_2010_Thursday_heute_default-6/"
max_batch_size = 1


@pipeline_def
def simple_pipeline():
    jpegs, labels = fn.readers.file(file_root=image_dir)
    images = fn.decoders.image(jpegs, device='mixed')

    return images, labels


pipe = simple_pipeline(batch_size=max_batch_size, num_threads=1, device_id=0)

pipe.build()

pipe_out = pipe.run()
print(pipe_out)
