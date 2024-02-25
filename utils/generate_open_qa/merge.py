import json
import glob
import numpy as np


split = 'EgoTimeQA'
src1 = f'tmp/annotations.{split}_*.json'
tgt = f'annotations.{split}.json'
paths = glob.glob(src1)
paths = sorted(paths)
print('Merging')    

merge = []
for p in paths:
    print(p)
    x = json.load(open(p))
    merge += x

all_duration_sec = [(x['moment_end_frame'] - x['moment_start_frame']) / 30 for x in merge]
mean_duration_sec = np.asarray(all_duration_sec).mean()
# normalize duration_sec
for x in merge:
    start_sec = x['moment_start_frame'] / 30
    end_sec = x['moment_end_frame'] / 30
    center_sec = (start_sec + end_sec) / 2
    duration_sec = (end_sec - start_sec) / mean_duration_sec
    x['moment_start_frame'] = (center_sec - duration_sec / 2) * 30
    x['moment_end_frame'] = (center_sec + duration_sec / 2) * 30

print(f'into {tgt}')
with open(tgt, 'w') as f:
    json.dump(merge, f)

print(len(merge))
