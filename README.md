# it's about time
Code repository for "It's About Time: Analog clock Reading in the Wild"

Packages required: 
`pytorch` (used 1.9, any reasonable version should work), `kornia` (for homography), `einops`, `scikit-learn` (for RANSAC), `tensorboardX` (for logging)

Using pretrained model:
- prediction `python predict.py` will predict on your data (or by default, whatever is in `data/demo`). This does assume the images being already cropped, we use CBNetv2. (you could instead add something like a yolov5 to the code if you prefer not installing anything extra).
- evaluation `python eval.py` (requires dataset) should return the numbers reported in the paper

Training:
- `sh full_cycle.sh` should do the job
- if you want to do it individually, then do use
  -  `train.py` train on SynClock
  -  `generate_pseudo_labels.py` use the model to generate pseudo labels for timelapse
  -  `train_refine.py` train on SynClock+timelapse. 
  -  The latter two can be repeated iteratively.

Dataset (Train):
- SynClock is generated on the fly (via `SynClock.py`)
- Timelapse will be uploaded later.

Dataset (Eval):
- COCO and OpenImages: The `.csv` files in `data/` contains the image ids, predicted bbox's (by CBNetV2), gt bbox's, and the manual time label. We will upload this subset later for convenience, but if you already have the respective datasets it should already work.
- Clock Movies do not contain bbox's. We may not be able to release the data directly due to copyright, but the csv files do contain the image file names, and they are scraped from https://theclock.fandom.com/wiki/Special:NewFiles

Note:
src/cyclic_ransac.py is adapted from the source code of scikit-learn (authored by Johannes Schönberger under BSD 3 clause license), to fit a sawtooth wave for cyclic linear data.

Dataset:
all are available to download from: https://drive.google.com/drive/folders/14FFDev1Omia_6E48Csw22kfbVAlW10hd?usp=sharing

Teaser video:
https://youtu.be/cbiMACA6dRc
