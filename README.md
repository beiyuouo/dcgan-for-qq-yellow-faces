# Intro
This repo is for generating QQ yellow faces using DCGAN.

# How to run
1. `python run.py --epoch=10 --lr=0.0002 --grid_w=16 --grid_h=4`
`grid_w` and `grid_h` is for result showing.

# How to use your own dataset
1. Put your `gif/jpg/png` images in Extracted folder.
2. run `python data_preprocessing.py`, `png` will convert to `jpg`, `gif` will split by frames and save as `jpg`. 
