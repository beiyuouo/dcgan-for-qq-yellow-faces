import argparse
import os
from PIL import Image, ImageSequence

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='data/Extracted', help='folder to input images')
parser.add_argument('--output_path', default='data/Output', help='folder to output images')

opt = parser.parse_args()


def parseGIF(opt, gifname):
    im = Image.open(os.path.join(opt.input_path, gifname))
    iter = ImageSequence.Iterator(im)

    index = 1
    for frame in iter:
        # print("image %d: mode %s, size %s" % (index, frame.mode, frame.size))
        frame = frame.convert('RGB')
        frame.save(os.path.join(opt.output_path, '{}_{:2d}.jpg'.format(gifname, index)))
        index += 1

def parseJPG(opt, jpgname):
    im = Image.open(os.path.join(opt.input_path, jpgname))
    im = im.convert('RGB')
    im.save(os.path.join(opt.output_path, '{}.jpg'.format(jpgname)))

def parsePNG(opt, pngname):
    im = Image.open(os.path.join(opt.input_path, pngname))
    im = im.convert('RGB')
    im.save(os.path.join(opt.output_path, '{}.jpg'.format(pngname)))

if __name__ == "__main__":
    list = os.listdir(opt.input_path)
    for i in list:
        print(i)
        if i.endswith('GIF') or i.endswith('gif'):
            parseGIF(opt, i)
        elif i.endswith('JPG') or i.endswith('jpg'):
            parseJPG(opt, i)
        elif i.endswith('PNG') or i.endswith('png'):
            parseJPG(opt, i)
