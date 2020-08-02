import argparse
import os
from PIL import Image, ImageSequence

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', default='result', help='folder to input images')
parser.add_argument('--output_path', default='result', help='folder to output images')

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
    return im
    im.save(os.path.join(opt.output_path, '{}.jpg'.format(jpgname)))


if __name__ == "__main__":
    list = os.listdir(opt.input_path)
    frames = []
    for i in list:
        print(i)
        if i.endswith('JPG') or i.endswith('jpg'):
            frames.append(parseJPG(opt, i))
    print(len(frames))
    frames[0].save(os.path.join(opt.output_path, 'out.gif'), save_all=True, append_images=frames[1:])
