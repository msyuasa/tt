import cv2
import numpy as np
from argparse import ArgumentParser
import glob

def main():
    parser = ArgumentParser()
    parser.add_argument("--filepath", dest="filepath", type=str)
    parser.add_argument("--dirpath", dest="dirpath", type=str)
    parser.add_argument("--save", dest="save", type=str, default=".")
    parser.add_argument("--img_type", dest="img_type", type=str, default="png")
    args = parser.parse_args()

    if args.filepath:
        make_contour_image(args.filepath, args.save)
    elif args.dirpath:
        files = glob.glob(args.dirpath + "/*." + args.img_type)
        for file in files:
            make_contour_image(file, args.save)
    else:
        raise TypeError("extract_line takes exactly 1 argument ('--filepath' or '--dirpath')")


def make_contour_image(path, save):
    kernel = np.ones((5,5), np.uint8)
    # グレースケールで画像を読み込む.
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # 白い部分を膨張させる.
    dilated = cv2.dilate(gray, kernel, iterations=1)

    # 差をとる.
    diff = cv2.absdiff(dilated, gray)

    # 白黒反転
    contour = 255 - diff

    #name = path.split["/"][-1].split["."][0]
    name = path.split("/")[-1].split(".")[0]
    cv2.imwrite("{0}/{1}_line.jpg".format(save, name), contour)
    return contour

def make_resize_image(path, save):
    img = cv2.imread(path)
    shpae = img.shape #heights, width, channel
    height = 256
    width = 256
    resize_img = cv2.resize(img, (width, height))

    #name = path.split["/"][-1].split["."][0]
    name = path.split("/")[-1].split(".")[0]
    cv2.imwrite("{0}/{1}_rs.jpg".format(save, name), resize_img)
    return resize_img

def concat_image(path1, path2, save):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    himg = cv2.hconcat([img1, img2])
    name = path1.split("/")[-1].split(".")[0]
    cv2.imwrite("{0}/{1}_h.jpg".format(save, name), himg)
    return himg

def cut_image(path, args):
    img = cv2.imread(path)
    height, width, channel = img.shape
    cut = 40
    # if args.img_type == jpg:
    #     cut_image = img[40:width+40,:,:]
    # else:
    #     cut_image = img[:width,:,:]
    cut_image = img[40:width+40,:,:] if args.img_type == "jpg" else img[:width,:,:]

    name = path.split("/")[-1].split(".")[0]
    cv2.imwrite("{0}/{1}_ct.jpg".format(save, name), cut_image)

if __name__ == "__main__":
    main()

# process   cut --> resize --> linearizle --> concatinate(color + linear) 
