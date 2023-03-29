import os
import argparse

from PIL import Image
import numpy as np

import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator
import cv2
import uuid
from subprocess import call

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def jj(*args):
    return os.path.join(*args)


def post_precess(img, wh):
    img = (img.squeeze() + 1.) / 2 * 255
    img = img.astype(np.uint8)
    img = cv2.resize(img, (wh[0], wh[1]))
    return img


def save(frame, path, name):
    img_name = name + '.png'
    img = Image.fromarray(frame.astype('uint8'))  # convert image to uint8:  frame.astype('uint8')
    img.save(path + img_name)


def load_image(image_path, x32=False):
    img = Image.open(image_path).convert("RGB")

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img


def to_32s(x):
    return 256 if x < 256 else x - x % 32

def x32(img, x32=False):
    img = img.convert("RGB")

    if x32:
        # def to_32s(x):
        #     return 256 if x < 256 else x - x % 32
        w, h = img.size
        img = img.resize((to_32s(w), to_32s(h)))

    return img


def img2Comix(args,net,device,image):
    #device = args.device
    
    #net = Generator()
    #net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    #net.to(device).eval()
    #print(f"model loaded: {args.checkpoint}")
    #os.makedirs(args.output_dir, exist_ok=True)


    #image = load_image(os.path.join(args.input_dir, image_name), args.x32)
    image = x32(image,args.x32)

    with torch.no_grad():
        image = to_tensor(image).unsqueeze(0) * 2 - 1
        out = net(image.to(device), args.upsample_align).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        out = to_pil_image(out)
    return out



def video2comix(args):
    device = args.device

    net = Generator()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.to(device).eval()
    print(f"model loaded: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)



    inputmp4 = jj(args.input_dir, 'f1_short.mp4')
    outputmp4 = jj(args.output_dir, 'f1_short_comi.mp4')
    outputh264 = jj(args.output_dir, 'f1_short_comi_h264.mp4')

    vc = cv2.VideoCapture(inputmp4)
    print('video open status:', vc.isOpened())

    # 获取视频宽度
    frame_width = int(cv2.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 获取视频高度
    frame_height = int(cv2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('video w,h:', str(frame_width), ',', str(frame_height))
    # fps
    fps = vc.get(cv2.CAP_PROP_FPS)
    print('video fps:', str(fps))
    # target coim fps
    fps_flow = args.fps  # fps / 2
    print('video fps:', str(fps_flow))

    # MPEG can not play in web
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    wh = (frame_width, frame_height)
    #宽高32位格式化，不知道是不是会提高效率?
    if x32:
        wh = (to_32s(frame_width), to_32s(frame_height))
    flow_video = cv2.VideoWriter(outputmp4, fourcc, fps_flow, wh)
    i = 0
    step = fps / fps_flow #根据fps算要跳几帧，比如：24fpx/5fpx=4，即每4帧读取下，1秒大概读取4~5帧
    n = 0 #帧数
    imgname = uuid.uuid4().hex
    while vc.isOpened():
        vc.set(cv2.CAP_PROP_POS_FRAMES, n)  # 截取指定帧数
        n = n + step #跳帧
        su, frame = vc.read()
        if frame is None:
            print('video2comix - read frame is none')
            break
        if su:
            # transform frame 2 comix
            # 风格化处理
            style_frame = img2Comix(args,net,device,frame)
            save(frame, args.output_dir, imgname + str(i))
            if i == 0:
                save(style_frame, args.output_dir, imgname + str(i) + "_c")

            img = style_frame.astype('uint8')

            flow_video.write(img)
            # flow_video.write(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            # cv2.waitKey(1)
            i = i + 1
            if i > 50:
                break
    print('video saved.')
    call(["ffmpeg", "-i", outputmp4, "-vcodec", "libx264", "-f", "mp4", outputh264])
    print('finish')
    flow_video.release()
    vc.release()
    return 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./weights/paprika.pt',
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='./samples/inputs',
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./samples/results',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
    )
    parser.add_argument(
        '--upsample_align',
        type=bool,
        default=False,
        help="Align corners in decoder upsampling layers"
    )
    parser.add_argument(
        '--x32',
        action="store_true",
        help="Resize images to multiple of 32"
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=5,
        help="fps >0 and < 20"
    )

    args = parser.parse_args()
    
    video2comix(args)
