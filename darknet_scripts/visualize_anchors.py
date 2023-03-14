import cv2
import numpy as np
import sys
import argparse
from os import listdir
from os.path import isfile, join


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-anchor_dir', default = 'generated_anchors/voc-anchors', 
                        help='path to anchors\n', ) 
    
    args = parser.parse_args()
    
    
    print "anchors list you provided{}".format(args.anchor_dir)

    [H,W] = (416,416)
    stride = 32
    
    cv2.namedWindow('Image')
    cv2.moveWindow('Image',100,100)

    colors = [(255,0,0),(255,255,0),(0,255,0),(0,0,255),(0,255,255),(55,0,0),(255,55,0),(0,55,0),(0,0,25),(0,255,55)]

    anchor_files = [f for f in listdir(args.anchor_dir) if (join(args.anchor_dir, f)).endswith('.txt')]
    for anchor_file in anchor_files:
        blank_image = np.zeros((H,W,3),np.uint8)
        

        f = open(join(args.anchor_dir,anchor_file))
        line = f.readline().rstrip('\n')

        anchors = line.split(', ')

        filename = join(args.anchor_dir,anchor_file).replace('.txt','.png')
        
        print filename

        stride_h = 10
        stride_w = 3
        if 'caltech' in filename:
            stride_w = 25
            stride_h = 10

        for i in range(len(anchors)):
            (w,h) = map(float,anchors[i].split(','))


            w=int (w*stride)
            h=int(h*stride)
            print w,h
            offset_x = 10+i*stride_w # this offset is just to make sure starting coordinates of anchors do not overlap each other
            offset_y = 10+i*stride_h
            
            cv2.rectangle(blank_image,(offset_x,offset_y),(offset_x+w,offset_y+h),colors[i])

            #cv2.imshow('Image',blank_image)

             
            cv2.imwrite(filename,blank_image)
            #cv2.waitKey(10000)

if __name__=="__main__":
    main(sys.argv)
