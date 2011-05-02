#!/usr/bin/python
import cv
import sys
from optparse import OptionParser


# Creates a x/y velocity field on the interval [0,1]
def createVelocityField(velx , vely, delta):
    new_velx = cv.CreateMat(velx.rows, velx.cols, cv.CV_32FC1)
    new_vely = cv.CreateMat(vely.rows, vely.cols, cv.CV_32FC1)
    for x in range (0, velx.cols):
        for y in range (0, velx.rows):    
            new_velx[y,x] = velx[y,x] * delta
            new_vely[y,x] = vely[y,x] * delta

    return (new_velx, new_vely)


#Apply a simple median Filter to each component of the velocity field
#This function is really slow !!
def filterVelocityField(velx,vely,kernel_size=3):
    new_velx = cv.CreateMat(velx.rows, velx.cols, cv.CV_32FC1)
    new_vely = cv.CreateMat(vely.rows, vely.cols, cv.CV_32FC1)
    
    for x in range (kernel_size//2, velx.cols-kernel_size//2):
        for y in range (kernel_size//2, velx.rows-kernel_size//2):    
            values = []
            for dx in range(kernel_size):
                for dy in range(kernel_size):
                     values.append(velx[y-kernel_size//2+dy,x-kernel_size//2+dx])
            values.sort()
            new_velx[y,x] = values[len(values)//2]            

    for x in range (kernel_size//2, vely.cols-kernel_size//2):
        for y in range (kernel_size//2, vely.rows-kernel_size//2):    
            values = []
            for dx in range(kernel_size):
                for dy in range(kernel_size):
                     values.append(vely[y-kernel_size//2+dy,x-kernel_size//2+dx])
            values.sort()
            new_vely[y,x] = values[len(values)//2]            

    return (new_velx,new_vely)


def parseCommandLine(argv):
    parser = OptionParser()
    parser.add_option("-l","--left", action="store", type="string",
            dest="left", help="left image", default = None)
    parser.add_option("-r","--right", action="store", type="string", 
            dest="right", help="right image", default = None)
    parser.add_option("-a","--autoextract", action="store_true", 
            dest="xextract", help="automatically split the left image into two images",
            default=False)
    parser.add_option("-A","--autoextracty", action="store_true", 
            dest="yextract", help="automatically split the left image into two images (vertically)",
            default=False)
    parser.add_option("-m","--maxrange", action="store", type="string", 
            dest="maxrange", help="maximum search range (x,y)",
            default = "32,32")
    parser.add_option("-b","--blocksize", action="store", type="string", 
            dest="blocksize", help="search block size (w,h)",
            default = "32,32")
    parser.add_option("-s","--shiftsize", action="store", type="string", 
            dest="shiftsize", help="distance between search windows (x,y)",
            default = "10,10")
    parser.add_option("-f","--filtersize", action="store", type="int", 
            dest="filtersize", help="size of the median filter",
            default = "0")
    parser.add_option("-c","--crop", action="store", type="string", 
            dest="crop", help="crop the image (x,y,w,h)",
            default = "-1,-1,-1,-1")
    parser.add_option("-p","--imageshift", action="store", type="string", 
            dest="imageshift", help="shift the right image (x,y)",
            default = "0,0")
    parser.add_option("-S","--steps", action="store", type="int", 
            dest="steps", help="number of interpolation steps",
            default = 20)
    parser.add_option("-q","--headless", action="store", type="int", 
            dest="headless", help="without graphical output",
            default = "0")
    parser.add_option("-C","--autocrop", action="store", type="int", 
            dest="autocrop", help="automatically crop the image",
            default = "0")
    parser.add_option("-F","--composite_file", action="store", type="string",
            dest="composite_file", help="write a pikupiku composite file",
            default = None)
    parser.add_option("-P","--subpixel", action="store", type="int",
            dest="subpixel", help="use subpixel blocks",
            default = 0)
    parser.add_option("-G","--smooth", action="store", type="int",
            dest="smooth", help="Gaussian smoothing",
            default = 0)


    (options, args) = parser.parse_args(argv)

    arguments = {}
    arguments["headless"] = options.headless
    arguments["steps"] = options.steps
    arguments["left"] = options.left
    arguments["right"] = options.right
    arguments["autocrop"] = options.autocrop
    arguments["extract"] = 0
    if options.xextract:
        arguments["extract"] = 1
    elif options.yextract:
        arguments["extract"] = 2

    cfields=options.crop.split(",")
    arguments["cropx"] = int(cfields[0])
    arguments["cropy"] = int(cfields[1])
    arguments["cropw"] = int(cfields[2])
    arguments["croph"] = int(cfields[3])

    arguments["smooth"] = options.smooth

    arguments["subpixel"] = False
    if options.subpixel == 1:
        arguments["subpixel"] = True
            
    pfields = options.imageshift.split(",")
    arguments["imageshiftx"] = int(pfields[0])
    arguments["imageshifty"] = int(pfields[1])


    mfields = options.maxrange.split(",")
    arguments["maxrange"] = (int(mfields[0]), int(mfields[1]))

    bfields = options.blocksize.split(",")
    arguments["blocksize"] = (int(bfields[0]), int(bfields[1]))


    sfields = options.shiftsize.split(",")
    arguments["shiftsize"] = (int(sfields[0]), int(sfields[1]))

    arguments["filtersize"] = options.filtersize
    arguments["composite_file"] = options.composite_file

    return arguments
    

if __name__=="__main__":
    cmdline = parseCommandLine(sys.argv)

    output_mode = "files"
    if cmdline["composite_file"] is not None:
        output_mode = "composite"
        output_file = cmdline["composite_file"]

    bsize = cmdline["blocksize"]
    shiftsize = cmdline["shiftsize"]
    maxrange = cmdline["maxrange"]
    steps = cmdline["steps"]
    filtersize = cmdline["filtersize"]


    left = None
    right = None
    left_orig = None     #The original images
    right_orig = None    #they may be color images

    cropsize = [cmdline["cropx"],cmdline["cropy"],cmdline["cropw"],cmdline["croph"]]
   
    imageshift_x=cmdline["imageshiftx"] 
    imageshift_y=cmdline["imageshifty"] 
    swap_lr=False    

    if cmdline["extract"] == 1:
        image_filename = cmdline["left"]
        if cmdline["left"] is None and cmdline["right"] is not None:
            image_filename = cmdline["right"]
            swap_lr=True
        tmp = cv.LoadImageM(image_filename, cv.CV_LOAD_IMAGE_GRAYSCALE)
        left = cv.GetSubRect(tmp, (0,0,int(tmp.cols/2), tmp.rows))
        right = cv.GetSubRect(tmp, (int(tmp.cols/2),0,int(tmp.cols/2), tmp.rows))
        tmp_orig = cv.LoadImageM(image_filename, cv.CV_LOAD_IMAGE_UNCHANGED)
        left_orig = cv.GetSubRect(tmp_orig, (0,0,int(tmp.cols/2), tmp.rows))
        right_orig = cv.GetSubRect(tmp_orig, (int(tmp.cols/2),0,int(tmp.cols/2), tmp.rows))
        

    elif cmdline["extract"] == 2:
        image_filename = cmdline["left"]
        if cmdline["left"] is None and cmdline["right"] is not None:
            image_filename = cmdline["right"]
            swap_lr = True
        tmp = cv.LoadImageM(image_filename, cv.CV_LOAD_IMAGE_GRAYSCALE)
        left = cv.GetSubRect(tmp, (0,0,tmp.cols, int(tmp.rows/2)))
        right = cv.GetSubRect(tmp, (0,int(tmp.rows/2),tmp.cols, int(tmp.rows/2)))
        tmp_orig = cv.LoadImageM(image_filename, cv.CV_LOAD_IMAGE_UNCHANGED)
        left_orig = cv.GetSubRect(tmp, (0,0,tmp.cols, int(tmp.rows/2)))
        right_orig = cv.GetSubRect(tmp, (0,int(tmp.rows/2),tmp.cols, int(tmp.rows/2)))
        
    else:
        left = cv.LoadImageM(cmdline["left"], cv.CV_LOAD_IMAGE_GRAYSCALE)
        right = cv.LoadImageM(cmdline["right"], cv.CV_LOAD_IMAGE_GRAYSCALE)
        left_orig = cv.LoadImageM(cmdline["left"], cv.CV_LOAD_IMAGE_UNCHANGED)
        right_orig = cv.LoadImageM(cmdline["right"], cv.CV_LOAD_IMAGE_UNCHANGED)

    # Swap the left and right image
    if swap_lr:
            tmp_im = left
            left = right
            right=tmp_im
            tmp_im = left_orig
            left_orig = right_orig
            right_orig=tmp_im

    # Handle the image shift
    left_size = (0,0,left.cols-imageshift_x, left.rows-imageshift_y)
    right_size = (imageshift_x,imageshift_y,right.cols-imageshift_x, right.rows-imageshift_y)

    tmp = left
    left = cv.GetSubRect(tmp, left_size)
    tmp = right
    right = cv.GetSubRect(tmp, right_size)

    tmp = left_orig
    left_orig = cv.GetSubRect(tmp, left_size)
    tmp = right_orig
    right_orig = cv.GetSubRect(tmp, right_size)


    if cropsize[2] < 0:
        if cmdline["autocrop"] == 1:
            cropsize = [2*bsize[0],2*bsize[1],left.cols-2*bsize[0],left.rows-2*bsize[1]]
        else:
            cropsize = [0,0,left.cols,left.rows]


    velx = cv.CreateMat((left.rows-bsize[1])/shiftsize[1],(left.cols-bsize[0])/shiftsize[0], cv.CV_32FC1)
    vely = cv.CreateMat((left.rows-bsize[1])/shiftsize[1],(left.cols-bsize[0])/shiftsize[0], cv.CV_32FC1)

    cv.Zero(velx)
    cv.Zero(vely)
    cv.CalcOpticalFlowBM(left,right,bsize, shiftsize, maxrange, False, velx, vely)


    if filtersize > 0:
        (tmp_velx, tmp_vely) = filterVelocityField(velx,vely, filtersize)
        velx = tmp_velx
        vely = tmp_vely



    intermediate_frames = []
    flow_frames = []
    if output_mode == "composite":
        out_composite_image = cv.CreateMat(cropsize[3], (steps+1)*cropsize[2], left_orig.type)

#    print str(cropsize)


    for d in range (0, steps+1):
        base_progress = float(d)/float(steps+1)
#        print "%.2f %%"%(100.0*base_progress)
        dst = cv.CreateMat(left.rows, left.cols, cv.CV_8UC1)
        output = cv.CloneMat(left_orig)

        cv.AddWeighted(left_orig, 1.0 - (float(d)/float(steps)), right_orig, (float(d)/float(steps)), 0.0, output) 

        cv.Zero(dst)
        (new_velx, new_vely) = createVelocityField(velx,vely,float(d)/steps)


        index = 0

        block = cv.CreateMat(bsize[1], bsize[0], left_orig.type)
        block1 = cv.CreateMat(bsize[1], bsize[0], left_orig.type)
        block2 = cv.CreateMat(bsize[1], bsize[0], left_orig.type)

#        block_orig = cv.CreateMat(block[1], block[0], left_orig.type)

        for x in range (0, new_velx.cols):
            for y in range (0, new_velx.rows):
                
                if index%100 == 0:
                    print "%.2f %%"%(100.0* (base_progress + float(index)/(float(new_velx.cols)*float(new_velx.rows)*float(steps+1))) )
                index = index +1


                xpos = x*shiftsize[0] + bsize[0]/2        
                ypos = y*shiftsize[1] + bsize[1]/2        
                cv.Circle(dst, (xpos, ypos), 2, (255,255,255), 1)
                xtpos = xpos + new_velx[y,x]
                ytpos = ypos + new_vely[y,x]
                xfinalPos = xpos + velx[y,x]
                yfinalPos = ypos + vely[y,x]

                x_subpixel = xtpos - float(int(xtpos))
                try:
                    cv.Line(dst, (int(xpos), int(ypos)), (int(xtpos), int(ytpos)), (255,0,0),1)
                    if xpos - bsize[0]/2 > 0 and xpos + bsize[0]/2 < left_orig.cols-1 and \
                       ypos- bsize[1]/2 > 0 and ypos + bsize[1]/2 < left_orig.rows and \
                       xfinalPos - bsize[0]/2 > 0 and xfinalPos + bsize[0]/2 < right_orig.cols-1 and \
                       yfinalPos- bsize[1]/2 > 0 and yfinalPos + bsize[1]/2 < right_orig.rows-1:


                        tmpMatLeft = cv.GetSubRect(left_orig, (xpos - bsize[0]/2, ypos- bsize[1]/2,bsize[0],bsize[1]))    
                        tmpMatRight = cv.GetSubRect(right_orig, (xfinalPos - bsize[0]/2, yfinalPos- bsize[1]/2,bsize[0],bsize[1]))    


                        if cmdline["subpixel"]:
                            tmpMatLeft1 = cv.GetSubRect(left_orig, (1+xpos - bsize[0]/2, ypos- bsize[1]/2,bsize[0],bsize[1]))    
                            tmpMatRight1 = cv.GetSubRect(right_orig, (1+xfinalPos - bsize[0]/2, yfinalPos- bsize[1]/2,bsize[0],bsize[1]))    

                            tmpOutput = cv.GetSubRect(output, (int(xtpos) - bsize[0]/2, int(ytpos)- bsize[1]/2,bsize[0],bsize[1]))        

                            cv.AddWeighted(tmpMatLeft, 1.0 - (float(d)/float(steps)), tmpMatRight, (float(d)/float(steps)), 0.0, block) 
                            cv.AddWeighted(tmpMatLeft1, 1.0 - (float(d)/float(steps)), tmpMatRight1, (float(d)/float(steps)), 0.0, block1) 
            
                            #The destination block is positioned with subpixel accuracy, thats
                            #why the weighting of the source blocks has to be inverted
                            # x_subpixel, 1-x_subpixel instead of 1-x_subpixel, x_subpixel
                            cv.AddWeighted(block, x_subpixel , block1, 1.0-x_subpixel, 0.0, block2) 
                            cv.Copy(block2, tmpOutput) 
                        else:
                            tmpOutput = cv.GetSubRect(output, (int(xtpos) - bsize[0]/2, int(ytpos)- bsize[1]/2,bsize[0],bsize[1]))        
                            cv.AddWeighted(tmpMatLeft, 1.0 - (float(d)/float(steps)), tmpMatRight, (float(d)/float(steps)), 0.0, tmpOutput) 



                except:
                    pass # Invalid position


        flow_frames.append(dst)
        cropped_out = cv.CloneMat(cv.GetSubRect(output,(cropsize[0],cropsize[1],cropsize[2],cropsize[3])))
        if cmdline["smooth"] > 0:
            cv.Smooth(cropped_out,cropped_out, cv.CV_GAUSSIAN, cmdline["smooth"] ) # Smooth with Gaussian Kernel

        intermediate_frames.append(cropped_out)
        if output_mode == "composite":
            dst = cv.GetSubRect(out_composite_image, (d*cropsize[2],0,cropsize[2],cropsize[3]))
            cv.Copy(cropped_out, dst)
        else:
            cv.SaveImage("intermediate_%04d.bmp"%d, cropped_out)
            cv.SaveImage("intermediate_%04d.bmp"%(2*steps - d), cropped_out)


    if output_mode == "composite":
        cv.SaveImage(output_file, out_composite_image)

    if not cmdline["headless"] == 1:
        running=True
        direction=1
        pos = 0
        while running:
            pos = pos + direction

            if pos >= len(intermediate_frames)-1 or pos <= 0:
                direction = direction * (-1)

            cv.ShowImage("Output", intermediate_frames[pos])
            cv.ShowImage("Flow", flow_frames[pos])
            cv.WaitKey(20)
