#!/usr/bin/python
import sys
import numpy
import cv2
from optparse import OptionParser

def cvToNumpy(mat):
  im = numpy.zeros((mat.rows,mat.cols), dtype=numpy.uint8)
  for y in range(mat.rows):
    for x in range(mat.cols):
      im[y,x] = mat[y,x]
  return im

# Creates a x/y velocity field on the interval [0,1]
def createVelocityField(velx , vely, delta):
    x_rows = velx.shape[0]
    y_rows = vely.shape[0]
    x_cols = velx.shape[1]
    y_cols = vely.shape[1]
    new_velx = cv.CreateMat(x_rows, x_cols, cv.CV_32FC1)
    new_vely = cv.CreateMat(y_rows, y_cols, cv.CV_32FC1)
    for x in range (0, x_cols):
        for y in range (0, x_rows):    
            new_velx[y,x] = velx[y,x] * delta
            new_vely[y,x] = vely[y,x] * delta

    return (new_velx, new_vely)


"""Extracts a block from an image and takes care of the boundary cases
   Always returns an image of size (width,height). Non existing parts
   of the source image are black
"""
def safeBlockExtract(image, xpos, ypos, width, height):
  #from pudb import set_trace; set_trace()
  shape = list(image.shape)
  shape[0] = height
  shape[1] = width
  block = numpy.zeros(shape)
  
  xpos = int(xpos)
  ypos = int(ypos)
  width = int(width)
  height = int(height)
  
  real_x = min(image.shape[1], 0 if xpos < 0 else xpos)
  real_y = min(image.shape[0], 0 if ypos < 0 else ypos)
  
  corr_x = -min(xpos+real_x, 0)
  corr_y = -min(ypos+real_y, 0)
  
  tmp_x = min (xpos + width, image.shape[1])
  tmp_y = min (ypos + height, image.shape[0])
    
  corr_w = -min((tmp_x-xpos-width), 0)
  corr_h = -min((tmp_y-ypos-height), 0)
     
  real_width = width - corr_x - corr_w
  real_height = height - corr_y - corr_h
  
  if real_width > 0 and real_height > 0:
    block[corr_y:corr_y+real_height, corr_x:corr_x+real_width] = image[real_y:real_y+real_height, real_x:real_x+real_width]
  
  return block


"""ULTRASLOW !!!! 
   use with caution
"""  
def filterVelocityField(velx,vely, image1, image2, bsize=16):
    newvelx = numpy.copy(velx)
    newvely = numpy.copy(vely)
    
    for y in xrange(velx.shape[0]):
      for x in xrange(bsize, velx.shape[1]-bsize):
        
        vx1 = velx[y,x-bsize]
        vx2 = velx[y,x]
        vx3 = velx[y,x+bsize]
        
        vy1 = vely[y,x-bsize]
        vy2 = vely[y,x]
        vy3 = vely[y,x+bsize]
        
        
        xpos1 = int(x + vx1)
        xpos2 = int(x + vx2)
        xpos3 = int(x + vx3)
        ypos1 = int(y + vy1)
        ypos2 = int(y + vy2)
        ypos3 = int(y + vy3)

        srcblock = safeBlockExtract(image1, x-bsize/2,y-bsize/2, bsize,bsize)
        block1 = safeBlockExtract(image2, xpos1-bsize/2,ypos1-bsize/2, bsize,bsize)
        block2 = safeBlockExtract(image2, xpos2-bsize/2,ypos2-bsize/2, bsize,bsize)
        block3 = safeBlockExtract(image2, xpos3-bsize/2,ypos3-bsize/2, bsize,bsize)
        
        dist1 = numpy.sum(numpy.abs(block1-srcblock))
        dist2 = numpy.sum(numpy.abs(block2-srcblock))
        dist3 = numpy.sum(numpy.abs(block3-srcblock))

        #print "%f %f %f (%f,%f) (%f,%f) (%f,%f)"%(dist1, dist2, dist3, vx1,vy1, vx2,vy2,vx3,vy3)

        
        vx = vx1
        vy = vy1
        
        if dist2<=dist1:
          vx = vx2
          vy = vy2

        if dist3<=dist2 and dist3<=dist1:
          vx = vx3
          vy = vy3

        newvelx[y,x] = vx
        newvely[y,x] = vy

        
        
  
    return newvelx,newvely
    
    
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
            dest="filtersize", help="filterblocksize (warning slow!!)",
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
    parser.add_option("-B","--bg", action="store", type="int",
            dest="background", help="Interpolated image in background",
            default = 1)


    (options, args) = parser.parse_args(argv)

    arguments = {}
    arguments["headless"] = options.headless
    arguments["steps"] = options.steps
    arguments["left"] = options.left
    arguments["right"] = options.right
    arguments["autocrop"] = options.autocrop
    arguments["background"] = options.background
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
    drawbg = True if cmdline["background"] else False


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
        tmp = cv2.imread(image_filename, cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)
        left = tmp[:,:int(tmp.shape[1]/2)]
        right = tmp[:,int(tmp.shape[1]/2):]
        tmp_orig = cv2.imread(image_filename, cv2.cv.CV_LOAD_IMAGE_UNCHANGED)
        left_orig = tmp_orig[:,:int(tmp.shape[1]/2)]
        right_orig = tmp_orig[:,int(tmp.shape[1]/2):]
        

    elif cmdline["extract"] == 2:
        image_filename = cmdline["left"]
        if cmdline["left"] is None and cmdline["right"] is not None:
            image_filename = cmdline["right"]
            swap_lr = True
        tmp = cv2.imread(image_filename, cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)
        left = tmp[:int(tmp.shape[0]/2),:]
        right = tmp[int(tmp.shape[0]/2):,:]
        tmp_orig = cv2.imread(image_filename, cv2.cv.CV_LOAD_IMAGE_UNCHANGED)
        left_orig = tmp_orig[:int(tmp.shape[0]/2),:]
        right_orig = tmp_orig[int(tmp.shape[0]/2):,:]
        
    else:
        left = cv2.imread(cmdline["left"], cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)
        right = cv2.imread(cmdline["right"], cv2.cv.CV_LOAD_IMAGE_GRAYSCALE)
        left_orig = cv2.imread(cmdline["left"], cv2.cv.CV_LOAD_IMAGE_UNCHANGED)
        right_orig = cv2.imread(cmdline["right"], cv2.cv.CV_LOAD_IMAGE_UNCHANGED)

    # Swap the left and right image
    if swap_lr:
            left,right = right,left
            left_orig,right_orig = right_orig,left_orig
            
    # Handle the image shift
    left_size = (0,0,left.shape[1]-imageshift_x, left.shape[0]-imageshift_y)
    right_size = (imageshift_x,imageshift_y,right.shape[1], right.shape[0])

    left = left[left_size[1]:left_size[3], left_size[0]: left_size[2]]
    right = right[right_size[1]:right_size[3], right_size[0]: right_size[2]] 
    
    left_orig = left_orig[left_size[1]:left_size[3], left_size[0]: left_size[2]]
    right_orig = right_orig[right_size[1]:right_size[3], right_size[0]: right_size[2]] 


    if cropsize[2] < 0:
        if cmdline["autocrop"] == 1:
            cropsize = [2*bsize[0],2*bsize[1],left.shape[1]-2*bsize[0],left.shape[0]-2*bsize[1]]
        else:
            cropsize = [0,0,left.shape[1],left.shape[0]]
   
    flow = cv2.calcOpticalFlowFarneback(left, right, pyr_scale=0.5, levels=3, winsize=bsize[0], iterations=10, poly_n=5, poly_sigma=1.1, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN, flow=None)
    
    velx = flow[:,:,0]
    vely = flow[:,:,1]

    #TODO: Limit the searchrange in the optical flow detector
    # simply clamping the motion will yield inferior results
    velx[velx > maxrange[0]] = maxrange[0]
    velx[velx < -maxrange[0]] = -maxrange[0]

    vely[vely > maxrange[1]] = maxrange[1]
    vely[vely < -maxrange[1]] = -maxrange[1]
    
    
    if filtersize > 1:
        (tmp_velx, tmp_vely) = filterVelocityField(velx,vely, left, right, filtersize)
        velx = tmp_velx
        vely = tmp_vely

    intermediate_frames = []
    flow_frames = []
    if output_mode == "composite":
        shape = list(left_orig.shape)
        shape[0] = cropsize[3]
        shape[1] = (2*steps+1)*cropsize[2]
        out_composite_image = numpy.zeros(shape)



    for d in range (0, steps+1):
        base_progress = float(d)/float(steps+1)
#        print "%.2f %%"%(100.0*base_progress)
        dst = numpy.copy(left)
        dst[:,:] = 0
        if drawbg:
          output = (1.0 - (float(d)/float(steps))) * numpy.float32(left_orig) + (float(d)/float(steps)) * numpy.float32(right_orig)
        else:
          output = numpy.zeros(left_orig.shape, dtype=numpy.float32)

        delta = float(d)/steps
        new_velx = delta * velx
        new_vely = delta * vely


        #Store all blocks that are moved, to sort them according to their
        #distance to the camera
        movements = []

        for y in range (bsize[1]/2, new_velx.shape[0],shiftsize[1]):
          for x in range (bsize[0]/2, new_velx.shape[1],shiftsize[0]):
            
                xpos = int(x)
                ypos = int(y)
                
                xtpos = xpos + new_velx[y,x]
                ytpos = ypos + new_vely[y,x]
                xfinalPos = xpos + velx[y,x]
                yfinalPos = ypos + vely[y,x]
                """The first element is the x-shift of the block
                   smaller shifts (maybe even negative) belong to
                   objects closer to the camera and have to be drawn last"""
                movements.append( (new_velx[y,x], xpos, ypos, xtpos, ytpos, xfinalPos, yfinalPos ) )

        index = 0

        movements.sort(reverse=True)

        for m in movements:
                if index%100 == 0:
                    print "%.2f %%"%(100.0* (base_progress + float(index)/(len(movements)*float(steps+1))) )
                index = index +1
                xpos = m[1]        
                ypos = m[2]
                xtpos = int(m[3])
                ytpos = int(m[4])
                xfinalPos = int(m[5])
                yfinalPos = int(m[6])
                x_subpixel = m[3] - float(int(xtpos))


                cv2.circle(dst, (xpos, ypos), 2, (255,255,255), 1)

                cv2.line(dst, (int(xpos), int(ypos)), (int(xtpos), int(ytpos)), (255,0,0),1)
                if xpos - bsize[0]/2 > 0 and xpos + bsize[0]/2 < left_orig.shape[1]-1 and \
                   ypos- bsize[1]/2 > 0 and ypos + bsize[1]/2 < left_orig.shape[0] and \
                   xfinalPos - bsize[0]/2 > 0 and xfinalPos + bsize[0]/2 < right_orig.shape[1]-1 and \
                   yfinalPos- bsize[1]/2 > 0 and yfinalPos + bsize[1]/2 < right_orig.shape[0]-1:
                    
                   tmpMatLeft = left_orig[ypos- bsize[1]/2:ypos+ bsize[1]/2, xpos - bsize[0]/2:xpos + bsize[0]/2]
                   tmpMatRight = right_orig[yfinalPos- bsize[1]/2:yfinalPos+ bsize[1]/2, xfinalPos - bsize[0]/2:xfinalPos + bsize[0]/2]    

                   if cmdline["subpixel"]:
                       tmpMatLeft1 = left_orig[ypos- bsize[1]/2:ypos+ bsize[1]/2, 1+xpos - bsize[0]/2:1+xpos + bsize[0]/2 ]
                       tmpMatRight1 = right_orig[yfinalPos- bsize[1]/2:yfinalPos+ bsize[1]/2, 1+xfinalPos - bsize[0]/2:1+xfinalPos + bsize[0]/2 ]    
                       
                       block = cv2.addWeighted(tmpMatLeft, 1.0 - (float(d)/float(steps)), tmpMatRight, (float(d)/float(steps)), 0.0) 
                       block1 = cv2.addWeighted(tmpMatLeft1, 1.0 - (float(d)/float(steps)), tmpMatRight1, (float(d)/float(steps)), 0.0) 
       
                       #The destination block is positioned with subpixel accuracy, thats
                       #why the weighting of the source blocks has to be inverted
                       # x_subpixel, 1-x_subpixel instead of 1-x_subpixel, x_subpixel
                       block2 = cv2.addWeighted(block, x_subpixel , block1, 1.0-x_subpixel, 0.0) 
                       output[int(ytpos)- bsize[1]/2:int(ytpos)+ bsize[1]/2,  int(xtpos) - bsize[0]/2:int(xtpos) + bsize[0]/2] = block2
                  
                   else:
                       b = cv2.addWeighted(tmpMatLeft, 1.0 - (float(d)/float(steps)), tmpMatRight, (float(d)/float(steps)), 0.0) 
                       output[int(ytpos)- bsize[1]/2:int(ytpos)+ bsize[1]/2,int(xtpos) - bsize[0]/2:int(xtpos) + bsize[0]/2] = b

        flow_frames.append(dst)
        cropped_out = numpy.copy(output[cropsize[1]:cropsize[3]+cropsize[1], cropsize[0]:cropsize[2]+cropsize[0]])
        if cmdline["smooth"] > 0:
            cropped_out = cv2.GaussianBlur( cropped_out, (3,3), 0.5)

        intermediate_frames.append(numpy.uint8(cropped_out))
        if output_mode == "composite":
            out_composite_image[:,d*cropsize[2]:(d+1)*cropsize[2]] = cropped_out
            out_composite_image[:,(2*steps-d)*cropsize[2]:(2*steps-d+1)*cropsize[2]] = cropped_out
        else:
            cv2.imwrite("intermediate_%04d.bmp"%d, numpy.uint8(cropped_out))
            cv2.imwrite("intermediate_%04d.bmp"%(2*steps - d), numpy.uint8(cropped_out))

    if output_mode == "composite":
        cv2.imwrite(output_file, out_composite_image)

    if not cmdline["headless"] == 1:
        running=True
        direction=1
        pos = 0
        while running:
            pos = pos + direction

            if pos >= len(intermediate_frames)-1 or pos <= 0:
                direction = direction * (-1)

            cv2.imshow("Output", intermediate_frames[pos])
            cv2.imshow("Flow", flow_frames[pos])
            cv2.waitKey(20)
