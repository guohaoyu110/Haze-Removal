'''
define 4 haze relevent feature function
'''
import cv2
import numpy as np

class HazeRelevent:
    def __init__(self, image, scale):
        self.image = image
        self.scale = scale

    def DarkChannel(self):
        JDark = np.zeros((5, 5))
        pad_size = self.scale[0] / 2
        padimage = self.image[(5-pad_size):(10+pad_size), (5-pad_size):(10+pad_size), :]
        for i in xrange(5):
            for j in xrange(5):
                patch = padimage[i:(i+self.scale[0]), j:(j+self.scale[0]), :]
                JDark[i,j] = np.min(patch)
        return JDark

    def GetHue(self):
        image = self.image[5:10,5:10,:]
        height, width, channel = image.shape
        image_si = image.copy()
        for i in xrange(height):
            for j in xrange(width):
                for k in xrange(channel):
                    if (image_si[i,j,k] < 0.5):
                        image_si[i,j,k] = 1 - image_si[i,j,k]
        image = np.uint8(image * 255.0)
        image_si = np.uint8(image_si * 255.0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        image_si = cv2.cvtColor(image_si, cv2.COLOR_BGR2HLS)
        Hue = image[:,:,0]
        Hue_si = image_si[:,:,0]
        Hue = Hue / 180.0
        Hue_si = Hue_si / 180.0
        JHue = np.abs(Hue - Hue_si)
        return JHue

    def GetSaturation(self):
        height, width, __ = self.image.shape
        JSat = np.zeros((height, width))
        for i in xrange(height):
            for j in xrange(width):
                JSat[i, j] = 1 - self.zero_dive(np.min(self.image[i,j,:]), np.max(self.image[i,j,:]))
        pad_size = self.scale[0] / 2
        padimage = JSat[(5-pad_size):(10+pad_size), (5-pad_size):(10+pad_size)]
        JSat = np.zeros((5, 5))
        for i in xrange(5):
            for j in xrange(5):
                patch = padimage[i:(i+self.scale[0]), j:(j+self.scale[0])]
                JSat[i,j] = np.max(patch)
        return JSat

    def zero_dive(self, a, b):
        if(b != 0 ):
            return (a * 1.0)/b
        else:
            return 1

    def GetVariance(self, s=5):
        height, width, channel = self.image.shape
        var_image = np.zeros((height, width))
        pad_size = s / 2
        padimage_r = np.pad(self.image[:,:,0], (pad_size, pad_size), 'constant', constant_values=(np.inf, np.inf))
        padimage_g = np.pad(self.image[:,:,1], (pad_size, pad_size), 'constant', constant_values=(np.inf, np.inf))
        padimage_b = np.pad(self.image[:,:,2], (pad_size, pad_size), 'constant', constant_values=(np.inf, np.inf))
        padimage = np.zeros((height+pad_size*2,width+pad_size*2, 3))
    	padimage[:,:,0] = padimage_r
    	padimage[:,:,1] = padimage_g
    	padimage[:,:,2] = padimage_b
        for i in xrange(height):
            for j in xrange(width):
                patch = padimage[i:i+s, j:j+s,:]
                center = np.ones((s,s,channel))*self.image[i,j,:]
                var = (patch - center) ** 2
                var = var[var <= 1]
                var_image[i,j] =  np.sqrt( np.sum(var) / var.shape )
        return var_image

    def GetContrast(self):
        var_image = self.GetVariance()
        JCon = np.zeros((5, 5))
        pad_size = self.scale[0] / 2
        padimage = var_image[(5-pad_size):(10+pad_size), (5-pad_size):(10+pad_size)]
        for i in xrange(5):
            for j in xrange(5):
                patch = padimage[i:(i+self.scale[0]), j:(j+self.scale[0])]
                JCon[i,j] = np.max(patch)
        return JCon