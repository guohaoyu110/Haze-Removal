import os
import cv2
import random
import hazerelevent
import numpy as np
import cPickle
from scipy import misc
from sklearn.ensemble import RandomForestRegressor

def Getfeature(patch):
    D = np.array([])
    C = np.array([])
    S = np.array([])
    scale = [1,4,7,10]
    for r in scale:
        hazeobj = hazerelevent.HazeRelevent(patch, scale)
        D = np.append(D, np.sort(np.reshape(hazeobj.DarkChannel(),(1, -1))) )
        C = np.append(C, np.sort(np.reshape(hazeobj.GetContrast(),(1, -1))) )
        S = np.append(S, np.sort(np.reshape(hazeobj.GetSaturation(),(1, -1))) )
    H = np.sort(np.reshape(hazeobj.GetHue(),(1, -1)))
    feature = D
    feature = np.append(feature,C)
    feature = np.append(feature,S)
    feature = np.append(feature,H)
    return feature

# base_path = '/home/lab-zeng.lingke/Downloads/datasets/dehaze/build/'
base_path = './dehaze/'
patch_size = 16
patch_nums = 10
A = 1
Features = []
Labels = []
for __, __, files in os.walk(base_path):
    for pic in files:
        print 'Process image: ', (len(Features)/500 + 1)
        image = cv2.imread(base_path + pic)
        image = misc.imresize(image, 0.5)
        for scale in [1, 1.25, 1.5, 1.75, 2]:
            reimg = misc.imresize(image, 1.0/scale)
            x_pos = random.sample(range(0, ((reimg.shape)[0]-patch_size)), patch_nums)
            y_pos = random.sample(range(0, ((reimg.shape)[1]-patch_size)), patch_nums)
            pos_num = len(x_pos)
            for patch_num in range(0, pos_num):
                pos = [x_pos[patch_num], y_pos[patch_num]]
                img_sample = reimg[pos[0]:(pos[0]+patch_size), pos[1]:(pos[1]+patch_size), :]
                img_sample = img_sample / 255.0
                for t_num in range(1,11):
                    #t = random.random()
                    t = t_num/10.0
                    haze_img = t*img_sample + A * (1-t)
                    haze_img[(haze_img<0)] = 0
                    haze_img[(haze_img>1)] = 1
                    feature = Getfeature(haze_img)
                    Features.append(feature)
                    Labels.append(t)
                print 'Process patch', patch_num
order = random.sample(xrange(len(Features)), len(Features))
rFeatures = []
rLabels = []
for i in xrange(len(order)):
    rFeatures.append(Features[i])
    rLabels.append(Labels[i])
    
print 'DataSet Finished!'
np.save('random_data5.npy', rFeatures)
np.save('random_label5.npy', rLabels)
print 'DataSet Saved!'

model = RandomForestRegressor(n_estimators=200, max_features=(1/3.0), n_jobs=24)
model = model.fit(rFeatures, rLabels)
print 'Model Finished!'
with open('RandomModel.pkl', 'wb') as f:
    cPickle.dump(model, f)
print 'Model Saved!'
