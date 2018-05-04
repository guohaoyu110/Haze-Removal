# Haze-Removal-Project

## 主要参考文献

Visibility in bad weather from a single image
Guided image filtering 
Fast Haze Removal Algorithm for Surveillance Video   这是一篇讲如何对视频进行快速去雾的文章，没涉及到具体的算法，不过可以看看。
Fast image dehazing using guided joint bilateral filter
Efficient Image Dehazing with Boundary Constraint and Contextual Regularization

### 何恺明之前单幅图像去雾的经典方法是：
（1）最大化局部对比度：R. Tan, Visibility in Bad Weather from a Single Image, CVPR’08
（2）独立成分分析：R. Fattal, Single Image Dehazing, SIGGRAPH’08

### 新的成果：
下面试几篇比较新的去雾文献，文章等级比较高，大家感兴趣可以根据文章题目自行查询下载文献。
我发现大部分去雾方法中，主要还是提出新颖的算法致力于对大气光值的准确估计以及透射率的计算和修正。

### 单幅图像

1、2008,Single image dehazing,R. Fattal,ACM Transactions on Graphics.
2、2014,Efficient Image Dehazing with Boundary Constraint and Contextual Regularization, G. Meng, Y. Wang, J. Duan, S. Xiang and C. Pan,ICCV            
3、2016,Non-local image dehazing,D. Berman, T. Treibitz and S. Avidan, CVPR.
4、2009,Single image haze removal using dark channel prior,K. He, J. Sun and X. Tang,CVPR.
5、2017,Single Image Dehazing Based on the Physical Model and MSRCR Algorithm,J. B. Wang, K. Lu, J. Xue, N. He and L. Shao,TCSVT
6、2013,Hardware Implementation of a Fast and Efficient Haze Removal Method,Y. H. Shiau, H. Y. Yang, P. Y. Chen and Y. Z. Chuang,TCSVT
7、2014,Visibility Restoration of Single Hazy Images Captured in Real-World Weather Conditions，S. C. Huang, B. H. Chen and W. J. Wang,TCSVT
10、2017,Haze Removal Using the Difference-Structure-Preservation Prior,L. Y He, J. Z. Zhao, N. N. Zheng and D.Y. Bi,TIP      
  






S.G. Narasimhan and S.K. Nayar， 多幅图像（同一场景不同时间、天气）去雾 主页
NASA， Retinex理论增强，主页。 
Ana Belén Petro总结了NASA的Retinex理论，源代码，不过不是matlab版本的。
Kopf，Deep Photo: Model-Based Photograph Enhancement and Viewing，3D场景去雾，没有源码。主页地址
Fattal, single image dehazing, 主页*matlab代码*
Fattal 2014，Automatic Recovery of the Atmospheric Light in Hazy Images，大气光恢复去雾，有代码，主页
Fattal 2014，Dehazing using Color-Lines，无代码，主页 
这里有个Matlab script converting jet-color images into [0,1] transmission values 主页
Tarel,Fast visibility restoration from a single color or gray level image,matlab代码*实验主页*
He kaiming， single image dehazing using dark channel prior,实验主页 
其guided image dehazing，主页还有matlab代码

Nishino，bayesian defogging，贝叶斯去雾，主页

Ancuti，inverse-image dehazing， fusion-based dehazing，水下融合去雾,个人主页*半反去雾主页*
Ketan Tang, 基于学习的去雾Investigating haze-relevant features in a learning framework for image dehazing， 实验主页
Gibson，维纳滤波去雾，fast single image fog removal using the adaptive wiener filter，主页
Meng gaofeng,改进的暗原色去雾efficinet image dehazing with boundary constraint contextual regularization，matlab代码
Yoav Y.Schechner,一直研究偏振去雾算法，典型的代表作，blind haze separation, advanced visiblity improvement based on polarization filtered images，主页
yk wang，Single Image Defogging by Multiscale Depth Fusion，也是基于贝叶斯和马尔可夫来去雾，暂时没公布matlab代码。主页
Jin-Hwan Kim, optimized contrast enhancement for real-time image and video dehazing, 关于图像增强和视频去雾的，主页有代码，但是是C程序的。主页

ECCV2016 Single Image Dehazing via Multi-Scale Convolutional Neural Networks，主页
2015年的一篇CVPR，主页有代码，Simultaneous Video Defogging and Stereo Reconstruction 链接 
此外还有文章主页：http://www.lizhuwen.com/pages/Stereo%20in%20Fog.html
### 关于去雾算法质量评价对比 
1、Zhengying Chen，Quality Assessment for Comparing Image Enhancement Algorithms（CVPR2014），基于学习的去雾算法排序方法，据说有数据库，但得填表找他们要，主页 
2、Gibson，A No-Reference Perceptual Based Contrast Enhancement Metric for Ocean Scenes in Fog（TIP，2013），一种CEM评价方法，不过也是基于学习的，数据库和代码都有。主页 
3、Hautiere，Blind contrast enhancement assessment by gradient ratioing at visible edges，三种忙评价方法。代码网络上有，原作者编写的在这里，主页
