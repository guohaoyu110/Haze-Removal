clear

clc

close all
% the aim is basicly no-foggy image, its dark channel is 0
% and the foggy image, return the dark channel to 0 and then get the image
% without fog 
kenlRatio = .01; % from the image, in terms of brightness, we take the fisrt 0.01 images 
minAtomsLight = 240; % the min atmosphere light 

image_name ='pumpkins.png';
img=imread(image_name);
figure,imshow(uint8(img)), title('src');

sz=size(img); % sz=[360 480 3]

w=sz(2);  % w=480 the width of the image 

h=sz(1);  % h=360 the height of the image 

dc = zeros(h,w);

for y=1:h

    for x=1:w

        dc(y,x) = min(img(y,x,:)); % this is the dark channel image 

    end

end
% the picture only has the image from the three channels 
figure,imshow(uint8(dc)),title('Min(R,G,B)');

krnlsz = floor(max([3, w*kenlRatio, h*kenlRatio])) %to the closest integer

dc2 = minfilt2(dc, [krnlsz,krnlsz]); %the min filter radius is 4 cm 

dc2(h,w)=0;

figure,imshow(uint8(dc2)), title('After filter ');

t = 255 - dc2;% dc2 is the image with fogs

figure,imshow(uint8(t)),title('t');% t is the see through rate 

t_d=double(t)/255;

sum(sum(t_d))/(h*w)


A = min([minAtomsLight, max(max(dc2))])

J = zeros(h,w,3);

img_d = double(img);

J(:,:,1) = (img_d(:,:,1) - (1-t_d)*A)./t_d;

J(:,:,2) = (img_d(:,:,2) - (1-t_d)*A)./t_d;

J(:,:,3) = (img_d(:,:,3) - (1-t_d)*A)./t_d;

figure,imshow(uint8(J)), title('J'); % the J image is the clear image after the filter 

% figure,imshow(rgb2gray(uint8(abs(J-img_d)))), title('J-img_d');
% a = sum(sum(rgb2gray(uint8(abs(J-img_d))))) / (h*w)
% return;
%----------------------------------
r = krnlsz*4;
eps = 10^-6;

% filtered = guidedfilter_color(double(img)/255, t_d, r, eps);
filtered = guidedfilter(double(rgb2gray(img))/255, t_d, r, eps);

t_d = filtered;

figure,imshow(t_d,[]),title('filtered t');

J(:,:,1) = (img_d(:,:,1) - (1-t_d)*A)./t_d;

J(:,:,2) = (img_d(:,:,2) - (1-t_d)*A)./t_d;

J(:,:,3) = (img_d(:,:,3) - (1-t_d)*A)./t_d;
% 


figure,imshow(uint8(J)), title('J_guild_filter');% the clear image using the 'Guided Image Filter'

%----------------------------------
%imwrite(uint8(J), ['_', image_name])