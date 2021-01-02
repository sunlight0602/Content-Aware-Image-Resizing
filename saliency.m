img = imread('image/cat.jpg');
img = rgb2gray(img);

sal_map = ROI_saliency_map(img);

imshow(sal_map);
