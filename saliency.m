image_list = dir('image/*.jpg');

for i = 1:size(image_list)
    disp(image_list(i).name)
    img = imread(strcat('image/', image_list(i).name));
    img = rgb2gray(img);

    sal_map = ROI_saliency_map(img);

    figure(i);
    subplot(1, 2, 1);
    imshow(img);
    title('original');

    subplot(1, 2, 2);
    imshow(sal_map);
    title('saliency map');
end
