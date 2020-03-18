%% blur
% script to quickly blur images
clc; clear all;
%%  
% load in un-blurred image 
fname = 'peppers.jpg'
I = imread(fname);

% save un-blurred image
imwrite(I, 'peppers.png')

% blur image
Iblur = imgaussfilt(I,2);

% save blurred image
imwrite(Iblur, 'peppers_blurry.png')
