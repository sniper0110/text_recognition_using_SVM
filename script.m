clear all, clc;

%% Saving images after transforming them into grayscale images
newPath = '/Users/mac/Documents/MSCV 01/Semester2/Pattern Recognition/Project/new_data/code/grayImages/';
for j=53:62
    
   path = '/Users/mac/Documents/MSCV 01/Semester2/Pattern Recognition/Project/new_data/English/Img/GoodImg/Bmp/';
   if(j >= 1 && j <= 9)
       samp = strcat('Sample00', num2str(j));
   else
       samp = strcat('Sample0', num2str(j));
   end

   fullPath = strcat(path, samp, '/');
   srcFiles = dir(fullPath);
   
   trainSet = int16(length(srcFiles) * 80 / 100);
   
   j
   % Where to save grayscale images
   savingPath = strcat(newPath, 'Samples', num2str(j), '/');
   
   for i = 1 : length(srcFiles)
      
      %
      trainPath = strcat(savingPath, 'train/');
      testPath = strcat(savingPath, 'test/');
      
      filename = strcat(fullPath,srcFiles(i).name);
      if(findstr(srcFiles(i).name, 'img') >= 1)
          i
          Img = imread(filename);
          if(size(Img, 3) == 3)
             grayImage = rgb2gray(Img);
          else
             grayImage = Img;
          end
          
          if(i <= trainSet+2)
             s = strcat(trainPath, srcFiles(i).name); 
             imwrite(grayImage, s);
          else
             s = strcat(testPath, srcFiles(i).name); 
             imwrite(grayImage, s, 'png');
          end
      end
      
      
   

   end

end

%% Finding the average height and width to normalize the images

meanLength = 0; % heigth
meanWidth = 0;  % width
nbrImages = 0;

for j=1:62
    
    
    fullPath = strcat(newPath, 'Samples', num2str(j), '/train/');
    srcFiles = dir(fullPath);
    
    for i=1:length(srcFiles)
        
       if(findstr(srcFiles(i).name, 'img') >= 1)
            filename = strcat(fullPath,srcFiles(i).name);
            im = imread(filename);
            meanLength = meanLength + size(im, 1); 
            meanWidth = meanWidth + size(im, 2);
            nbrImages = nbrImages + 1;
       end
        
    end
    
end

% meanLength/nbrImages
% meanWidth/nbrImages
%%
% Path to pre-processed images
resPath = '/Users/mac/Documents/MSCV 01/Semester2/Pattern Recognition/Project/new_data/code/grayImagesResized/';

%% Resizing Images to have same size
for j=1:62
   path1 = strcat(newPath, 'Samples', num2str(j), '/train/'); 
   path2 = strcat(newPath, 'Samples', num2str(j), '/test/');
   
   srcFiles1 = dir(path1);
   srcFiles2 = dir(path2);
   
   for i1=1:length(srcFiles1)
       if(findstr(srcFiles1(i1).name, 'img') >= 1)
           s = strcat(newPath, 'Samples', num2str(j), '/train/', srcFiles1(i1).name);
           im = imread(s);
           
           im = imresize(im, [91, 65]);
           
           
           resTrainPath = strcat(resPath, 'Samples', num2str(j), '/train/', srcFiles1(i1).name);
           imwrite(im, resTrainPath);
       end
   end
   
   for i2=1:length(srcFiles2)
       if(findstr(srcFiles2(i2).name, 'img') >= 1)
           s = strcat(newPath, 'Samples', num2str(j), '/test/', srcFiles2(i2).name);
           im = imread(s);
           im = imresize(im, [91, 65]);
           
           resTestPath = strcat(resPath, 'Samples', num2str(j), '/test/', srcFiles2(i2).name);
           imwrite(im, resTestPath);
       end
   end
      
end

%% Training set : Extracting HOG features from images and forming X and y matrix
clc;
X = [];
y = [];
count = 0;

for j=1:62
    
    
    fullPath = strcat(resPath, 'Samples', num2str(j), '/train/');
    srcFiles = dir(fullPath);
    
    for i=1:length(srcFiles)
        
       if(findstr(srcFiles(i).name, 'img') >= 1)
            filename = strcat(fullPath,srcFiles(i).name);
            im = imread(filename);
            % extract HOG features
            [hog_10x10, vis10x10] = extractHOGFeatures(im,'CellSize',[10 10]);
            X = [hog_10x10; X];
            y = [j; y];
            count = count + 1;
       end
        
    end
    
end

%% Predictions using SVM
Mdl = fitcecoc(X,y); % Multi-class classification using SVM


%% Testing HOG features
img = imread('/Users/mac/Documents/MSCV 01/Semester2/Pattern Recognition/Project/new_data/code/grayImagesResized/Samples11/train/img011-00004.png');
imshow(img)
% Extract HOG features 
[hog_10x10s, vis10x10] = extractHOGFeatures(img,'CellSize',[10 10]);
predict(Mdl, hog_10x10s)

%plot(vis10x10);
%title({'CellSize = [10 10]'; ['Length = ' num2str(length(hog_10x10))]});

%% Predictions for training set

predictions_train = [];
predictions_test = [];

for i=1:size(X,1)
        pred1 = predict(Mdl, X(i,:)); 
        predictions_train = [pred1, predictions_train]; 
end


%% Test set : Extracting HOG features from images and forming X_train and y_train
clc;
X_test = [];
y_test = [];
count = 0;

for j=1:62
 
    fullPath = strcat(resPath, 'Samples', num2str(j), '/test/');
    srcFiles = dir(fullPath);
    
    for i=1:length(srcFiles)
        
       if(findstr(srcFiles(i).name, 'img') >= 1)
            filename = strcat(fullPath,srcFiles(i).name);
            im = imread(filename);
            % extract HOG features
            [hog_10x10, vis10x10] = extractHOGFeatures(im,'CellSize',[10 10]);
            X_test = [hog_10x10; X_test];
            y_test = [j; y_test];
            count = count + 1;
       end
        
    end
    
end

%% Predictions for test set

predictions_test = [];

for i=1:size(X_test,1)
        pred1 = predict(Mdl, X_test(i,:)); 
        predictions_test = [pred1, predictions_test]; 
end

%% Computing model accuracy

a1=predictions_train(end:-1:1);
train_accuracy = mean(a1' == y) * 100

a2=predictions_test(end:-1:1);
test_accuracy = mean(a2' == y_test) * 100





