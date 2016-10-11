%初次运行一次，之后不再运行
[net_bn, info_bn] = cnn_mnist('batchNormalization', true);
load('E:\学习\机器学习\matconvnet-1.0-beta20\data\mnist-zyp-simplenn-bnorm\imdb.mat');
im=imread('E:\学习\机器学习\matconvnet-1.0-beta20\photos\QQ截图20160922172145.png');
im=imresize(im,[64 64 ]);
imshow(im);
im = single(im);
im = im - images.data_mean;
res = vl_simplenn(net_bn, im,[],[],...
                      'accumulate', 0, ...
                      'mode', 'test', ...
                      'backPropDepth', inf, ...
                      'sync', 0, ...
                      'cudnn', 1) ;
scores = res(11).x(1,1,:);
[bestScore, best] = max(scores);
switch best
    case 1
        title('判断结果：不是苹果');
    case 2
        title('判断结果：1个苹果');
    case 3
        title('判断结果：2个苹果');
    case 4 
        title('判断结果：3个苹果');
end
