%��������һ�Σ�֮��������
[net_bn, info_bn] = cnn_mnist('batchNormalization', true);
load('E:\ѧϰ\����ѧϰ\matconvnet-1.0-beta20\data\mnist-zyp-simplenn-bnorm\imdb.mat');
im=imread('E:\ѧϰ\����ѧϰ\matconvnet-1.0-beta20\photos\QQ��ͼ20160922172145.png');
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
        title('�жϽ��������ƻ��');
    case 2
        title('�жϽ����1��ƻ��');
    case 3
        title('�жϽ����2��ƻ��');
    case 4 
        title('�жϽ����3��ƻ��');
end
