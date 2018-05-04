function dehazedImageRGB = mscnndehazing(imagename, gamma)

%% option
whitebalance = 1;   % 0 or 1
adaptT = 1;         % 0 or 1
% gamma = 13;
method = 'our'; 
opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.useGpu = false;

% for ii = 1:length(hazy_data)
%     ii=3;
%     disp(ii)
    % read hazy images
    img0 = imread(imagename);
%     imwrite(img0,['./resultsall/',hazy_data(ii).name]);
    [row,col,chann]=size(img0);
    
    % estimate A
    img_n=im2double(img0);
    A = Airlight(img_n, method, 15); 
    img=single(img0)/255;
    
    if whitebalance == 1
        % white balance
        for c=1:3
            img(:,:,c) = img(:,:,c)/A(c);
        end
    end
    % load net parameters
    cnnet='MSCNNDehazing_CL.mat';
    load(cnnet);
    n = numel(net.layers);
    res = struct('x', cell(1,n+1), 'dzdx', cell(1,n+1), 'dzdw', cell(1,n+1), ...
        'aux', cell(1,n+1), 'time', num2cell(zeros(1,n+1)), 'backwardTime', num2cell(zeros(1,n+1))) ;
    res(1).x=img;
    for i=1:n-1
      l = net.layers{i};
      res(i).time = tic ;
      switch l.type
        case 'conv'
          res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride) ;
        case 'pool'
          res(i+1).x = vl_nnpool(res(i).x, l.pool, 'pad', l.pad, 'stride', l.stride, 'method', l.method) ;

        case 'upsmpl'
          res(i+1).x = upSmpl(res(i).x, 'fwd');

        case 'normalize'
          res(i+1).x = vl_nnnormalize(res(i).x, l.param);
        case 'softmax'
          res(i+1).x = vl_nnsoftmax(res(i).x) ;
        case 'loss'
          res(i+1).x = vl_nnloss(res(i).x, l.class) ;
        case 'softmaxloss'
          res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
        case 'relu'
          res(i+1).x = vl_nnrelu(res(i).x) ;
        case 'noffset'
          res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
        case 'dropout'
          if opts.disableDropout
            res(i+1).x = res(i).x ;
          elseif opts.freezeDropout
            [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i+1).aux) ;
          else
            [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
          end
        case 'custom'
          res(i+1) = l.forward(l, res(i), res(i+1)) ;
        otherwise
          error('Unknown layer type %s', l.type) ;
      end
      if opts.conserveMemory && ~doder && i < numel(net.layers) - 1
        % TODO: forget unnecesary intermediate computations even when
        % derivatives are required
        res(i).x = [] ;
      end
      res(i).time = toc(res(i).time);
    end


    l = net.layers{end};
    map = lcomb(res(n).x, l.filters, l.biases);
    map1 = imresize(map,[row,col]);
%     imwrite(map1,['./resultsall/',hazy_data(ii).name,'_trainsm_layer1.png']);
    
    %% fine layer
    clear net.layers;
%     nnet='NYUStereo_layer2_7535510_dcay_0_0005_convSz_7  5  3_channel6_5  10.o1046.mat';
    nnet='MSCNNDehazing_FL.mat';
    load (nnet);

    n = numel(net.layers) ;
    res = struct('x', cell(1,n+1), 'dzdx', cell(1,n+1), 'dzdw', cell(1,n+1), ...
        'aux', cell(1,n+1), 'time', num2cell(zeros(1,n+1)), 'backwardTime', num2cell(zeros(1,n+1))) ;
    res(1).x=img;
    for i=1:n-1
      l = net.layers{i};
      res(i).time = tic ;
      switch l.type
        case 'conv'
          res(i+1).x = vl_nnconv(res(i).x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride) ;
          if i == 1
              [row1,col1,chann1,numb1]=size(map1);
              [row2,col2,chann2,numb2]=size(res(i+1).x);
              if row1~=row2 || col1~=col2
                  map1 = imresize(map1,[row2,col2]);
                  res(i+1).x(:,:,6)=map1;
              end
          end
        case 'pool'
          res(i+1).x = vl_nnpool(res(i).x, l.pool, 'pad', l.pad, 'stride', l.stride, 'method', l.method) ;

        case 'upsmpl'
          res(i+1).x = upSmpl(res(i).x, 'fwd');

        case 'normalize'
          res(i+1).x = vl_nnnormalize(res(i).x, l.param) ;
        case 'softmax'
          res(i+1).x = vl_nnsoftmax(res(i).x) ;
        case 'loss'
          res(i+1).x = vl_nnloss(res(i).x, l.class) ;
        case 'softmaxloss'
          res(i+1).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
        case 'relu'
          res(i+1).x = vl_nnrelu(res(i).x) ;
        case 'noffset'
          res(i+1).x = vl_nnnoffset(res(i).x, l.param) ;
        case 'dropout'
          if opts.disableDropout
            res(i+1).x = res(i).x ;
          elseif opts.freezeDropout
            [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i+1).aux) ;
          else
            [res(i+1).x, res(i+1).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
          end
        case 'custom'
          res(i+1) = l.forward(l, res(i), res(i+1)) ;
        otherwise
          error('Unknown layer type %s', l.type) ;
      end
      if opts.conserveMemory && ~doder && i < numel(net.layers) - 1
        % TODO: forget unnecesary intermediate computations even when
        % derivatives are required
        res(i).x = [] ;
      end
      res(i).time = toc(res(i).time) ;
    end


    l = net.layers{end};
    map = lcomb(res(n).x, l.filters, l.biases);
    map2 = imresize(map,[row,col]);
%     imwrite(map_layer2,['./resultsall/',hazy_data(ii).name,'_trainsm_layer2.png']);
% end
    %% final dehazing
    
    %Transmission post-precessing
    if adaptT == 1
        map = impt(img_n, map2, A);
    end
    map(map<0.1)=0.1;
    map(map>0.9)=0.9;
%     imwrite(map,['./resultsall/',hazy_data(ii).name,'_trainsm_layer2_post.png']);
%     for gamma = 8:17
     dehazedImageRGB = estimate_J(img_n, map, A, gamma); 
%         imwrite(dehazedImageRGB, ['./resultsall/',hazy_data(ii).name,'_dehazed_adaptT',num2str(gamma),'.png']);
%     end
    figure;subplot(221);imshow(map1);title('Transmission of Coarselayer');
    subplot(222);imshow(map2);title('Transmission of Finelayer');
    subplot(223);imshow(img0);title('hazy image');subplot(224);imshow(dehazedImageRGB);title('dehazed image');
% end
