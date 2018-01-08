function [map, subratio, indices, counts, pdf] = ...
    gensubmap (N, type, val, shift, mode)
%Genearte semi-random sub-sampling map (logical).
%
% N
%   The size of the map or the signal iteself. If a scalar is given then 
%   assume [N N]. For vectors use the usual [1 n] or [n 1]. If a vector or
%   matrix of actual values is given (useful for TYPE='ORACLE'), then 
%   size(N) will be used.
%
% TYPE
%   For 2D maps, TYPE can be:
%   - An N-by-N matrix which will be used as probability weights
%     (distribution). N must equal N=size(TYPE).
%   - A function handle @(X,Y) used to generate an N-by-N matrix of 
%     probability weights (distribution). Its domain must be [0,1)^2 and will
%     generate the matrix of weights by evaluating it at all (X,Y) = (k/N,j/N)
%     with j,k = 0,...,N-1.
%   - String 'RANDOM', uniform random (same as 'squares', [1; val])
%   - String 'SQUARES', multi-level squares in the center, fftshift'ed
%   - String 'CORNERSQUARES', same but in the upper-left corner
%   - String 'CIRCLES', multi-level circles in the center, fftshift'ed
%   - String 'CORNERCIRCLES', same but in the upper-left corner
%   - String 'RADIAL', radial lines in the center
%   - String 'CORNERRADIAL', same but in the upper-left corner
%   - String 'HALFHALF', a half-half scheme in the center
%   - String 'CORNERHALFHALF', a half-half scheme in the upper-left corner
%   - String 'ZIGZAG', JPEG like, from upper left corner, without 8x8 blocks
%   - String 'ORACLE', oracle thresholding map, N must be a 2D matrix of coefs
%   
%   For 1D maps, TYPE can be:
%   - An N long vector which will be used as probability weights
%     (distribution). N must equal N=length(TYPE).
%   - A function handle @(X) used to generate an N long vector of probability
%     weights (distribution). Its domain must be [0,1) and will generate the
%     vector of weights by evaluating it at all X = k/N, with k = 0,...,N-1.
%   - String 'RANDOM|RAND|RND', uniform random (same as 'squares', [1; val])
%   - String 'SQUARES|SQ', multilevel squares in the center
%   - String 'CORNERSQUARES|CSQ', multilevel squares on the left side
%   - String 'HALFHALF|HH', a half-half scheme in the center
%   - String 'CORNERHALFHALF|CHH', a half-half scheme on the left side
%   - String 'ORACLE', an oracle thresholding map, N must be a vector of coefs.
%
%   All 1D string maps (Except oracle) can be posfixed by 'POISSON|POISS' to
%   take the random samples using a Poisson disc restriction. TODO: Implement
%   this for 2D as well.
%
% VAL
%   - If TYPE='RANDOM' then VAL must be a fraction or natural number
%     representing the subsampling fraction or number of samples to take.
%
%   - If TYPE='SQUARES|CORNERSQUARES|CIRCLES|CORNERCIRCLES' then VAL
%     must be a 2-row matrix where the 1st row represents the boundaries of
%     the squares/circles and the second contains the fraction of random 
%     samples to take in that region bounded by the squares/circles. For 
%     example, [0.1, 0.25, 0.5, 1  ;  1, 0.5, 0.1, 0.01] would generate
%     concentric squares/circles of diameters 0.1, 0.25, 0.5 of the total
%     length. If the last element is 1 then takes the entire square (as 
%     above). The second row of fractions 1, 0.5, 0.1, 0.01 would take full
%     sampling in the center square/circle of 0.1 diameter, 50% random 
%     sampling in the ring defined by the squares/circles of 0.1 and 0.25
%     diameter, 10% in the ring defined by the squares/circles of 0.25 and
%     0.5 diameter, and 1% outside the square/circle of 0.5 diameter (up to
%     the total square).
%
%   - If TYPE='RADIAL|CORNERRADIAL' then VAL must be a number number of lines
%     to take radially.
%
%   - If TYPE='HALFHALF|CORNERHALFHALF' then VAL is either a fraction (<=1)
%     or a natural > 1 which will be split in two, half in the center fully
%     sampled, and half uniformly random in the rest of the map.
%
%   - If TYPE='ORACLE' then VAL is either a fraction (<=1) or a natural > 1,
%     which will select the VAL largest values in X (in absolute value).
%
%   - IF TYPE is a function_handle or matrix/vector of weights, then VAL is
%     the fraction (<=1) or number (>1) of samples to take. Examples:
%     * GENSUBSAMPLMAP(1024, @(x,y)(1-sqrt((x^2+y^2)/2))^4, 0.25)
%       builds a 1024-by-1024 2D map subsampled 25% with decreasing density
%       from the upper-left corner.
%     * GENSUBSAMPLMAP(1024, @(x,y)(1-sqrt(((2*x-1)^2+(2*y-1)^2)/2))^4, 0.25)
%       builds a 1024-by-1024 2D map subsampled 25% with decreasing density
%       from the center.
%     * X=1./(1:1024); GENSUBSAMPLMAP([1 1024], X, 256)
%       builds a 1-by-1024 1D map with 256 samples taken semi-randomly
%       according to the weights specified by the matrix X which assignes a
%       weight of 1/k for index k.
%
% SHIFT
%   Whether to FFTSHIFT the map (boolean).
%
% MODE
%   A string with possible values 'normal', 'search', 'fill'. The 'search'
%   value just causes execution to terminate early in order to speed up the
%   code when the function is run in a loop. See GENSUBMAPSEARCH().

global debug;
if nargin < 4, shift = false; end % fftshift the map?
if nargin < 5, mode = 'normal'; end % either: normal, search, fill

mapsource = [];
if ~ischar(type) && ~isa(type,'function_handle'), N = size(type);
elseif length(N) < 2, N = [N N];
elseif length(N) > 2, mapsource = N; N = size(N);
end

if ismatrix(val) && size(val,1) == 2
    lengths = [0 val(1, :)];
    fractions = [0 val(2, :)];
    counts = zeros(1,length(lengths)-1);
elseif isscalar(val) && val <= 1
    val = round(val * prod(N));
    counts = val;
end

switch mode
    case 'fill'
        sample = @samplefill;
    otherwise
        sample = @samplenormal;
end

n = 0;
map = false(N);
pdf = zeros(N);
if ischar(type), type = lower(type); end

if isa(type,'function_handle') % type=function to generate prob weights
    m = numel(map);
    %weights = zeros(N);
    func = func2str(type);
    func = strrep(func, '/', './');
    func = strrep(func, '*', '.*');
    func = strrep(func, '^', '.^');
    func = str2func(func);
    if isvector(map)
        weights = func(0:1/m:1-1/m); % [0,1) interval
        %for w=1:m, weights(w) = type((w-1)/m); end
    else
        [x,y] = meshgrid(0:1/N(1):1-1/N(1), 0:1/N(2):1-1/N(2)); % [0,1)^2 space
        weights = func(x,y);
        %for w1=1:N(1)
        %    for w2=1:N(2)
        %        weights(w1,w2) = type((w1-1)/N(1),(w2-1)/N(2));
        %    end
        %end
    end
    indices = datasample(1:m, val, 'weights', weights(:), 'replace', false);
    map(indices) = true;

elseif ~ischar(type) % type = matrix of prob weights
    indices = datasample(1:numel(map), val, 'weights', type(:), 'replace', false);
    map(indices) = true;

elseif regexp(type, '^random')
    if regexp(type,'poiss')
        if isvector(map)
            indices = samplepoisson1d(1, prod(N), val);
        else
            % 2D poisson not yet implemented
            indices = randsample(prod(N), val, false);
        end
    else
            indices = randsample(prod(N), val, false);
    end
    map(indices) = true;

elseif regexpi(type, '^(corner)?radial')
    errvector();
    if regexp(type,'^corner')
        N = N(1);
        lines = val;
        iscorner = true;
    else
        N = N(1)/2;
        lines = val/2;
        iscorner = false;
    end        
    map = false([N+1 N+1]);
    map(1,:) = true;
    for k=1:lines/2
        a = pi/2/lines*k;
        x=1+round(linspace(0,N*tan(a),N+1));
        y=1:N+1;
        indices=sub2ind([N+1, N+1], x, y);
        map(indices)=true;
    end;
    map = map | map';
    if ~iscorner
        map = [
            rot90(map(2:end,2:end),2) rot90(map(1:end-1,2:end),1)
            rot90(map(2:end,1:end-1),3)   map(1:end-1,1:end-1)
        ];
    else
        map = map(1:end-1,1:end-1);
    end
    n = nnz(map);

elseif regexpi(type, '^(corner)?circles')
    errvector();
    if regexp(type,'^corner')
        m = max(N);
        % coords of all CORNER-circles
        y = repmat(0:m-1, [m 1]);
        iscorner = true;
    else
        % prepare x^2+y^2 for all x, y in [-rad,rad]
        % same as doing [x, y] = meshgrid(0.5-mid:mid-0.5, 0.5-mid:mid-0.5);
        % y = x.*x + y.*y; but uses only half the memory
        m = max(N) / 2;
        y = repmat(0.5-m:m-0.5, [2*m 1]);
        iscorner = false;
    end
    y = y .* y;
    y = y + y';
    for k=2:length(lengths)
        rad1 = round(lengths(k-1) * m);
        if lengths(k) < 1 % area between circles of radiuses rad1 and rad2
            rad2 = round(lengths(k) * m);
            zone = y<rad2*rad2 & y>=rad1*rad1;
        else  % area between circle of radius rad 1 and big square
            zone = y>=rad1*rad1;
        end
        pdf(zone) = fractions(k);
        if ~isempty(mapsource), zone = zone & mapsource; end
        indices = find(zone);
        %zones{k-1} = indices;
        nrand = round(fractions(k) * numel(indices));
        counts(k-1) = nrand;
        sample(indices, nrand);
        n = n + nrand;
    end
    %if ~iscorner, map=fftshift(map); end

elseif regexpi(type, '^(corner)?squares')
    if regexp(type,'^corner')
        m = max(N);
        iscorner = true;
        if isvector(map) % 1D map
            x = 0:m-1;
            y = x;
        else
            [x,y] = meshgrid(0:m-1, 0:m-1);
        end
    else
        m = max(N) / 2;
        iscorner = false;
        if isvector(map) % 1D map
            x = abs(0.5-m:m-0.5);
            y = x;
        else % 2D map
            [x,y] = meshgrid(0.5-m:m-0.5, 0.5-m:m-0.5);
            x = abs(x);
            y = abs(y);
        end
    end
    for k=2:length(lengths)
        if lengths(k-1) > 1
            lo = lengths(k-1);
            if lengths(k) < max(N)
                hi = lengths(k);
                indices = find(y<hi & x<hi & (y>=lo | x>=lo));
            else
                indices = find(y>=lo | x>=lo);
            end
        else
            lo = round(lengths(k-1) * m);
            if lengths(k) < 1 % area between squares lo and hi
                hi = round(lengths(k) * m);
                indices = find(y<hi & x<hi & (y>=lo | x>=lo));
            else  % area between square lo and big square
                indices = find(y>=lo | x>=lo);
            end
        end
        nrand = round(fractions(k) * numel(indices));
        counts(k-1)=nrand;
        if isvector(map)
            if regexp(type,'poiss')
                map(indices(samplepoisson1d(1,length(indices),nrand)))=true;
            else
                map(indices(randsample(length(indices),nrand,false)))=true;
            end
        else
            sample(indices, nrand);
        end
        n = n + nrand;
    end
    %if ~iscorner, map=fftshift(map); end
    
elseif regexpi(type, '^(corner)?halfhalf')
    iscorner = strcmpi(type(1),'c');
    if isvector(map)
        m = round(val/2);
        map(1:m) = true;
        nrest = val - m;
        %indices = datasample(find(~map), nrest, 'replace', false); % pick them at random, without replacement
        if regexp(type,'poiss')
            indices = samplepoisson1d(m+1, prod(N), nrest);
        else
            indices = datasample(m+1:prod(N), nrest, 'replace', false);
        end
        map(indices) = true;
        n = numel(indices);
        if ~iscorner, map = circshift(map, ...
                iif(isrow(map),[0 -round(m/2)],[-round(m/2) 0])); end
    else
        m = round(sqrt(val / 2));
        map(1:m,1:m) = true;
        nrest = val - m^2;
        indices = datasample(find(~map), nrest, 'replace', false); % pick them at random, without replacement
        map(indices) = true;
        n = numel(indices);
        if ~iscorner, map = circshift(map, [-round(m/2) -round(m/2)]); end
    end
    
elseif regexpi(type, '^zigzag')
    errvector();
    n = val;
    N = N(1);
    lporder = bdct_linapprox_ordering(N, N);
    indices = lporder(1:n);
    map = false([N N]);
    map(indices) = true;
    
elseif regexpi(type, '^oracle')
    n = val;
    [~, indices] = sort(abs(N(:)), 'descend');
    indices = indices(1:n);
    map = false(size(N));
    map(indices) = true; 
    
else
    error('SSM:TYPE', 'Unknown subsample type SUBTYPE=''%s''. See help %s.', type, upper(mfilename));
end        

subratio = n / numel(map);

if strcmp(mode,'search')
    indices = []; return;
end

if debug;
    fprintf('Subsampling = %.5f%%\n', subratio * 100)
    f=figure;
    imshow(map);
    dlg.Default = 'No';
    dlg.Interpreter = 'none';
    choice = questdlg('Continue?','Question','Yes','No',dlg);
    close(f);
    pause(0.5);
    if strcmp(choice, 'No'), indices=-1; return; end
end

% fftshift the map?
if shift
    map = fftshift(map);
end

indices = find(map);



%%  Internal helper functions  %%


function samplefill (ind, varargin)
    map(ind) = true;
end

function samplenormal (ind, n)
    ind = datasample(ind, n, 'Replace', false); % pick them at random, without replacement
    map(ind) = true;
end

function disc = genfilleddisc(r)
    disc = false(N/2);
    x_ = 1:r;
    y_ = floor(sqrt((r-1)^2-(x_-1).^2))+1;
    for k_=1:r; disc(x_(k_),1:y_(k_))=true; end
    disc = [rot90(disc,2) rot90(disc,1); rot90(disc,3) disc];
end

function errvector()
    if isvector(map)
        error('SSE:TYPE','Invalid TYPE for 1D map. See HELP %s.',mfilename);
    end
end
end
