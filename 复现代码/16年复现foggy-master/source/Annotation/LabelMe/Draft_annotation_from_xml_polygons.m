close all;
clear all;

class_ids = cityscapes_classnames2ids_eval();

% Read XML file with LabelMe annotation of a sample image.
% xml_file = '../../../../fog_20161213_081817/20161213_081817.xml';
xml_file = '../../../../data/Foggy_Zurich/Annotations/fog_20161213_081817/20161213_081817.xml';
x_doc = xmlread(xml_file);

% Retrieve image dimensions.
imagesize = x_doc.getElementsByTagName('imagesize');
assert(imagesize.getLength == 1);

nrows = imagesize.item(0).getFirstChild;
height = str2double(nrows.getTextContent);

ncols = nrows.getNextSibling;
width = str2double(ncols.getTextContent);

% Retrieve all polygons.
polygons_xml = x_doc.getElementsByTagName('polygon');
npolygons = polygons_xml.getLength;
polygons = cell(1, npolygons);

for i = 1:npolygons
    current_polygon_xml = polygons_xml.item(i-1);
    vertices_tmp = current_polygon_xml.getChildNodes;
    nvertices = vertices_tmp.getLength - 1;
    current_polygon_vertices = zeros(nvertices + 1, 2);
    for j = 1:nvertices
        % Index |vertices_tmp| with |j| instead of |j - 1|, because of the extra
        % first element node with name |user|.
        current_vertex = vertices_tmp.item(j);
        current_vertex_x = current_vertex.getFirstChild;
        x = str2double(current_vertex_x.getTextContent);
        y = str2double(current_vertex_x.getNextSibling.getTextContent);
        current_polygon_vertices(j, :) = [x, y];
    end
    % Add first vertex again in the last row to have a closed representation
    % which is preferable in MATLAB.
    current_vertex = vertices_tmp.item(1);
    current_vertex_x = current_vertex.getFirstChild;
    x = str2double(current_vertex_x.getTextContent);
    y = str2double(current_vertex_x.getNextSibling.getTextContent);
    current_polygon_vertices(nvertices + 1, :) = [x, y];
    polygons{i} = current_polygon_vertices;
end

% Boolean matrix indicating direction of pairwise polygon occlusions.
pairwise_occlusions = false(npolygons);

for i = 1:npolygons
    for j = i + 1:npolygons
        in_of_i_in_j = inpolygon(polygons{i}(1:end - 1, 1), polygons{i}(1:end - 1, 2),...
            polygons{j}(:, 1), polygons{j}(:, 2));
        in_of_j_in_i = inpolygon(polygons{j}(1:end - 1, 1), polygons{j}(1:end - 1, 2),...
            polygons{i}(:, 1), polygons{i}(:, 2));
        % Assumption: if polygon i has more vertices inside polygon j than vice
        % versa, then polygon i occludes polygon j.
        pairwise_occlusions(i, j) = nnz(in_of_i_in_j) >= nnz(in_of_j_in_i);
        pairwise_occlusions(j, i) = ~pairwise_occlusions(i, j);
    end
end

% Create pixel grid for given dimensions.
[X, Y] = meshgrid(1:width, 1:height);
pixels_X = X(:);
pixels_Y = Y(:);

% Initialize vector with polygon IDs to 0.
polygon_ids = zeros(size(pixels_X, 1), 1);

% Loop over all polygons and assign each pixel the ID of the polygon which is
% the closest to the camera (using LabelMe assumption for pairwise occlusions)
% out of those polygons that include the pixel.
for i = 1:npolygons
    in_i = inpolygon(pixels_X, pixels_Y, polygons{i}(:, 1), polygons{i}(:, 2));
    in_i_ids = polygon_ids(in_i);
    existing_ids = unique(in_i_ids);
    
    for j = existing_ids.'
        if j == 0
            % Assign all unoccupied pixels to the current polygon i.
            in_i_ids(in_i_ids == j) = i;
        else
            % Otherwise, check pairwise occlusion.
            if pairwise_occlusions(i, j)
                % Only when i occludes (i.e. is in front of) j, do assign to i
                % pixels that had been previously assigned to j.
                in_i_ids(in_i_ids == j) = i;
            end
        end
    end
    polygon_ids(in_i) = in_i_ids;
end

polygon_ids = reshape(polygon_ids, height, width);

% Present result.
figure;
imshow(polygon_ids, []);

% Resize, normalize range and save.
polygon_ids_small = imresize(polygon_ids, 0.25, 'bilinear');
polygon_ids_small = polygon_ids_small / npolygons;
% imwrite(polygon_ids_small, '../../../../fog_20161213_081817/Polygon_image.png');
