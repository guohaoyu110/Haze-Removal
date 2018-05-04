fname = 'lindau_000004_000019_camera.json';
fid = fopen(fname);
raw = fread(fid,inf);
str = char(raw');
fclose(fid);

data = JSON.parse(str);