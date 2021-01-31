% Step 1:
% Download the wheel file from https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/list.html
% (see https://xgboost.readthedocs.io/en/latest/build.html)

% Step 2:
% Set the xgboost_install_dir and wheel_fn in xgboost_install.m

% Step 3:
% run xgboost_install

%xgboost_install_dir = 'd:\cpcardio\physionet_2020\xgboost';
xgboost_install_dir = 'F:\MATLAB2020b\MachineLearning\xgboost\lib';
wheel_fn = 'F:\MATLAB2020b\MachineLearning\xgboost-1.2.0_SNAPSHOT+4729458a363c64291e84da28b408a0ac8d7851fa-py3-none-win_amd64.whl';

%

mkdir(xgboost_install_dir);

xgboost_install_dir_tmp = [xgboost_install_dir '\' 'tmp'];
mkdir(xgboost_install_dir_tmp);

cd(xgboost_install_dir);
%url = 'https://s3-us-west-2.amazonaws.com/xgboost-nightly-builds/release_1.1.0/xgboost-1.1.0%2B115e4c33608c3b0cee75402f1193e67fdb11ef9a-py3-none-win_amd64.whl';
%filename = 'xgboost.whl';
%outfilename = websave(filename,url);

wheel_fn = 'F:\Matlab\toolbox\MachineLearning\xgboost-1.2.0_SNAPSHOT+4729458a363c64291e84da28b408a0ac8d7851fa-py3-none-win_amd64.whl';
unzip(wheel_fn, xgboost_install_dir_tmp);

from = [xgboost_install_dir_tmp '\xgboost\lib\xgboost.dll'];
to = [xgboost_install_dir '\' 'xgboost.dll'];
movefile(from, to);

FileList = dir(fullfile(xgboost_install_dir_tmp, '**', 'vcomp140.dll'));

from = [FileList(1).folder '\' FileList(1).name];
to = [xgboost_install_dir '\' FileList(1).name];
movefile(from, to);

rmdir(xgboost_install_dir_tmp, 's');

url = 'https://raw.githubusercontent.com/dmlc/xgboost/master/include/xgboost/c_api.h';
filename = [xgboost_install_dir 'xgboost.h'];
outfilename = websave(filename,url);



