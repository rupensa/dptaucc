pkg load miscellaneous
pkg load statistics

%% global variables that need to be set before calling the main function clustering
global range;
global side_length;

%% example 2: clustering MNIST data; please download the data first

arg_list = argv ();
filename = arg_list{1};
load(filename);

% mnist.mat should contain a matrix called mnist, which should be a d*n matrix, where each column is a data point
k=double(file_k);
d=double(file_d);
n=double(file_n);
n_iter=file_n_iter;
max_val = double(file_maxval);
dataset = file_outname;
%eps=arg_list{2}
eps = double(file_eps);
delta=0.1;
range=max_val*sqrt(d);
%range=max_val;
side_length=max_val*2;
[ z_centers,clusters,u_centers,c_candidates,L_loss ]=clustering(file_data,n,d,k,eps,delta,n_iter);

%for i = 1:rows(clusters)
%  label = clusters(i,1);
%  for j = 1:columns(label{1})
%    row_label(j)=i;
%  endfor
%endfor


outname = [dataset "_labels" ".mat"];
save -mat ../../out_labels.mat clusters;

outname = [dataset "_loss" ".mat"];
save -mat ../../out_loss.mat L_loss;


