function UCR_time_series_test 
TRAIN =load('ShapesAll_TRAIN.tsv'); 
TEST = load('ShapesAll_TEST.tsv' ); 
TRAIN_class_labels = TRAIN(:,1); 
TRAIN(:,1) = []; 
TEST_class_labels = TEST(:,1); 
TEST(:,1) = [];
correct = 0; 
for i = 1 : length(TEST_class_labels) 
    classify_this_object = TEST(i,:);
    this_objects_actual_class = TEST_class_labels(i);
    predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels,classify_this_object);
    if predicted_class == this_objects_actual_class
        correct = correct + 1;
    end
    disp([int2str(i), ' out of ', int2str(length(TEST_class_labels)), ' done']) % Report progress
end
disp(['The error rate was ',num2str(1-correct/length(TEST_class_labels))])

function predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels,unknown_object)
best_so_far = inf;
for i = 1 : length(TRAIN_class_labels)
    compare_to_this_object = TRAIN(i,:);
    
    %distance = sqrt(sum((compare_to_this_object - unknown_object).^2)); % Euclidean distance
    %distance = DTW(compare_to_this_object,unknown_object); % DTW distance
    %distance = DDTW(compare_to_this_object,unknown_object); % DDTW distance
    %distance = LEDTW(compare_to_this_object,unknown_object); % LEDTW distance
    %distance = WDTW(compare_to_this_object,unknown_object); % WDTW distance
    %distance = TWDTW(compare_to_this_object,unknown_object); % TDTW distance
    distance = EWDTW(compare_to_this_object,unknown_object); % meDTW distance
   
    if distance < best_so_far
        predicted_class = TRAIN_class_labels(i);
        best_so_far = distance;
    end
end


function dist_LEDTW = LEDTW(X,Y)
[X_min,X_max]=M_m_extreme(X);
[Y_min,Y_max]=M_m_extreme(Y);
dist_min=DTW(X_min,Y_min);
dist_max=DTW(X_max,Y_max);
dist_LEDTW=dist_min+dist_max;
function [X_min,X_max]=M_m_extreme(X)
X_min=X(1);
X_max=X(1);
for i=2:(length(X)-1)
    if  (X(i)<X(i-1) && X(i)<=X(i+1))||(X(i)<=X(i-1) && X(i)<X(i+1))
        X_min=cat(1, X_min, X(i));
    elseif (X(i)>X(i-1) && X(i)>=X(i+1))||(X(i)>=X(i-1) && X(i)>X(i+1))
        X_max=cat(1, X_max, X(i));
    end
end
X_min=cat(1, X_min, X(end));
X_max=cat(1, X_max, X(end));


function dist_DDTW = DDTW(X,Y)
D_X=X;D_Y=Y;
for i=2:(length(X)-1)
    D_X(i)=((X(i)-X(i-1))+(X(i+1)-X(i-1))/2)/2;
end
for j=2:(length(Y)-1)
    D_Y(j)=((Y(j)-Y(j-1))+(Y(j+1)-Y(j-1))/2)/2;
end
D_X(1)=D_X(2);D_X(end)=D_X(length(X)-1);
D_Y(1)=D_Y(2);D_Y(end)=D_Y(length(Y)-1);
dist_DDTW=DTW(D_X,D_Y);


function dist_DTW= DTW(X,Y)
n=length(X);
m=length(Y);
D=1./zeros(n+1,m+1);
D(1,1)=0;
for i=2:n+1
    for j=2:m+1
        D(i,j)=(X(i-1)-Y(j-1))^2+min([D(i-1,j-1),D(i-1,j),D(i,j-1)]);
    end
end
D=D(2:n+1,2:m+1);
dist_DTW=sqrt(D(n,m));


function dist_meDTW = EWDTW(X,Y)
X_extreme = extreme(X);
Y_extreme = extreme(Y);
X_value = X_extreme(:,1);X_place = X_extreme(:,2);X_type = X_extreme(:,3);
Y_value = Y_extreme(:,1);Y_place = Y_extreme(:,2);Y_type = Y_extreme(:,3);
n=length(X_value);
m=length(Y_value);
D=1./zeros(n+1,m+1);
L=length(X);
D(1,1)=0;
for i=2:n+1
    for j=2:m+1
        D(i,j)=2/(1+exp(-8*(abs(X_place(i-1)-Y_place(j-1))/L*(1-0.2*X_type(i-1)*Y_type(j-1))-0.5)))*abs(X_value(i-1)-Y_value(j-1))+min([D(i-1,j-1),D(i-1,j),D(i,j-1)]);
    end
end
D=D(2:n+1,2:m+1);
dist_meDTW=D(n,m);
function X_extreme = extreme(X)
X_extreme=[X(1),1,0];
n=length(X);
for k=2:n-1
    if  (X(k)<X(k-1) && X(k)<=X(k+1))||(X(k)<=X(k-1) && X(k)<X(k+1))
        X_extreme=cat(1, X_extreme, [X(k),k,1]);
    elseif (X(k)>X(k-1) && X(k)>=X(k+1))||(X(k)>=X(k-1) && X(k)>X(k+1))
        X_extreme=cat(1, X_extreme, [X(k),k,-1]);
    end
end
X_extreme=cat(1, X_extreme, [X(n),n,0]);


function dist_TDTW= TWDTW(X,Y)
n=length(X);m=length(Y);
% X=fliplr(X);X=fliplr(X);
x=zeros(1,n);y=zeros(1,m);
D=1./zeros(n+1,m+1);D(1,1)=0;
for i=2:n+1
    for j=2:m+1
        x(i-1)=1-(i-1)/n;y(j-1)=1-(j-1)/m;
        D(i,j)=(X(i-1)-Y(j-1))^2*sqrt((x(i-1)^2+y(j-1)^2)/2)+ min([D(i-1,j-1),D(i,j-1),D(i-1,j)]);
    end
end
D=D(2:n+1,2:m+1);
dist_TDTW=D(n,m);


function dist_DTW= WDTW(X,Y)
n=length(X);
m=length(Y);
D=1./zeros(n+1,m+1);
D(1,1)=0;
for i=2:n+1
    for j=2:m+1
        D(i,j)=2/(1+exp(-0.25*(abs(i-j)-n/2)))*(X(i-1)-Y(j-1))^2+min([D(i-1,j-1),D(i-1,j),D(i,j-1)]);
    end
end
D=D(2:n+1,2:m+1);
dist_DTW=sqrt(D(n,m));
