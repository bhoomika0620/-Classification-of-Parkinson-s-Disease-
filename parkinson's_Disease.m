
## Load dataset
data = readtable('Parkinsson_disease.xls');

## Separate features and target
X = data{:,1:end-1};   % All columns except last
y = data{:,end};       % Last column is status (0 = healthy, 1 = Parkinson's)

## Split data into training and testing sets (70/30)
cv = cvpartition(size(X,1),'HoldOut',0.3);
idx = cv.test;

X_train = X(~idx,:);
y_train = y(~idx,:);
X_test  = X(idx,:);
y_test  = y(idx,:);

## Support Vector Machine (SVM)
svmModel = fitcsvm(X_train, y_train, 'KernelFunction','rbf','Standardize',true);
y_pred_svm = predict(svmModel, X_test);

## Evaluate SVM
confMat_svm = confusionmat(y_test, y_pred_svm);
accuracy_svm = sum(y_pred_svm == y_test) / length(y_test) * 100;

## K-Nearest Neighbors (KNN)
knnModel = fitcknn(X_train, y_train, 'NumNeighbors',5);
y_pred_knn = predict(knnModel, X_test);

## Evaluate KNN
confMat_knn = confusionmat(y_test, y_pred_knn);
accuracy_knn = sum(y_pred_knn == y_test) / length(y_test) * 100;

## Display Results
fprintf('SVM Accuracy: %.2f%%\n', accuracy_svm);
fprintf('KNN Accuracy: %.2f%%\n', accuracy_knn);

disp('Confusion Matrix - SVM:');
disp(confMat_svm);

disp('Confusion Matrix - KNN:');
disp(confMat_knn);
