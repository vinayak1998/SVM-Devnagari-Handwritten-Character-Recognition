# SVM-Devnagari-Handwritten-Character-Recognition
Solving the Character recognition problem as an SVM optimization problem using CVXOPT

##Support Vector Machine

**Name**

svm - Run the executable program for SVM

**Synopsis**
*Part a*`./svm <part> <tr> <ts> <out> <c_value>`
*Part b*`./svm <part> <tr> <ts> <out> <c_value> <gamma>`
*Part c*`./svm <part> <tr> <ts> <out> <c_value> <gamma>`

**Description**

This program will train svm model using given code on train data, make predictions on test data and write final predictions in given output file.

**Options**

-part  
    Part  i.e. a/b/c.  
-tr  
    File containing training data in csv format where 1st entry is the target  
-ts  
    File containing test data in csv format where 1st entry is the target  
-out  
    Output file for predictions. One value in each line.
-c_value  
    C is a regularization parameter that controls the trade-off between maximizing the margin and minimizing the training error.    
-gamma  
    Bandwidth parameter for RBF kernel

**Example**
    
`./svm a DHC_train.csv DHC_test.csv output 10`
`./svm b DHC_train.csv DHC_test.csv output 10 0.01`
`./svm c DHC_train.csv DHC_test.csv output 10 0.01`

### Parts

- **Part A**
    - Expressed the SVM dual problem using a linear kernel
    - Soft Margin formulation
    
- **Part B**
    - Solved the dual SVM Problem using a RBF Kernel(Gaussian Kernel)
  
- **Part C**
    - Implemented PCA algorithm using the SVD formulation.
    - Applied SVM with RBF Kernel on the projected data
    
**Data**

- DHC_train.csv: Train data  
- DHC_test.csv: Test data