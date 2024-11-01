import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, classification_report, mean_absolute_error

#Code for objective 1(the relationshup between rainfall and temperature to cocoa yield)

# Load the dataset
file = pd.read_excel(r"~/Desktop/Project_data/Excel/Ghana_data.xlsx")

data = file[['Year','Cocoa Yield', 'Rainfall', 'Temperature']].copy()  # Selecting columns of interest

# Convert 'Cocoa Yield' to categorical bins
data['Cocoa Yield Class'], bin_edges = pd.cut(file['Cocoa Yield'], bins=3, labels=[0, 1, 2], retbins=True)


# Separate features and target

X = data[['Rainfall', 'Temperature']]
y = data['Cocoa Yield Class']

    

#For r2 and r

X1 = data[['Rainfall']]
X2 = data[['Temperature']]

#The correlation between the parameters
correlation_matrix = data.corr()
print(correlation_matrix)

#Correlation matrix and plot
correlation_matrix = data.corr()
plt.figure(figsize=(8, 7))
hm = sbn.heatmap(correlation_matrix, annot=True, fmt=".3f", annot_kws={'size': 9}, cmap="Blues")
plt.title('Correlation Matrix')

plt.show()


# Instantiate and train the model
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=11)


classifier.fit(X, y)


# Make predictions

y_pred = classifier.predict(X)
y_pred_df = pd.DataFrame(y_pred, columns=['y_pred'])

# Evaluate the model
cm = confusion_matrix(y, y_pred)
accuracy_class = accuracy_score(y, y_pred)
print(f'Confusion Matrix : \n{cm}\n')
print(f'Accuracy : {accuracy_class:.5f}\n')



data['y_pred_df'] = y_pred_df  # Adding y_pred as a new column



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Assuming classifier is already trained
x_set, y_set = X.values, y.values
x1, x2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))

# Predict for each point in the meshgrid
Z = classifier.predict(np.array([x1.ravel(), x2.ravel()]).T)
Z = Z.reshape(x1.shape)

# Create the plot
plt.figure(figsize=(12, 6))
plt.contourf(x1, x2, Z, alpha=0.75, cmap=ListedColormap(('gray', 'green', 'purple')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

# Scatter plot of the test set
colors = ListedColormap(('gray', 'green', 'purple'))
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                color=colors(i), label=f'Class {j}')

plt.title('Random Forest Classifier (Test set)')
plt.xlabel('Rainfall')
plt.ylabel('Temperature')
plt.legend()

# Display the plot
plt.show()


# Do a linear regression model to find the r2 score 
model = LinearRegression()
#Find the r2 score of the model for your independents and dependent variables
model.fit(X,y) #Fit the model
r2_score1 = model.score(X,y)# Find the r2 score of Rainfall and Temperature to Cocoa Yield
print(f"The r2 score of Rainfall and Temperature to cocoa yield is {r2_score1:.5f}")
model.fit(X1, y)#Fit the model on the X1(Rainfall) and y(Cocoa Yield)
r2_score2 = model.score(X1,y)# Find the r2 score of rainfall and cocoa yield 
print(f"The r2 score of Rainfall to cocoa yield is {r2_score2:.5f}")
model.fit(X2, y)#Fit the model on the X2(Temperature) and y(Cocoa Yield)
r2_score3 = model.score(X2,y)# Find the r2 score of rainfall and cocoa yield 
print(f"The r2 score of Temperature to cocoa yield is {r2_score3:.5f}")




#Code for objective 2(the prediction for  cocoa yield using rainfall and temperature)

file.head()
file.info()
file.hist(figsize = (10, 8), bins = 50)
plt.show()



X = file[['Rainfall', 'Temperature']].values
y = file["Cocoa Yield"].values
#Scaling the parameters
y_sca = y.reshape(-1,1)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y_sca).ravel()

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.3, random_state=11)



# Define models to use 
svr = SVR(C=100000, epsilon=0.0001, kernel='linear')
mlr = LinearRegression()
rfr = RandomForestRegressor(n_estimators=100, random_state=10, max_features = 'sqrt', min_samples_split = 3, max_depth = 2)

clf_models = {
    "SVR": svr,
    "MLR": mlr,
    "RFR": rfr,
 }

#Begin the training and testing 
#A function for fitting the model and testing it
def model_fitting(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    MAE = mean_absolute_error(y_test, y_pred)
    accuracy = r2_score(y_test, y_pred)
    return accuracy, y_pred, MAE



# Train models and collect the test predictions

# For Training Set
print("\nFor the train set\n")
MAE_train_list = []
accuracy_scores_train = []
y_preds_list_train = []


for name, clf_model in clf_models.items():
    model_accuracy_train, y_preds_train, MAE_train = model_fitting(clf_model, X_train, y_train, X_train, y_train)
    print(f"Metrics for {name}")
    print(f"MAE: {MAE_train:.3f}")
    accuracy_scores_train.append(model_accuracy_train)
    y_preds_list_train.append((name, y_preds_train))

    # Feature names
    feature_names = ['Rainfall', 'Temperature']
    
    # Printing coefficients and intercepts for specific models
    if name == 'MLR':
        print(f"Coefficients: {clf_model.coef_}")
        print(f"Intercept: {clf_model.intercept_}")
    elif name == 'SVR':
        print(f"Coefficients: {clf_model.coef_}")
        print(f"Intercept: {clf_model.intercept_}")
    elif name == 'RFR':
        # Random Forest feature importance
        importances = clf_model.feature_importances_
        print("Feature Importances:")
        for feature, importance in zip(feature_names, importances):
            print(f"{feature}: {importance}")


# For Testing Set
print("\nFor the test set\n")
MAE_test_list = []
accuracy_scores_test = []
y_preds_list_test = []

for name, clf_model in clf_models.items():
    model_accuracy_test, y_preds_test, MAE_test = model_fitting(clf_model, X_train, y_train, X_test, y_test)
    print(f"Metrics for {name}")
    print(f"MAE: {MAE_test:.3f}")
    accuracy_scores_test.append(model_accuracy_test)
    y_preds_list_test.append((name, y_preds_test))


# Display model scores for the train set
model_scores_train = pd.DataFrame({"Algorithm": clf_models.keys(), "Accuracy for the train set": accuracy_scores_train}).sort_values("Accuracy for the train set", ascending=True)
print(model_scores_train.head(15))

# Display model scores for test set
model_scores_test = pd.DataFrame({"Algorithm": clf_models.keys(), "Accuracy for the test set": accuracy_scores_test}).sort_values("Accuracy for the test set", ascending=True)
print(model_scores_test.head(15))

#A graph of the prediction of the test set using matplotlib
import matplotlib.pyplot as plt

def plot_scatter_with_dynamic_limits_matplotlib(model_name, y_test, y_pred, title_suffix, file_path):
    # Determine dynamic axis limits based on y_test and y_pred
    min_value = min(y_test.min(), y_pred.min())  # Min value for axis limits
    max_value = max(y_test.max(), y_pred.max())  # Max value for axis limits
    
    # Create the scatter plot of actual vs. predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', s=50, label='Predicted vs Actual')
    
    # Add the diagonal line (where y_test == y_pred)
    plt.plot([min_value, max_value], [min_value, max_value], color='black', linewidth=2, label='Ideal Line (y=x)')
    
    # Customize the plot
    plt.title(f"{model_name} - Predicted vs Actual {title_suffix}")
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.xlim(min_value, max_value)  # Set the x-axis range dynamically
    plt.ylim(min_value, max_value)  # Set the y-axis range dynamically
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.legend()
    
    # Save the plot as an image file
    plt.savefig(file_path, format='png', bbox_inches='tight')  # Save as PNG with tight bounding box
    
    # Display the plot
    plt.show()
    
    # Close the figure to free memory
    plt.close()

# Example usage (assuming y_test and y_preds_test are available)
for model_name, y_preds_test in y_preds_list_test:
    file_path = f"{model_name}_predictions_vs_actual.png"  # Specify the file path
    plot_scatter_with_dynamic_limits_matplotlib(model_name, y_test, y_preds_test, "(Test Set)", file_path)
  




