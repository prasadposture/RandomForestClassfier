#importing requied libraries
import pandas as pd
import numpy as np
import warnings
print('Libraries Imported') #making sure the libraries are imported
#filtering the warnings cuz presence of warnigs would make the output messier
warnings.filterwarnings('ignore')


#loading the train data (with user input)
train_data=input('Enter the train data :')
#for application this has to be like 'upload your file' kinda option
train=pd.read_csv(train_data)

#loading the test data (with user input)
test_data=input('Enter the test data :')
test=pd.read_csv(test_data)

#loading the sample submission to save our predictions
#the sample submissions is given all the time along with test and train data
#but in case it isnt given we need to put it in try-except block
#we will have to make changes in the saving method as well
#new comment : assuming that the sample submission is already given
#cause it makes things lot easier and this program is written while considering the kaggle competetions
#where the sample submission is always given
sample_data=input('Enter the sample submission :')
sample=pd.read_csv(sample_data)
print('Data Loaded!') #making sure the data is loaded

#getting stastical information of the numeric variables
#storing it into a dataframe for using it later for filling missing values and 
#seperating numeric columns form categorical columns
train_describe=pd.DataFrame(train.describe())

#doing the same for test data
test_describe=pd.DataFrame(test.describe())

#describing with category
#we will put it in try...except block since some of the data many not have categorical attributes
#saving the stastical information will be useful while filling the missing values
try:
    train_describe_cat=pd.DataFrame(train.describe(include=['O']))
    test_describe_cat=pd.DataFrame(test.describe(include=['O']))
except Exception as e:
    pass

#We will fill these missing values -
#using median for numerical data
#using top for categorical data

#getting the list of numeric columns from stastical description
train_describe_columns=list(train_describe.columns)
test_describe_columns=list(test_describe.columns)

#getting the list of categorical columns from the stastical description
try:
    train_describe_cat_columns=list(train_describe_cat.columns)
    test_describe_cat_columns=list(test_describe_cat.columns)
except:
    pass

#filling the missing values for numerical values with the median
for i in train_describe_columns:
    train[i].fillna(train[i].median(), inplace=True)
for i in test_describe_columns:
    test[i].fillna(test[i].median(), inplace=True)
    
#filling the missing values for categorical values with the mostly reccurring value
try:
    for i in train_describe_cat_columns:
        train[i].fillna(train_describe_cat.iloc[2,train_describe_cat_columns.index(i)], inplace=True)
    for i in test_describe_cat_columns:
        test[i].fillna(test_describe_cat.iloc[2,test_describe_cat_columns.index(i)], inplace=True)
except:
    pass
print('We filled the missing values!') #making sure the missing values are filled

#setting ID's
#We will put it in try except block since sometimes we dont have seperate data
try:
    for i in list(train.columns):
        if 'Id' in i: #using the fact that infact Id have string 'Id' in their name eg PassengerID.
            Id=i
            print('The Id of data is :', Id)
        else:
            pass
    train.set_index(Id ,inplace=True)
    test.set_index(Id ,inplace=True)
except:
    pass

#we will drop the categorical columns which contain unique values with proportion 0.1 or more 
#to that of total value count
try:
    if Id in train_describe_cat_columns:
        train_describe_cat_columns.remove(Id)
    columns_to_be_dropped=[]
    for i in train_describe_cat_columns:
        a=test_describe_cat.iloc[1,test_describe_cat_columns.index(i)]/test_describe_cat.iloc[0,test_describe_cat_columns.index(i)]
        a=float(a)
        if a>0.1:
            columns_to_be_dropped.append(i)
    train.drop(columns=columns_to_be_dropped, inplace=True)
    test.drop(columns=columns_to_be_dropped, inplace=True)
    train_describe_cat=pd.DataFrame(train.describe(include=['O']))
    test_describe_cat=pd.DataFrame(test.describe(include=['O']))
    train_describe_cat_columns=list(train_describe_cat.columns)
    test_describe_cat_columns=list(test_describe_cat.columns)
    print('Columns to be dropped :',columns_to_be_dropped)
    
#Label Encoding
#Labelling the categorical values with numbers so that machine could understand it
#putting it try-except block because we may not have categorical values

    from sklearn.preprocessing import LabelEncoder
    for i in train_describe_cat_columns:
        le=LabelEncoder()
        arr=np.concatenate((train[i], test[i])).astype(str)
        le.fit(arr)
        train[i]=le.transform(train[i].astype(str))
        test[i]=le.transform(test[i].astype(str))
except:
    pass

#Getting the target variable aka the variable we gonna predict
#here we are using the fact that the target variable wouldn't be present in the test data
a=set(train.columns)
b=set(test.columns)
c=list(a-b)
label=c[0]
print('Target variable is :', label) #showing the target variable

#assiging dependent and independent variables
X=train.drop(label,axis=1)
y=train[label]

#train_test_split()
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=4)

#Modeling
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
pred_y=rf.predict(X_val)
from sklearn.metrics import accuracy_score
print('The accuracy is :',accuracy_score(y_val.values,pred_y))

#making predictions on test data
y_pred=rf.predict(test)

#saving the submissions in the form of pandas dataframe
#submission=pd.DataFrame({label:y_pred},index=test.index) : freq used in older versions
sample_columns=list(sample.columns)
sample[sample_columns[1]]=y_pred
file_name=input('Enter the title of submission :')
sample.to_csv(file_name)
print('SUBMISSIONS SAVED SUCCESSFULLY!!!')
sample.head()#making sure that the submission is saved.