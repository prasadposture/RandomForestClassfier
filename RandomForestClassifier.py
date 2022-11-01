#importing requied libraries
import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
#Setting WebApp title and Writing about the application
st.set_page_config(page_title='RandomForestClassifier WebApp', page_icon=':deciduous_tree:')
st.title('Random Forest Classifier')
st.write("___")
st.write('This web app is specifically made for the datasets given in kaggle competitions. '
         'All you need to do is upload the required files and hit the predict button. '
         'The predictions will be readily available for you to download. '
         'You can submit them directly to kaggle.')

#loading the  data (with user input)
train_data=st.file_uploader("Choose the train data :")
test_data=st.file_uploader("Choose the test data :")
sample_data=st.file_uploader('Choose the sample submission :')

#All the procedure is to be done if Predict button in pressed
if st.button('Predict'):
    train = pd.read_csv(train_data) #reading the train data
    test = pd.read_csv(test_data) #reading the test data
    sample=pd.read_csv(sample_data) #reading the sample data
    
    #stastical information about numeric columns of the train and test dataset
    train_describe=pd.DataFrame(train.describe())
    test_describe=pd.DataFrame(test.describe())
    
    #stastical information about the categorical columns of the train and test dataset
    #dataset may not have categorical attributes that is why putting this into try-except block
    try:
        train_describe_cat=pd.DataFrame(train.describe(include=['O']))
        test_describe_cat=pd.DataFrame(test.describe(include=['O']))
    except Exception as e:
        pass

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

    #setting ID's
    #We will put it in try except block since sometimes we dont have seperate data
    try:
        for i in list(train.columns):
            if 'Id' in i: #using the fact that infact Id have string 'Id' in their name eg PassengerID.
                Id=i
                st.write('Id :', Id)
            else:
                pass
        train.set_index(Id ,inplace=True)
        test.set_index(Id ,inplace=True)
    except:
        pass

    #dropping the columns based on the proportion of the unique values to the total values
    
    try:
        if Id in train_describe_cat_columns:
            train_describe_cat_columns.remove(Id)
        columns_to_be_dropped=[]
        for i in train_describe_cat_columns:
            a=test_describe_cat.iloc[1,test_describe_cat_columns.index(i)]/test_describe_cat.iloc[0,test_describe_cat_columns.index(i)]
            a=float(a)
            if a>0.01:
                columns_to_be_dropped.append(i)
        #dropping the columns
        train.drop(columns=columns_to_be_dropped, inplace=True)
        test.drop(columns=columns_to_be_dropped, inplace=True)
        
        #assingin
        train_describe_cat=pd.DataFrame(train.describe(include=['O']))
        test_describe_cat=pd.DataFrame(test.describe(include=['O']))
        train_describe_cat_columns=list(train_describe_cat.columns)
        test_describe_cat_columns=list(test_describe_cat.columns)
        
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
    st.write('Target Variable :', label) #showing the target variable

    #assiging dependent and independent variables
    X=train.drop(label,axis=1)
    y=train[label]

    #train_test_split()
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=4)

    #modelling
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    pred_y=rf.predict(X_val)
    from sklearn.metrics import accuracy_score
    st.write('Accuracy :',accuracy_score(y_val.values,pred_y))

    #making predictions on test data
    y_pred=rf.predict(test)

    #saving the submissions in the form of pandas dataframe
    sample_columns=list(sample.columns)
    sample[sample_columns[1]]=y_pred
    #file_name=input('Enter the title of submission :')
    sample.to_csv('Submission.csv', index=False)
    st.write('Predictions Made Sucessfully !')
    st.download_button('Download Submission', data='Submission.csv',
                       file_name='Submission.csv')
else:
    st.write("Predictions Aren't Made Yet")
st.write("___")
st.write('Find Me Here :')
with st.container():
    left, middle, right = st.columns(3)
    with left:
        st.write('[Kaggle](kaggle.com/prasadposture121)')
    with middle:
        st.write('[GitHub](https://github.com/prasadposture)')
    with right:
        st.write('[LinkedIn](https://www.linkedin.com/in/prasad-posture-6a3a77215/)')