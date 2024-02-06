import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

def pipeline(X_train,y_train):
# Define which columns are numeric and which are categorical
    numeric_features = X_train.select_dtypes(include=['number']).columns
    categorical_features = X_train.select_dtypes(exclude=['number']).columns

    # Create a transformer for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    # Create a transformer for categorical features
    categorical_transformer = Pipeline(steps=[
        ('Target Encoder', TargetEncoder()),
        ('scaler', StandardScaler())
    ])

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the complete pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit and transform the data
    pipeline.fit(X_train,y_train)

    transformed_data=pipeline.transform(X_train)

    new_column_names = np.concatenate((numeric_features,categorical_features)) # Adjust as needed
    X_train_transformed = pd.DataFrame(transformed_data, columns=new_column_names)
    return X_train_transformed,pipeline