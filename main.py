import pandas as pd
from ini_data import *
from sklearn.model_selection import train_test_split
from model import Model
from funct_feature_engenirin import *
from feature_sel import *
def split_df(df):
    X, X_test, y, y_test = train_test_split(df.drop('SalePrice',axis=1),
                                                        df['SalePrice'], test_size=0.20,
                                                        random_state=111)
    return X, X_test, y, y_test 

def initial_data_exp():

    initial_data(df)

    for col in df.columns:
        if df[col].nunique() < 20:
            graphics_categoric(df, col)
            boxplot_category(df, col)
        else:
            graphics_numeric(df, col)

### TO DO ###
            # Model per predir els les millors features per separar el model
##
if __name__ == "__main__":
    df = pd.read_csv('Data/house_advance.csv')
    X, X_test, y, y_test = split_df(df)
    X_train,y_train = model_11(X,y)
    model_11 = Model(X_train,y_train)
    model_11.best_model_selection()
    
