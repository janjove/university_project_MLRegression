from funct_feature_engenirin import *
def model_1_feature(X):
    columns = ['LotArea', 'GrLivArea', 'BedroomAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'GarageCars']
    X_copy = X[columns]
    X_copy['Overall_Cal'] = X['OverallQual'] + X['OverallCond']
    X_copy['Last_mod'] = X[['YearBuilt', 'YearRemodAdd']].max(axis=1)
    X_copy['Total_bath'] = X['FullBath'] + 0.5*X['HalfBath']
    X_copy['post_crisis'] = X.apply(pre_or_post_2008, axis=1)
    return X_copy