from funct_feature_engenirin import *

def model_1(X,y):     
    columnas_seleccionadas =['LotArea','GrLivArea','BedroomAbvGr','KitchenQual','TotRmsAbvGrd','GarageCars']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()
    ## Crem una variable que anomenrarem varaible global
    X_train['Overall_Cal'] = X['OverallQual'] + X['OverallCond']
    ## Data de la ultima modificacio
    X_train['Last_mod'] = X[['YearBuilt', 'YearRemodAdd']].max(axis=1)
    X_train['Total_bath'] = X['FullBath'] + 0.5*X['HalfBath']
    ## Funcio per crear si és posterior a 2006
    # Apply the function to each row
    X_train['post_crisis'] = X.apply(pre_or_post_2008_bool, axis=1)
    return X_train,y_train
def model_2(X,y):     
    columnas_seleccionadas =['LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr','KitchenQual','TotRmsAbvGrd','GarageCars']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()
    ## Crem una variable que anomenrarem varaible global
    X_train['Ext_Cal'] = X['ExterQual'].map(calificacio_a_numero) + X['ExterCond'].map(calificacio_a_numero)

    X_train['Last_mod'] = X[['YearBuilt', 'YearRemodAdd']].max(axis=1)
    X_train['Total_bath'] = X['FullBath'] + 0.5*X['HalfBath']+X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']
    X_train['BSMT'] = (X['TotalBsmtSF'] > 0).astype(int)
    X_train['second_floor'] = (X['2ndFlrSF'] > 0).astype(int)
    X_train['Remodelado'] = (X['YearRemodAdd'] > X['YearBuilt']).astype(int)
    X_train['pre_or_post_2008'] = X.apply(pre_or_post_2008_bool, axis=1)
    X_train['main_road'] = X.apply(main_road, axis=1)
    return X_train,y_train
def model_3(X,y):
    columnas_seleccionadas =['MSSubClass','LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr',
                         'TotRmsAbvGrd','GarageCars','CentralAir','1stFlrSF','2ndFlrSF','TotalBsmtSF']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()

    X_train['Ext_Cal'] = X['ExterQual'].map(calificacio_a_numero) + X['ExterCond'].map(calificacio_a_numero)
    ## Data de la ultima modificacio
    X_train['Last_mod'] = X[['YearBuilt', 'YearRemodAdd']].max(axis=1)
    X_train['Total_bath'] = X['FullBath'] + 0.5*X['HalfBath']+X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']

    ## Funcio per crear si és posterior a 2008
    X_train['BSMT'] = (X['TotalBsmtSF'] > 0).astype(int)
    X_train['second_floor'] = (X['2ndFlrSF'] > 0).astype(int)
    X_train['Remodelado'] = (X['YearRemodAdd'] > X['YearBuilt']).astype(int)

    X_train['GarageCal'] = X['GarageQual'].map(calificacio_a_numero)+X['GarageCond'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)

    # Apply the function to each row
    X_train['pre_or_post_2008'] = X.apply(pre_or_post_2008, axis=1)
    X_train['main_road'] = X.apply(main_road, axis=1)
    
    X_train['ad_station'] = X.apply(ad_station, axis=1)

    X_train['CentralAir'] = X_train['CentralAir'].map(air_condition_to_numeric)
    X_train['MSSubClass'] = X_train['MSSubClass'].astype('category')
    return X_train,y_train

def model_4(X,y):
    columnas_seleccionadas =['MSSubClass','LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr',
                         'TotRmsAbvGrd','GarageCars','CentralAir','TotalBsmtSF','YearBuilt','WoodDeckSF','Fireplaces']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()

   
    X_train['Ext_Cal'] = X['ExterQual'].map(calificacio_a_numero) + X['ExterCond'].map(calificacio_a_numero)
    ## Data de la ultima modificacio
    X_train['Last_mod'] = X[['YearBuilt', 'YearRemodAdd']].max(axis=1)
    X_train['Total_bath'] = X['FullBath'] + 0.5*X['HalfBath']+X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']

    ## Funcio per crear si és posterior a 2008

    X_train['BSMT'] = (X['TotalBsmtSF'] > 0).astype(int)
    X_train['second_floor'] = (X['2ndFlrSF'] > 0).astype(int)
    X_train['Remodelado'] = (X['YearRemodAdd'] > X['YearBuilt']).astype(int)

    
    X_train['GarageCal'] = X['GarageQual'].map(calificacio_a_numero)+X['GarageCond'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)

    # Apply the function to each row
    X_train['pre_or_post_2008'] = X.apply(pre_or_post_2008, axis=1)
    X_train['main_road'] = X.apply(main_road, axis=1)
  
    X_train['ad_station'] = X.apply(ad_station, axis=1)

    

    X_train['CentralAir'] = X_train['CentralAir'].map(air_condition_to_numeric)
    X_train['MSSubClass'] = X_train['MSSubClass'].astype('category')
    X_train['metres_porxo']= X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']

    X_train['damages'] = X['Functional'].map(functional_to_number)

    return X_train,y_train

def model_5(X,y):
    columnas_seleccionadas =['MSSubClass','LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr',
                         'TotRmsAbvGrd','GarageCars','CentralAir','TotalBsmtSF','YearBuilt','WoodDeckSF','Fireplaces','GarageArea']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()

   
    X_train['ExterQual'] = X['ExterQual'].map(calificacio_a_numero)
    X_train['ExterCond']= X['ExterCond'].map(calificacio_a_numero)
    ## Data de la ultima modificacio
    X_train['Last_mod'] = X[['YearBuilt', 'YearRemodAdd']].max(axis=1)
    X_train['Total_bath'] = X['FullBath'] + 0.5*X['HalfBath']+X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']

    ## Funcio per crear si és posterior a 2008

    X_train['Remodelado'] = (X['YearRemodAdd'] > X['YearBuilt']).astype(int)

    
    X_train['GarageCal'] = X['GarageQual'].map(calificacio_a_numero)+X['GarageCond'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)

    # Apply the function to each row
    X_train['pre_or_post_2008'] = X.apply(pre_or_post_2008, axis=1)
    
    X_train['CentralAir'] = X_train['CentralAir'].map(air_condition_to_numeric)
    X_train['MSSubClass'] = X_train['MSSubClass'].astype('category')
    X_train['metres_porxo']= X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']

    X_train['damages'] = X['Functional'].map(functional_to_number)

    return X_train,y_train

def model_6(X,y):
    columnas_seleccionadas =['MSSubClass','LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr',
                         'TotRmsAbvGrd','GarageCars','CentralAir','1stFlrSF','2ndFlrSF','TotalBsmtSF','YearBuilt', 'YearRemodAdd','MiscVal']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()

    X_train['Ext_Cal'] = X['ExterQual'].map(calificacio_a_numero) + X['ExterCond'].map(calificacio_a_numero)
    ## Data de la ultima modificacio
    X_train['Last_mod'] = X[['YearBuilt', 'YearRemodAdd']].max(axis=1)
    X_train['Total_bath'] = X['FullBath'] + 0.5*X['HalfBath']+X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']

    ## Funcio per crear si és posterior a 2008
    X_train['BSMT'] = (X['TotalBsmtSF'] > 0).astype(int)
    X_train['Remodelado'] = (X['YearRemodAdd'] > X['YearBuilt']).astype(int)

    X_train['GarageCal'] = X['GarageQual'].map(calificacio_a_numero)+X['GarageCond'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)


    X_train['CentralAir'] = X_train['CentralAir'].map(air_condition_to_numeric)
    X_train['MSSubClass'] = X_train['MSSubClass'].astype('category')
    X_train['damages'] = X['Functional'].map(functional_to_number)
    return X_train,y_train
def model_7(X,y):
    columnas_seleccionadas =['MSSubClass','LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr',
                         'TotRmsAbvGrd','GarageCars','CentralAir','1stFlrSF','2ndFlrSF','TotalBsmtSF','YearBuilt', 'YearRemodAdd','MiscVal',
                         ]
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()

    X_train['Ext_Cal'] = X['ExterQual'].map(calificacio_a_numero) + X['ExterCond'].map(calificacio_a_numero)
    ## Data de la ultima modificacio
    X_train['Total_bath'] = X['FullBath'] + 0.5*X['HalfBath']+X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']
    X_train['BsmtQual'] = X['BsmtQual'].map(calificacio_a_numero)

    ## Funcio per crear si és posterior a 2008
    X_train['BSMT'] = (X['TotalBsmtSF'] > 0).astype(int)
    X_train['Remodelado'] = (X['YearRemodAdd'] > X['YearBuilt']).astype(int)

    X_train['GarageCal'] = X['GarageQual'].map(calificacio_a_numero)+X['GarageCond'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)


    X_train['CentralAir'] = X_train['CentralAir'].map(air_condition_to_numeric)
    X_train['MSSubClass'] = X_train['MSSubClass'].astype('category')
    X_train['damages'] = X['Functional'].map(functional_to_number)
    return X_train,y_train

def model_8(X,y):
    X_train,y_train = model_3(X,y)
    return X_train,y_train

def model_9(X,y):
    columnas_seleccionadas =['MSSubClass','LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr',
                         'TotRmsAbvGrd','GarageCars','CentralAir','1stFlrSF','2ndFlrSF','TotalBsmtSF','MSZoning','HouseStyle',
                         'Foundation','FullBath']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()
    
    X_train['Ext_Cal'] = X['ExterQual'].map(calificacio_a_numero) + X['ExterCond'].map(calificacio_a_numero)
    ## Data de la ultima modificacio
    X_train['Last_mod'] = X[['YearBuilt', 'YearRemodAdd']].max(axis=1)
    X_train['bath_oGr'] = X['FullBath'] + 0.5*X['HalfBath']
    X_train['bath_BSMT']=X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']

    ## Funcio per crear si és posterior a 2008

    X_train['pre_or_post_2008'] = X.apply(pre_or_post_2008, axis=1)

    X_train['GarageCal'] = X['GarageQual'].map(calificacio_a_numero)+X['GarageCond'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)

    # Apply the function to each row

    X_train['CentralAir'] = X_train['CentralAir'].map(air_condition_to_numeric)
    X_train['MSSubClass'] = X_train['MSSubClass'].astype('category')

    X_train['good_position'] = X.apply(good_position, axis=1)

    return X_train,y_train

def model_10(X,y):
    columnas_seleccionadas =['MSSubClass','LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr',
                         'TotRmsAbvGrd','GarageCars','CentralAir','1stFlrSF','2ndFlrSF','TotalBsmtSF','MSZoning']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()
    
    X_train['Ext_Cal'] = X['ExterQual'].map(calificacio_a_numero) + X['ExterCond'].map(calificacio_a_numero)
    ## Data de la ultima modificacio
    X_train['Last_mod'] = X[['YearBuilt', 'YearRemodAdd']].max(axis=1)
    X_train['bath_oGr'] = X['FullBath'] + 0.5*X['HalfBath']
    X_train['bath_BSMT']=X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']

    ## Funcio per crear si és posterior a 2008

    X_train['GarageQual'] = X['GarageQual'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)

    # Apply the function to each ro

    X_train['pre_or_post_2008'] = X.apply(pre_or_post_2008, axis=1)

    X_train['CentralAir'] = X_train['CentralAir'].map(air_condition_to_numeric)
    X_train['MSSubClass'] = X_train['MSSubClass'].astype('category')
    return X_train,y_train
def model_11(X,y):
    columnas_seleccionadas =['MSSubClass','LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr',
                         'TotRmsAbvGrd','GarageCars','CentralAir','1stFlrSF','2ndFlrSF','TotalBsmtSF','YearBuilt', 'YearRemodAdd',
                         'GarageArea']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()

    X_train['Ext_Cal'] = X['ExterQual'].map(calificacio_a_numero) + X['ExterCond'].map(calificacio_a_numero)
    ## Data de la ultima modificacio
    #X_train['Last_mod'] = X[['YearBuilt', 'YearRemodAdd']].max(axis=1)
    X_train['Total_bath'] = X['FullBath'] + 0.5*X['HalfBath']

    ## Funcio per crear si és posterior a 2008

    X_train['Total_area'] = X['1stFlrSF']+X['2ndFlrSF']+X['TotalBsmtSF']


    X_train['GarageQual'] = X['GarageQual'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)

    X_train['CentralAir'] = X_train['CentralAir'].map(air_condition_to_numeric)
    X_train['MSSubClass'] = X_train['MSSubClass'].astype('category')
    X_train['damages'] = X['Functional'].map(functional_to_number)

    return X_train,y_train
def model_12(X,y):
    columnas_seleccionadas =['MSSubClass','LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr',
                         'TotRmsAbvGrd','GarageCars','CentralAir','1stFlrSF','2ndFlrSF','TotalBsmtSF','MSZoning','HouseStyle',
                         'FullBath','YearBuilt','YearRemodAdd','GarageArea','MasVnrArea']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()

    X_train['Ext_Cal'] = X['ExterQual'].map(calificacio_a_numero) + X['ExterCond'].map(calificacio_a_numero)
    ## Data de la ultima modificacio
    X_train['bath_oGr'] = X['FullBath'] + 0.5*X['HalfBath']
    X_train['bath_BSMT']=X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']
    X_train['Total_area'] = X['1stFlrSF']+X['2ndFlrSF']+X['TotalBsmtSF']


    #X_train['pre_or_post_2008'] = X.apply(pre_or_post_2008, axis=1)

    X_train['GarageCal'] = X['GarageQual'].map(calificacio_a_numero)+X['GarageCond'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)

    X_train['CentralAir'] = X_train['CentralAir'].map(air_condition_to_numeric)
    X_train['MSSubClass'] = X_train['MSSubClass'].astype('category')

    X_train['good_position'] = X.apply(good_position, axis=1)
    return X_train,y_train

def model_13(X,y):
    columnas_seleccionadas =['MSSubClass','LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr',
                         'TotRmsAbvGrd','GarageCars','CentralAir','1stFlrSF','2ndFlrSF','TotalBsmtSF','MSZoning','HouseStyle',
                         'FullBath','YearBuilt','YearRemodAdd','GarageArea','MasVnrArea','BldgType','Exterior1st','MiscVal']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()
    
    X_train['Ext_Cal'] = X['ExterQual'].map(calificacio_a_numero) + X['ExterCond'].map(calificacio_a_numero)
    ## Data de la ultima modificacio
    X_train['bath_oGr'] = X['FullBath'] + 0.5*X['HalfBath']
    X_train['bath_BSMT']=X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']
    X_train['Total_area'] = X['1stFlrSF']+X['2ndFlrSF']+X['TotalBsmtSF']

    X_train['HeatingQC'] = X['HeatingQC'].map(calificacio_a_numero)

    X_train['pre_or_post_2008'] = X.apply(pre_or_post_2008, axis=1)

    X_train['GarageCal'] = X['GarageQual'].map(calificacio_a_numero)+X['GarageCond'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)

    # Apply the function to each row

    X_train['CentralAir'] = X_train['CentralAir'].map(air_condition_to_numeric)
    X_train['MSSubClass'] = X_train['MSSubClass'].astype('category')

    X_train['good_position'] = X.apply(good_position, axis=1)
    return X_train,y_train

def model_14(X,y):
    columnas_seleccionadas =['MSZoning','LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr',
                            'TotRmsAbvGrd','GarageCars','CentralAir','1stFlrSF','2ndFlrSF','TotalBsmtSF','HouseStyle',
                            'FullBath','YearBuilt','YearRemodAdd','GarageArea','MasVnrArea','Exterior1st','SaleType','SaleCondition','BsmtExposure']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()
    X_train['Ext_Cal'] = X['ExterQual'].map(calificacio_a_numero) + X['ExterCond'].map(calificacio_a_numero)
    ## Data de la ultima modificacio
    X_train['bath_oGr'] = X['FullBath'] + 0.5*X['HalfBath']

    X_train['HeatingQC'] = X['HeatingQC'].map(calificacio_a_numero)

    ## Funcio per crear si és posterior a 2008

    X_train['pre_or_post_2008'] = X.apply(pre_or_post_2008, axis=1)

    X_train['GarageQual'] = X['GarageQual'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)

    # Apply the function to each row

    X_train['CentralAir'] = X_train['CentralAir'].map(air_condition_to_numeric)

    X_train['good_position'] = X.apply(good_position, axis=1)

    return X_train,y_train

def model_15(X,y):
    columnas_seleccionadas =['MSSubClass','LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr',
                         'TotRmsAbvGrd','GarageCars','CentralAir','1stFlrSF','2ndFlrSF','TotalBsmtSF','MSZoning','HouseStyle',
                         'FullBath','YearBuilt','YearRemodAdd','GarageArea','MasVnrArea']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()

    X_train['Ext_Cal'] = X['ExterQual'].map(calificacio_a_numero) + X['ExterCond'].map(calificacio_a_numero)
    ## Data de la ultima modificacio
    X_train['bath_oGr'] = X['FullBath'] + 0.5*X['HalfBath']
    X_train['bath_BSMT']=X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']
    X_train['Total_area'] = X['1stFlrSF']+X['2ndFlrSF']+X['TotalBsmtSF']
    #X_train['pre_or_post_2008'] = X.apply(pre_or_post_2008, axis=1)

    X_train['GarageCal'] = X['GarageQual'].map(calificacio_a_numero)+X['GarageCond'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)

    # Apply the function to each row
    X_train['CentralAir'] = X_train['CentralAir'].map(air_condition_to_numeric)
    X_train['MSSubClass'] = X_train['MSSubClass'].astype('category')
    X_train['good_position'] = X.apply(good_position, axis=1)

    return X_train,y_train

def model_15_split(X,y):
    separacio = 4400
    X_train,y_train = model_15(X,y)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)


    X_train_car = X_train[X_train['Total_area'] > separacio]
    y_train_car = y_train[X_train['Total_area'] > separacio]

    X_train_bar = X_train[X_train['Total_area'] <= separacio]
    y_train_bar = y_train[X_train['Total_area'] <= separacio]
    return X_train_car,y_train_car, X_train_bar, y_train_bar

def model_17(X,y):
    columnas_seleccionadas =['LotArea','Neighborhood','OverallQual','BedroomAbvGr',
                         'GarageCars','1stFlrSF','2ndFlrSF','TotalBsmtSF',
                         'YearBuilt','PoolArea','GarageArea']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()
    ## Data de la ultima modificacio
    X_train['bath_oGr'] = X['FullBath'] + 0.5*X['HalfBath']
    X_train['Total_area'] = X['1stFlrSF']+X['2ndFlrSF']+X['TotalBsmtSF']

    #X_train['pre_or_post_2008'] = X.apply(pre_or_post_2008, axis=1)

    X_train['GarageQual'] = X['GarageQual'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)
    return X_train,y_train

def model_17_split(X,y):
    separacio = 4400
    X_train,y_train = model_17(X,y)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    # Splitting X_train and y_train based on the condition of Total_area
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    X_train_car = X_train[X_train['Total_area'] > separacio]
    y_train_car = y_train[X_train['Total_area'] > separacio]

    X_train_bar = X_train[X_train['Total_area'] <= separacio]
    y_train_bar = y_train[X_train['Total_area'] <= separacio]
    return X_train_car,y_train_car, X_train_bar, y_train_bar

def model_18(X,y):
    columnas_seleccionadas =['LotArea','Neighborhood','OverallQual','BedroomAbvGr',
                         'GarageCars','1stFlrSF','2ndFlrSF','TotalBsmtSF',
                         'YearBuilt','GarageArea','MasVnrArea']

    X_train = X[columnas_seleccionadas]
    y_train=y.copy()
    ## Data de la ultima modificacio
    X_train['bath_oGr'] = X['FullBath'] + 0.5*X['HalfBath']
    X_train['Total_area'] = X['1stFlrSF']+X['2ndFlrSF']+X['TotalBsmtSF']

    #X_train['pre_or_post_2008'] = X.apply(pre_or_post_2008, axis=1)

    X_train['GarageQual'] = X['GarageQual'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)
    X_train['HeatingQC'] = X['HeatingQC'].map(calificacio_a_numero)

    X_train['good_position'] = X.apply(good_position, axis=1)
    X_train['metres_porxo']= X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']
    return X_train,y_train

def model_18_split(X,y):
    separacio = 4400
    X_train,y_train = model_17(X,y)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    # Splitting X_train and y_train based on the condition of Total_area
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    X_train_car = X_train[X_train['Total_area'] > separacio]
    y_train_car = y_train[X_train['Total_area'] > separacio]

    X_train_bar = X_train[X_train['Total_area'] <= separacio]
    y_train_bar = y_train[X_train['Total_area'] <= separacio]
    return X_train_car,y_train_car, X_train_bar, y_train_bar
def model_19(X,y):
    columnas_seleccionadas =['MSSubClass','LotArea','Neighborhood','OverallQual','OverallCond','GrLivArea','BedroomAbvGr',
                          'TotRmsAbvGrd','GarageCars','CentralAir','1stFlrSF','2ndFlrSF','TotalBsmtSF',
                          'FullBath','YearBuilt','YearRemodAdd','GarageArea','MasVnrArea']
    X_train = X[columnas_seleccionadas]
    y_train=y.copy()
    X_train['Ext_Cal'] = X['ExterQual'].map(calificacio_a_numero) + X['ExterCond'].map(calificacio_a_numero)
    ## Data de la ultima modificacio
    X_train['bath_oGr'] = X['FullBath'] + 0.5*X['HalfBath']
    X_train['bath_BSMT']=X['BsmtFullBath'] + 0.5*X['BsmtHalfBath']
    X_train['Total_area'] = X['1stFlrSF']+X['2ndFlrSF']+X['TotalBsmtSF']



    X_train['GarageCal'] = X['GarageQual'].map(calificacio_a_numero)+X['GarageCond'].map(calificacio_a_numero)
    X_train['KitchenQual'] = X['KitchenQual'].map(calificacio_a_numero)

    # Apply the function to each row
    X_train['CentralAir'] = X_train['CentralAir'].map(air_condition_to_numeric)

    X_train['good_position'] = X.apply(good_position, axis=1)

    X_train['normal_sale'] = X.apply(normal_sale, axis=1)
    return X_train,y_train
def model_19_split(X,y):
    separacio = 4400
    X_train,y_train = model_19(X,y)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    # Splitting X_train and y_train based on the condition of Total_area
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    X_train_car = X_train[X_train['Total_area'] > separacio]
    y_train_car = y_train[X_train['Total_area'] > separacio]

    X_train_bar = X_train[X_train['Total_area'] <= separacio]
    y_train_bar = y_train[X_train['Total_area'] <= separacio]
    return X_train_car,y_train_car, X_train_bar, y_train_bar