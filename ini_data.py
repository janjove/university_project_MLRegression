import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def descrife_df(df):
    print(df.describe())
    print(df.info())
def nans(df):
    print ("Tenim Nans?",has_nans(df)) #té nans?
    percent_nan_fila = df.isna().mean(axis = 1)*100
    print(percent_nan_fila.sort_values())
def has_nans(df): #mirar si té nans
  return df.isna().sum().sum()>0

def get_percentage_nan_per_column(df):
   df.isna().sum().sort_values()/len(df)*100.
def nans_columnes(df):
    percentage_nan = get_percentage_nan_per_column(df)
    print("hola")
    print(percentage_nan)
    print("adeu")


def initial_data(df):
   print("Descripció inicial")
   descrife_df(df)
   print("Nans files")
   nans(df)
   print("Nans columnes") ## error
   nans_columnes(df)

def graphics_categoric(df,colum):
    column = df[colum]
    category_counts = column.value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title(f'Distribution of {colum}')
    plt.xlabel('Category')
    plt.ylabel('Frequency')
    plt.show()
    
def graphics_numeric(df,colum):
    column = df[colum]

    plt.figure(figsize=(10, 6))
    sns.histplot(column, bins=30, kde=True, color="red")
    plt.title(f'Distribution of {colum}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show() 

def boxplot_category(df,colum):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=colum, y='SalePrice', data=df)
    plt.title('Comparison of Target Variable Distribution Across Categories')
    plt.xlabel(f'{colum}')
    plt.ylabel('Target Variable Distribution')
    plt.show()