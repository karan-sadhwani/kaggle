import matplotlib.gridspec as gridspec
# import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# import matplotlib.style as style

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


def temp_functions(train_df):

    

    col_translate = {"Semana" : "Week_number",
                    "Agencia_ID" : "Sales_Depot_ID",
                    "Canal_ID" : "Sales_Channel_ID",
                    "Ruta_SAK" : "Route",
                    "Cliente_ID" : "Client_ID",
                    "NombreCliente" : "Client_name",
                    "Producto_ID" : "Product_ID",
                    "NombreProducto" : "Product_Name",
                    "Venta_uni_hoy" : "Sales_unit_this_week",
                    "Venta_hoy" : "Sales_this_week",
                    "Dev_uni_proxima" : "Returns_unit_next_week",
                    "Dev_proxima" : "Returns_next_week",
                    "Demanda_uni_equil" : "Adjusted_Demand"}
    train_df = 






def split_cat_num(df):
    num_df = df.select_dtypes(include=numerics)
    cat_df = df.select_dtypes(exclude=numerics)
    return num_df, cat_df

def missing_vals(df):
    """This function takes a df as input and returns two columns, total missing values and total missing values percentage"""
    missing_abs = df.isnull().sum()
    missing_pct = round(df.isnull().sum()/len(df)*100,2)
    stats_cat = pd.concat([missing_abs, missing_pct], axis=1, keys=['missing_abs','missing_concat'])
    return stats_cat

# ======================================================================================================
# numerical
# ======================================================================================================

def plot(df, feature):
## Creating a customized chart. and giving in figsize and everything. 
    style.use('fivethirtyeight')
    fig = plt.figure(constrained_layout=True, figsize=(15,10))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(df.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 )
    plt.show()

# def multicollinearity:
    
def get_numeric_stats(df):
    stats_num = df.describe().transpose().reset_index()
    stats_num['iqr'] = stats_num['75%'] - stats_num['25%']
    stats_num['outlier_upper_lim'] = stats_num['75%'] + stats_num['iqr']*1.5
    stats_num['outlier_lower_lim'] = stats_num['25%'] - stats_num['iqr']*1.5
    listy = []
    for index, row in stats_num.iterrows():
        listy.append(len(df[(df[row['index']] > row['outlier_upper_lim']) & \
         (df[row['index']] < row['outlier_lower_lim'])]))
    print(listy)
    skewness = df.skew()
    kurtosis = df.kurt()
    #stats_num['negative_vals'] = np.where(stats_num[,"min"]>0,1,0)
    
    
  
    
    stats_num = pd.concat([skewness, kurtosis], axis=1, keys=['skewness','kurtosis','negative_vals'])

    return stats_num
