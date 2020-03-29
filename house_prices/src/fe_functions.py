# missing data imputation
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import MissingIndicator
# scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
# variable transformation
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
# categorical  
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# ======================================================================================================
# Missing Data Imputation
# ======================================================================================================

"""models which work with missing data - most tree models and KNN"""

# Complete Case Analysis
""" 
Complete case analysis implies analysing only those observations in the dataset that contain values
in all the variables. In other words, in complete case analysis we remove all observations with missing values. 
This procedure is suitable when there are few observations with missing data in the dataset. 
But, if the dataset contains missing data across multiple variables, or some variables contain 
a high proportion of missing observations, we can easily remove a big chunk of the dataset, and this is undesired. 
Criteria:
- less than 5% missing, MAR
"""
def dropna(df, cols):
    df = df.dropna(subset=cols, inplace=True)
    return df
    
# Mean / Median / Mode Imputation
""" 
We can replace missing values with the mean, the median or the mode of the variable. 
Mean / median / mode imputation is widely adopted in organisations and data competitions. 
Although in practice this technique is used in almost every situation, the procedure is 
suitable if data is missing at random and in small proportions. If there are a lot of missing 
observations, however, we will distort the distribution of the variable, as well as its 
relationship with other variables in the dataset. Distortion in the variable distribution may 
affect the performance of linear models. For categorical variables, replacement by the mode,
is also known as replacement by the most frequent category.
Criteria: 
- less than 15% missing, MAR
- if skewed; median, if over 75% a specific value; Mode, else: Mean 
Model criteria:
-
"""
def impute_mean(train_df, test_df, cols):
    for col in cols:
        mean = train_df.loc[train_df[col].notnull(),col].mean(skipna=True)
        train_df[col] = train_df[col].fillna(value=mean)
        test_df[col] = test_df[col].fillna(value=mean)
    return train_df, test_df

def impute_median(train_df, test_df, cols):
    for col in cols:
        mean = train_df.loc[train_df[col].notnull(),col].median(skipna=True)
        train_df[col] = train_df[col].fillna(value=median)
        test_df[col] = test_df[col].fillna(value=median)
    return train_df, test_df

def impute_mode(df, cols):
    si = SimpleImputer(strategy='most_frequent')
    df[cols] = si.fit_transform(df[cols])
    return df

# Replacement by Arbitrary Value
""" Replacement by an arbitrary value, as its names indicates, refers to replacing missing data 
by any, arbitrarily determined value, but the same value for all missing data. 
Replacement by an arbitrary value is suitable if data is not missing at random, 
or if there is a huge proportion of missing values. If all values are positive, 
a typical replacement is -1. Alternatively, replacing by 999 or -999 are common practice.
We need to anticipate that these arbitrary values are not a common occurrence in the variable. 
Replacement by arbitrary values however may not be suited for linear models, 
as it most likely will distort the distribution of the variables, and therefore 
model assumptions may not be met.
For categorical variables, this is the equivalent of replacing missing observations with the
label “Missing” which is a widely adopted procedure.
Criteria:
- more than 15% missing OR MNAR
Model criteria:
- not suited to linear models
- suited to tree models which cannot handle missing data
"""


# Missing Value Indicator
"""
The missing indicator technique involves adding a binary variable to indicate whether the value is 
missing for a certain observation. This variable takes the value 1 if the observation is missing,
or 0 otherwise. One thing to notice is that we still need to replace the missing values in the 
original variable, which we tend to do with mean or median imputation. By using these 2 techniques
together, if the missing value has predictive power, it will be captured by the missing indicator, 
and if it doesn’t it will be masked by the mean / median imputation. These 2 techniques in combination
tend to work well with linear models. But, adding a missing indicator expands the feature space and, 
as multiple variables tend to have missing values for the same observations, many of these newly 
created binary variables could be identical or highly correlated.
Criteria:
- MNAR
Model criteria:
- can lead to multicollinearity 
"""

## Random Sample Imputation
## End of Distribution Imputation
## Multivariate 

# Imputation using models
# TODO

# ======================================================================================================
# Categorical Encoding
# ======================================================================================================


# ======================================================================================================
# Variable Transformation
# ======================================================================================================

# Logarithm
""" Takes log of a variable
Criteria - skewed distributions
Model Crtieria - 
""""
def log(df, cols):
    for col in cols:
        df[col] = np.log1p(df[col])
    return df

# Box Cox
def box_cox(df, cols):
""" Applies an adaptation of an exponential transformation
Criteria - skewed distributions, positive values only
Model Crtieria - 
""""
    for col in cols:
        df[col] = boxcox1p(df[col], boxcox_normmax(df[col] + 1))
    return df

# Yeo Johnson
def yeo_johnson(df, cols):
""" Applies an adaptation of an exponential transformation
Criteria - skewed distributions, positive values only
Model Crtieria - 
""""
    for col in cols:
        df[col] = boxcox1p(df[col], boxcox_normmax(df[col] + 1))
    return df


# ======================================================================================================
# Discretisation
# ======================================================================================================


# ======================================================================================================
# Outlier Engineering
# ======================================================================================================

# Outlier removal


# Capping



# Treating outliers as missing values 



# ======================================================================================================
# Scaling
# ======================================================================================================
"""models which work with missing data - most tree models and KNN """

# Standardisation
def standard_scale(df, cols):
""" Subtracts mean from each value and divides by the standard deviation. This
makes variables have 0 mean and unit-variance
Criteria - normal distribution
Model-criteria - NA
"""
    standard_scaler = StandardScaler()
    df[cols] = standard_scaler.fit_transform(df[cols])
    return df

# Min-Max Scaling
def min_max_scale(df, cols):
""" Subtracts minimum from each value and divides by the value range
Criteria - partially skewed distribution
Model-criteria - NA
"""
    standard_scaler = MinMaxScaler()
    df[cols] = standard_scaler.fit_transform(df[cols])
    return df

# Robust Scaling
def robust_scale(df, cols):
""" Remove median from each value and divide by the interquartile range
Criteria - highly skewed distribution
Model-criteria - NA   
"""
    standard_scaler = StandardScaler()
    df[cols] = standard_scaler.fit_transform(df[cols])
    return df

# ======================================================================================================
# Date and Time Engineering
# ======================================================================================================


