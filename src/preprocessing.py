import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OneHotEncoder

# import settings
import settings


def remove_duplicates(df):
    # Remove duplicates

    duplicates_id = df['data_ref'][df['data_ref'].duplicated()]
    df = df[~df['data_ref'].isin(duplicates_id)]

    return df


def combine_dataframes(df1, df2):
    # Combine dataframes

    df = df1.merge(right=df2.drop(columns='date'), how='inner', on='data_ref')
    df = df.drop(columns='data_ref')

    return df


''' Numerical columns '''


def change_datatype(df):
    # Change datatypes to datetime and numeric respectively

    # Convert date to the datetime format
    df['date'] = pd.to_datetime(df.date, dayfirst=True)

    # Change datatypes and coerce '-' and '--' values to nulls
    for x in settings.num_columns:
        df[x] = pd.to_numeric(df[x], errors='coerce')

    return df


def change_negative_values(df):
    # Change negative values to positive values

    negative_features = []

    for neg_feature in settings.num_columns:
        check_less_than_zero_df = df.loc[df[neg_feature] < 0]
        if len(check_less_than_zero_df) > 0:
            negative_features.append(neg_feature)

    # Convert the negative values to positive by taking the absolute
    for neg_feature in negative_features:
        df[neg_feature] = df[neg_feature].abs()

    return df


def impute_data(df):
    # Impute missing values

    # From EDA: we remove rows with 5 or more columns with missing fields before imputation
    df = df[df.isnull().sum(axis=1) < 5]

    # fit regression model using Bayesian Ridge
    imputer = IterativeImputer(estimator=BayesianRidge())

    # impute missing values
    imputed_data = imputer.fit_transform(df[settings.num_columns])

    # substitute imputed values for missing values
    df[settings.num_columns] = imputed_data

    # change negative values to zero
    df[settings.num_columns] = df[settings.num_columns].clip(lower=0)

    return df


def fix_numeric(df):
    # Fix numerical features (missing data, negative features)

    df = change_datatype(df)
    df = change_negative_values(df)
    df = impute_data(df)

    return df


''' Categorical columns '''


def fix_categorical(df):
    # Fix categorical features

    # Convert all values to uppercase
    for x in settings.cat_columns:
        df[x] = df[x].str.upper()

    # From EDA, we found that "Dew Point Category" and "Wind Direction" have inconsistent labelling
    df['Dew Point Category'] = df['Dew Point Category'].replace(
        settings.Dew_Point_Category_replacements)

    df['Wind Direction'] = df['Wind Direction'].replace(
        settings.Wind_Direction_replacements)
    df['Wind Direction'] = df['Wind Direction'].str.replace(
        '.', '', regex=False)
    return df


def encoding(df):
    # Encoding
    # Ordinal encoding for Dew Point Category and Daily Solar Panel Efficiency
    df['Dew Point Category'] = df['Dew Point Category'].replace(
        settings.Dew_Point_Category_encoding)

    df['Daily Solar Panel Efficiency'] = df['Daily Solar Panel Efficiency'].replace(
        settings.Daily_Solar_Panel_Efficiency_encoding)

    # Encode Wind Direction
    def direction_to_angle(direction):
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        angles = np.linspace(0, 2*np.pi, len(directions), endpoint=False)
        return angles[directions.index(direction)]
    df['Wind Direction'] = df['Wind Direction'].apply(direction_to_angle)

    return df


def feature_selection(df):
    # Feature selection

    # Take the average of PM25 and PSI values
    df['pm25_and_psi'] = df[settings.pm25_and_psi_cols].mean(axis=1)
    df = df.drop(columns=settings.pm25_and_psi_cols)

    # Drop rainfall measurements and cloud cover
    df = df.drop(columns=settings.rainfall_cols + ['Cloud Cover (%)'])

    # Convert date to day of the year to include effects of seasonality
    df.insert(0, 'day_of_the_year', df['date'].dt.dayofyear.astype(int))
    df.drop(columns='date', inplace=True)

    return df


def preprocessing(weather_df, air_quality_df):

    # Remove duplicates
    print(settings.G + "Removing duplicates" + settings.W)
    weather_df = remove_duplicates(weather_df)
    air_quality_df = remove_duplicates(air_quality_df)

    # Combine the datasets
    print(settings.G + "Combining the 2 datasets" + settings.W)
    df = combine_dataframes(weather_df, air_quality_df)

    # Fixing errorneous values
    print(settings.G + "Fixing errorneous values" + settings.W)
    df = fix_numeric(df)
    df = fix_categorical(df)

    # Encoding
    print(settings.G + "Encoding" + settings.W)
    df = encoding(df)

    # Feature Selection
    print(settings.G + "Feature selection" + settings.W)
    df = feature_selection(df)

    return df
