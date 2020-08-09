import pandas as pd


def encode_column_with_list(df, col, prefix='_'):
    """
    One-hot encode column with list values
    INPUT: 
        df: Original dataframe
        col: Name of the column
        prefix: String to prefix the new columns with
    
    OUTPUT:
        A new data frame
    """
    col_vals = []
    for lst in df[col]:
        for val in lst:
            if val not in col_vals:
                col_vals.append(val)
    
    # one-hot encode col_val
    for val in col_vals:
        df[prefix + val] = df[col].apply(lambda x: 1 if val in x else 0)
    
    # Drop column
    df.drop([col], axis=1, inplace=True)
    
    return df


def create_dummy_df(df, cat_cols, dummy_na, custom_prefix=None):
    """
    INPUT:
    df - pandas dataframe with categorical variables you want to dummy
    cat_cols - list of strings that are associated with names of the categorical columns
    dummy_na - Bool holding whether you want to dummy NA vals of categorical columns or not
    
    OUTPUT:
    df - a new dataframe that has the following characteristics:
            1. contains all columns that were not specified as categorical
            2. removes all the original columns in cat_cols
            3. dummy columns for each of the categorical columns in cat_cols
            4. if dummy_na is True - it also contains dummy columns for the NaN values
            5. Use a prefix of the column name with an underscore (_) for separating 
    """
    for col in cat_cols:
        prefix = custom_prefix if custom_prefix is not None else col
        try:
            # Convert categorical variable into dummy/indicator variable
            dummy_cols_df = pd.get_dummies(df[col], prefix=prefix, prefix_sep='_', drop_first=False, dummy_na=dummy_na)
            # Remove the converted column
            df_without_col = df.drop(col, axis=1)
            # Concatenate the two columns
            df = pd.concat([df_without_col, dummy_cols_df], axis=1)
        except:
            continue

    return df
