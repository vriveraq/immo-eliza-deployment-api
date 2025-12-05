"""
Imputing missing values, encoding, rescaling
"""
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from joblib import dump
from pgeocode import Nominatim

def create_postal_price_df(clean_data = pd.DataFrame):
    #level encoding (median price per code) for postal code
    postal_mean_price = clean_data.groupby("postal_code")["price"].mean()
    clean_data["postal_encoded"] = clean_data["postal_code"].map(postal_mean_price)
    return clean_data[['postal_code', 'postal_encoded']]

def frequency_encoder_reg_prov(clean_data: pd.DataFrame):
    geo = Nominatim("be")

    # 1) Get unique postal codes as strings
    postal_codes = (
         clean_data["postal_code"]
        .dropna()
        .astype(int)       # ensure clean integers
        .astype(str)       # convert to string for pgeocode
        .unique()
        .tolist()
    )

    # 2) Query pgeocode for these postal codes
    geo_raw = geo.query_postal_code(postal_codes)

    # Ensure we have a DataFrame (pgeocode returns Series if a single code)
    if isinstance(geo_raw, pd.Series):
        geo_df = geo_raw.to_frame().T
    else:
        geo_df = geo_raw

    # 3) Keep only the relevant columns from pgeocode
    #    - postal_code  → to merge on
    #    - state_name   → region_name
    #    - county_name  → province_name
    geo_df = geo_df[["postal_code", "state_name", "county_name"]].dropna(
        subset=["postal_code"]
    )

    # 4) Rename columns to the names we want in our pipeline
    geo_df = geo_df.rename(
        columns={
            "postal_code": "post_code",
            "state_name": "region_name",
            "county_name": "province_name",
        }
    )

    clean_data["postal_code"] = clean_data["postal_code"].astype(str)
    geo_df["post_code"] = geo_df["post_code"].astype(str)


    # Merge into encoded_df
    clean_data = clean_data.merge(
        geo_df[["post_code", "province_name", "region_name"]],
        left_on="postal_code",
        right_on="post_code",
        how="left",
    )

    # Frequency encoding
    prov_freq = clean_data["province_name"].value_counts(normalize=True)
    reg_freq  = clean_data["region_name"].value_counts(normalize=True)

    clean_data["province_encoded"] = clean_data["province_name"].map(prov_freq)
    clean_data["region_encoded"]   = clean_data["region_name"].map(reg_freq)

    return clean_data[['postal_code', 'province_encoded', 'region_encoded']]

def preprocessing_create_obj(clean_data = pd.DataFrame):

    scaler = RobustScaler()
    scaler.fit_transform(clean_data[["price"]])
    #price_scaled = scaler.fit_transform(price.reshape(-1, 1))
    dump(scaler, "./price_scaler.joblib")

    # properties_data.drop(columns=["garden_surface", "terrace_surface", 'swimming_pool', "locality_name", 'equipped_kitchen', 'furnished',
    #   'open_fire'] )

    clean_data.state_of_building = clean_data.state_of_building.replace("To be renovated", "To renovate")

    # scaling of numerical data
    # to_scale = ["living_area", "number_of_rooms"]
    #scaling(properties_data, to_scale) #scaling of numeric data with wide range
    

    #encoding of categorical data
    to_encode = ['type_of_property', 'subtype_of_property', 'state_of_building']
    encoder = OneHotEncoder(sparse_output=False)

    one_hot_df = pd.DataFrame(encoder.fit_transform(clean_data[to_encode]), 
                              columns=encoder.get_feature_names_out(to_encode))
    
    encoded_df = pd.concat([clean_data, one_hot_df], axis=1)
    encoded_df = encoded_df.drop(to_encode, axis=1)

    return scaler, encoder

def preprocessing_new_data(clean_data, properties_data = pd.DataFrame):


    scaler, encoder = preprocessing_create_obj(clean_data)

    to_encode = ['type_of_property', 'subtype_of_property', 'state_of_building']
    one_hot_df = pd.DataFrame(encoder.transform(properties_data[to_encode]), columns=encoder.get_feature_names_out(to_encode))
    encoded_df = pd.concat([properties_data, one_hot_df], axis=1)
    encoded_df = encoded_df.drop(to_encode, axis=1)
   
    postal_price_map = create_postal_price_df(clean_data)
    user_postal_code = encoded_df['postal_code'].values[0]
    encoded_df['postal_encoded'] = postal_price_map['postal_encoded'][postal_price_map['postal_code']==user_postal_code].values[0]
     
    region_prov_map = frequency_encoder_reg_prov(clean_data)
    print(region_prov_map)
    encoded_df['province_encoded'] = region_prov_map['province_encoded'][region_prov_map['postal_code']==user_postal_code]
    encoded_df['region_encoded'] = region_prov_map['region_encoded'][region_prov_map['postal_code']==user_postal_code]

    # Drop unused columns
    df_model = encoded_df.drop(
        columns=[
            "locality_name",
            "postal_code",
        ]
    )

    return df_model