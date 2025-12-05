"""
Imputing missing values, encoding, rescaling
"""
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from joblib import dump
from pgeocode import Nominatim

def scaling(group, l): # robust scaling because of the outliers present throughout the categories (extreme big values and skews)
    scaler = RobustScaler()
    group[l] = scaler.fit_transform(group[l])
    #dump(scaler, "./price_scaler.joblib")

def preprocessing(properties_data = pd.DataFrame):

   # properties_data.drop(columns=["garden_surface", "terrace_surface", 'swimming_pool', "locality_name", 'equipped_kitchen', 'furnished',
    #   'open_fire'] )

    properties_data.state_of_building = properties_data.state_of_building.replace("To be renovated", "To renovate")

    # scaling of numerical data
    to_scale = ["living_area", "number_of_rooms"]
    scaling(properties_data, to_scale) #scaling of numeric data with wide range
    scaler = RobustScaler()
    properties_data["price"] = scaler.fit_transform(properties_data[["price"]])
    #price_scaled = scaler.fit_transform(price.reshape(-1, 1))
    dump(scaler, "./price_scaler.joblib")

    #encoding of categorical data
    to_encode = ['type_of_property', 'subtype_of_property', 'state_of_building']
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(properties_data[to_encode])
    one_hot_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(to_encode))

    encoded_df = pd.concat([properties_data, one_hot_df], axis=1)
    encoded_df = encoded_df.drop(to_encode, axis=1)
    """
    # frequency encoding for locality because too many categories
    frequency = encoded_df["locality_name"].value_counts(normalize=True)
    encoded_df["locality_frequency"] = encoded_df["locality_name"].map(frequency)
    """
    #level encoding (median price per code) for postal code
    postal_mean_price = encoded_df.groupby("postal_code")["price"].mean()
    encoded_df["postal_encoded"] = encoded_df["postal_code"].map(postal_mean_price)

    """
    #add provinces and regions
    gdf_pc = gpd.read_file("./postal-codes-belgium.geojson")
    lookup = gdf_pc[["post_code", "province_name_french", "region_name_french"]].drop_duplicates()
    lookup["post_code"] = lookup["post_code"].astype(int)

    encoded_df = encoded_df.merge(lookup, left_on="postal_code", right_on="post_code", how="left")

    prov_freq = encoded_df["province_name_french"].value_counts(normalize=True)
    reg_freq  = encoded_df["region_name_french"].value_counts(normalize=True)

    encoded_df["province_encoded"] = encoded_df["province_name_french"].map(prov_freq)
    encoded_df["region_encoded"]    = encoded_df["region_name_french"].map(reg_freq)

    df_model = encoded_df.drop(columns=[
        "locality_name",
        "postal_code",
        "province_name_french",
        "region_name_french",
    ])

    return df_model
"""

    geo = Nominatim("be")

    # 1) Get unique postal codes as strings
    postal_codes = (
        encoded_df["postal_code"]
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

    encoded_df["postal_code"] = encoded_df["postal_code"].astype(str)
    geo_df["post_code"] = geo_df["post_code"].astype(str)


    # Merge into encoded_df
    encoded_df = encoded_df.merge(
        geo_df[["post_code", "province_name", "region_name"]],
        left_on="postal_code",
        right_on="post_code",
        how="left",
    )

    # Frequency encoding
    prov_freq = encoded_df["province_name"].value_counts(normalize=True)
    reg_freq  = encoded_df["region_name"].value_counts(normalize=True)

    encoded_df["province_encoded"] = encoded_df["province_name"].map(prov_freq)
    encoded_df["region_encoded"]   = encoded_df["region_name"].map(reg_freq)

    # Drop unused columns
    df_model = encoded_df.drop(
        columns=[
            "locality_name",
            "postal_code",
            "post_code",
            "province_name",
            "region_name",
        ]
    )

    return df_model