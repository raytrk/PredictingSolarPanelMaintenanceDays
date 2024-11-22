import sqlite3
import pandas as pd

# Local imports
import models
import preprocessing

if __name__ == "__main__":
    conn = sqlite3.connect(r"data/weather.db")
    w_df = pd.read_sql_query("SELECT * from weather", conn)
    conn = sqlite3.connect(r"data/air_quality.db")
    aq_df = pd.read_sql_query("SELECT * from air_quality", conn)
    combined_df = preprocessing.preprocessing(aq_df, w_df)
    models.run_models(combined_df)
