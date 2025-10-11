import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def stg_int_brand_master():
    # Database connection
    db_url = os.getenv('DATABASE_URL')
    engine = create_engine(db_url)

    # Read from source table
    query = "SELECT brand_id, brand_name FROM stg.brand_master"
    df = pd.read_sql(query, engine)

    # Apply transformations
    df['process_ind'] = 'Y'
    current_timestamp = datetime.now()
    df['audit_insrt_dt'] = current_timestamp
    df['audit_insrt_nm'] = 'ETL Process'
    df['audit_updt_dt'] = current_timestamp
    df['audit_updt_nm'] = 'ETL Process'
    df['audit_batch_id'] = 'Batch_001'

    # Select only the target columns
    target_columns = [
        'brand_id',
        'brand_name',
        'process_ind',
        'audit_insrt_dt',
        'audit_insrt_nm',
        'audit_updt_dt',
        'audit_updt_nm',
        'audit_batch_id'
    ]
    df = df[target_columns]

    # Write to target table (incremental load)
    df.to_sql('brand_master', con=engine, schema='int', if_exists='append', index=False)

# Call the function to execute the ETL process
stg_int_brand_master()
