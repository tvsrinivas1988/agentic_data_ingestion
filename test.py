import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def stg_int_brand_master():
    # Database connection
    db_url = os.getenv("DATABASE_URL")
    engine = create_engine(db_url)

    # Read from the staging table
    with engine.begin() as conn:
        df = pd.read_sql_table('brand_master', con=conn, schema='stg')

    # Prepare the data for incremental load
    df['process_ind'] = 'I'
    df['audit_insrt_dt'] = datetime.now()
    df['audit_insrt_nm'] = 'ETL Process'
    df['audit_updt_dt'] = datetime.now()
    df['audit_updt_nm'] = 'ETL Process'
    df['audit_batch_id'] = 'Batch_001'

    # Check for existing records in the target table
    existing_brands = pd.read_sql_query("SELECT brand_id FROM int.brand_master", con=engine)
    existing_brands_set = set(existing_brands['brand_id'])

    # Update process indicator for existing records
    df['process_ind'] = df['brand_id'].apply(lambda x: 'U' if x in existing_brands_set else 'I')

    # Separate inserts and updates
    inserts = df[df['process_ind'] == 'I']
    updates = df[df['process_ind'] == 'U']

    # Insert new records
    if not inserts.empty:
        inserts.to_sql('brand_master', con=engine, schema='int', if_exists='append', index=False)
        print(f"Inserted {len(inserts)} new records into int.brand_master.")

    # Update existing records
    if not updates.empty:
        # Create a temporary table for updates
        updates_temp = updates[['brand_id', 'brand_name', 'audit_updt_dt']]
        updates_temp.to_sql('tmp_brand_master', con=engine, index=False, if_exists='replace')

        update_sql = """
        UPDATE int.brand_master AS target
        SET target.brand_name = src.brand_name,
            target.audit_updt_dt = CURRENT_TIMESTAMP
        FROM tmp_brand_master AS src
        WHERE target.brand_id = src.brand_id;
        """
        
        with engine.begin() as conn:
            conn.execute(text(update_sql))
            print(f"Updated {len(updates)} existing records in int.brand_master.")

# Call the function
stg_int_brand_master()
