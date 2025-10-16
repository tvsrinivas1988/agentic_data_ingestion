
import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import uuid

def stg_int_customers():
    engine = create_engine(os.getenv("DATABASE_URL"))
    
    with engine.begin() as conn:
        # Load source and target data
        src_df = pd.read_sql("SELECT * FROM stg.stg_customers", conn)
        tgt_df = pd.read_sql("SELECT customer_code, customer_name, customer_type, address_line1, address_line2, city, state, postal_code, country, phone_number, email, region, territory, process_ind FROM int.customers", conn)

        # Normalize data types
        src_df = src_df.fillna('')
        tgt_df = tgt_df.fillna('')

        # Merge to find new, updated, and deleted records
        merged_df = src_df.merge(tgt_df, on='customer_code', how='outer', indicator=True)

        # Determine process_ind and prepare data for insert/update
        new_rows = merged_df[merged_df['_merge'] == 'left_only']
        updated_rows = merged_df[merged_df['_merge'] == 'both']
        deleted_rows = merged_df[merged_df['_merge'] == 'right_only']

        # Prepare for inserts and updates
        new_rows['process_ind'] = 'I'
        updated_rows['process_ind'] = 'U'

        # Set audit fields
        current_timestamp = datetime.now()
        current_user = 'current_user'  # Replace with actual user retrieval logic
        batch_id = str(uuid.uuid4())

        for col in ['audit_insrt_dt', 'audit_updt_dt']:
            new_rows[col] = current_timestamp
            updated_rows[col] = current_timestamp

        for col in ['audit_insrt_nm', 'audit_updt_nm']:
            new_rows[col] = current_user
            updated_rows[col] = current_user

        updated_rows['audit_batch_id'] = batch_id
        new_rows['audit_batch_id'] = batch_id

        # Insert new rows
        if not new_rows.empty:
            new_rows.to_sql('customers', conn, schema='int', if_exists='append', index=False)

        # Update existing rows
        for index, row in updated_rows.iterrows():
            update_sql = text("""
                UPDATE int.customers
                SET customer_name = :customer_name,
                    customer_type = :customer_type,
                    address_line1 = :address_line1,
                    address_line2 = :address_line2,
                    city = :city,
                    state = :state,
                    postal_code = :postal_code,
                    country = :country,
                    phone_number = :phone_number,
                    email = :email,
                    region = :region,
                    territory = :territory,
                    process_ind = :process_ind,
                    audit_updt_dt = :audit_updt_dt,
                    audit_updt_nm = :audit_updt_nm,
                    audit_batch_id = :audit_batch_id
                WHERE customer_code = :customer_code
            """)
            params = {
                'customer_name': row['customer_name'],
                'customer_type': row['customer_type'],
                'address_line1': row['address_line1'],
                'address_line2': row['address_line2'],
                'city': row['city'],
                'state': row['state'],
                'postal_code': row['postal_code'],
                'country': row['country'],
                'phone_number': row['phone_number'],
                'email': row['email'],
                'region': row['region'],
                'territory': row['territory'],
                'process_ind': row['process_ind'],
                'audit_updt_dt': row['audit_updt_dt'],
                'audit_updt_nm': row['audit_updt_nm'],
                'audit_batch_id': row['audit_batch_id'],
                'customer_code': row['customer_code']
            }
            conn.execute(update_sql, params)

        # Summary
        print(f"Inserted: {len(new_rows)}, Updated: {len(updated_rows)}, Deleted: {len(deleted_rows)}, Skipped: 0")

stg_int_customers()
