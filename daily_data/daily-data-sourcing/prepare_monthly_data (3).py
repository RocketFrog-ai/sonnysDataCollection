import json
import pandas as pd
import numpy as np

import psycopg2
import pandas as pd

from tqdm import tqdm

def fetch_data_control(table_name, site_id):

    query = f"""
                WITH base AS (
                SELECT
                    created_date_utc::timestamp AS created_ts,
                    retail_or_mem,
                    trans_type_name
                FROM {table_name}
                WHERE site_id = '{site_id}'
                AND retail_or_mem IN ('retail','membership')
            ),

            inception AS (
                SELECT MIN(created_ts) AS inception_date
                FROM base
            ),

            final AS (
                SELECT
                    b.*,
                    i.inception_date,

                    -- Months since inception (0-based)
                    (
                        (DATE_PART('year', b.created_ts) - DATE_PART('year', i.inception_date)) * 12 +
                        (DATE_PART('month', b.created_ts) - DATE_PART('month', i.inception_date))
                    ) AS month_index

                FROM base b
                CROSS JOIN inception i
            )

            SELECT
                -- Month from inception (1-based)
                month_index + 1 AS month_number,

                -- Quarter from inception
                FLOOR(month_index / 3) + 1 AS quarter_number,

                -- Year from inception
                FLOOR(month_index / 12) + 1 AS year_number,

                DATE_TRUNC('month', created_ts) AS calendar_month,

                -- Retail
                SUM(
                    CASE 
                        WHEN retail_or_mem = 'retail' THEN
                            CASE 
                                WHEN trans_type_name = 'Return' THEN -1 
                                ELSE 1 
                            END
                        ELSE 0
                    END
                ) AS wash_count_retail,

                -- Membership
                SUM(
                    CASE 
                        WHEN retail_or_mem = 'membership' THEN
                            CASE 
                                WHEN trans_type_name = 'Return' THEN -1 
                                ELSE 1 
                            END
                        ELSE 0
                    END
                ) AS wash_count_membership,

                -- Total
                SUM(
                    CASE 
                        WHEN trans_type_name = 'Return' THEN -1 
                        ELSE 1 
                    END
                ) AS wash_count_total

            FROM final

            GROUP BY
                month_index,
                calendar_month

            ORDER BY month_index;
            """

    conn = psycopg2.connect(
        host="qv-repgendev-psql1.postgres.database.azure.com",
        port=5432,
        dbname="reportgen",
        user="sondbadmin",
        password="emnHY7vwiAKwvbXiQDfr"
    )

    curr = conn.cursor()
    curr.execute(query)

    rows = curr.fetchall()
    column_names = [desc[0] for desc in curr.description]

    data = pd.DataFrame(rows, columns=column_names)

    curr.close()

    return data

def fetch_data_chem(location_id):
    query = f"""
                WITH base AS (
                SELECT
                    location_id,
                    year,
                    month,
                    net_rev_retail_wash_based,
                    net_rev_member_wash_based,
                    cars_washed,

                    -- Convert year + month → date
                    MAKE_DATE(year, month, 1) AS calendar_month

                FROM chem_car_count
                WHERE location_id = '{location_id}'
            ),

            inception AS (
                SELECT
                    location_id,
                    MIN(calendar_month) AS inception_date
                FROM base
                GROUP by location_id
            ),

            final AS (
                SELECT
                    b.*,
                    i.inception_date,

                    -- Month index from inception
                    (
                        (DATE_PART('year', b.calendar_month) - DATE_PART('year', i.inception_date)) * 12 +
                        (DATE_PART('month', b.calendar_month) - DATE_PART('month', i.inception_date))
                    ) AS month_index

                FROM base b
                JOIN inception i
                ON b.location_id = i.location_id
            )

            SELECT
                location_id,

                calendar_month,

                -- Inception-based metrics
                month_index + 1 AS month_number,
                FLOOR(month_index / 3) + 1 AS quarter_number,
                FLOOR(month_index / 12) + 1 AS year_number,

                -- Already aggregated values
                net_rev_retail_wash_based AS wash_count_retail,
                net_rev_member_wash_based AS wash_count_membership,
                cars_washed AS wash_count_total

            FROM final

            ORDER BY calendar_month;
                        
            """
    
    conn = psycopg2.connect(
        host="qv-repgendev-psql1.postgres.database.azure.com",
        port=5432,
        dbname="reportgen",
        user="sondbadmin",
        password="emnHY7vwiAKwvbXiQDfr"
    )

    curr = conn.cursor()
    curr.execute(query)

    rows = curr.fetchall()
    column_names = [desc[0] for desc in curr.description]

    data = pd.DataFrame(rows, columns=column_names)

    curr.close()

    data['wash_count_retail'] = data['wash_count_retail'].astype(float).round().astype(int)
    data['wash_count_membership'] = data['wash_count_membership'].astype(float).round().astype(int)

    return data

MAP_TABLE_CLIENT_ID = {}

names_data = None
with open(r"/home/lakshya.tomar@adsonnysdirect.com/report_gen/special_alogs/gaps.json") as file:
    names_data = json.load(file)

for table_name, sites_metadata in names_data.items():
    if len(list(sites_metadata.values())) > 0:
        client_id = list(sites_metadata.values())[0]["client_id"]
        MAP_TABLE_CLIENT_ID[client_id] = table_name

data = pd.read_csv(r"/home/lakshya.tomar@adsonnysdirect.com/report_gen/special_alogs/data_store/data_less_2_with_source_EC3.csv")

inception_data = None

count = 0
append_data = False

for idx, row in tqdm(data.iterrows(), total=len(data)):

    append_data = False

    if row["source"] == "control" and row["client_id"] in MAP_TABLE_CLIENT_ID.keys():
        temp_data = fetch_data_control(MAP_TABLE_CLIENT_ID[row["client_id"]], row["location_id"])
        append_data = True
    elif row["source"] == "chem":
        temp_data = fetch_data_chem(row["location_id"])
        append_data = True
    else:
        print(f"Skipping......{row["client_id"]} , {row["location_id"]}, {row["site_client_id"]}")

    if append_data == True:

        temp_data["client_id"] = row["client_id"]
        temp_data["location_id"] = row["location_id"]
        temp_data["site_client_id"] = row["site_client_id"]
        temp_data["source"] = row["source"]

        if inception_data is None:
            inception_data = temp_data
        else:
            inception_data = pd.concat([inception_data, temp_data], ignore_index=True)

inception_data.to_csv(r"./montly_data.csv",index=False)