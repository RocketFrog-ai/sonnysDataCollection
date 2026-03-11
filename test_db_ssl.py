import pymysql

try:
    conn = pymysql.connect(
        host="son-eus2-proforma-mysql02.mysql.database.azure.com",
        user="mysqladmin",
        password="ceu0hqf9jev3KQD*cme",
        database="proforma_schema",
        port=3306,
        ssl={"check_hostname": False}
    )
    print("SUCCESS: Connected with ssl={'check_hostname': False}")
    conn.close()
except Exception as e:
    print(f"FAILED with ssl={{'check_hostname': False}}: {e}")

try:
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    conn = pymysql.connect(
        host="son-eus2-proforma-mysql02.mysql.database.azure.com",
        user="mysqladmin",
        password="ceu0hqf9jev3KQD*cme",
        database="proforma_schema",
        port=3306,
        ssl=ctx
    )
    print("SUCCESS: Connected with ssl=SSLContext")
    conn.close()
except Exception as e:
    print(f"FAILED with ssl=SSLContext: {e}")

