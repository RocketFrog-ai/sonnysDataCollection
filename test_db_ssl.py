import os

import psycopg


def main() -> None:
    host = os.environ.get("PGHOST")
    dbname = os.environ.get("PGDATABASE", "postgres")
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    port = int(os.environ.get("PGPORT", "5432"))
    sslmode = os.environ.get("PGSSLMODE", "require")

    missing = [k for k, v in {"PGHOST": host, "PGUSER": user, "PGPASSWORD": password}.items() if not v]
    if missing:
        raise SystemExit(f"Missing env vars: {', '.join(missing)}")

    with psycopg.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password,
        port=port,
        sslmode=sslmode,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            print(cur.fetchone())


if __name__ == "__main__":
    main()