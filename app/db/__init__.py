"""DB persistence for the FastAPI backend.

Currently one thing lives here: the pin read-through cache (`pin_cache`) backed by the Azure MySQL
`proforma_schema` database (CAR_WASH_DB_URL). It is entirely best-effort — if the DB is unset or
unreachable, every helper degrades to a no-op and the API serves live results. Nothing else in the
repo talks to a database; this is the single, isolated entry point.
"""
