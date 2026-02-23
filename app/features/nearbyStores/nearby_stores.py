from app.features.nearbyStores.nearby_costcos import get_costco_info
from app.features.nearbyStores.nearby_walmart import get_walmart_info
from app.features.nearbyStores.nearby_target import get_target_info
from app.features.nearbyGasStations.get_nearby_gas_stations import get_gas_station_info


def get_nearby_stores_data(latitude: float, longitude: float):
    costco = get_costco_info(latitude, longitude)
    walmart = get_walmart_info(latitude, longitude)
    target = get_target_info(latitude, longitude)
    gas = get_gas_station_info(latitude, longitude)
    out = {
        "distance_from_nearest_costco": costco.get("distance_from_nearest_costco") if costco else None,
        "count_of_costco_5miles": costco.get("count_of_costco_5miles", 0) if costco else 0,
        "nearest_costco": costco.get("nearest_costco") if costco else None,
        "count_of_walmart_5miles": walmart.get("count_of_walmart_5miles", 0) if walmart else 0,
        "distance_from_nearest_walmart": walmart.get("distance_from_nearest_walmart") if walmart else None,
        "nearest_walmart": walmart.get("nearest_walmart") if walmart else None,
        "distance_from_nearest_target": target.get("distance_from_nearest_target") if target else None,
        "count_of_target_5miles": target.get("count_of_target_5miles", 0) if target else 0,
        "nearest_target": target.get("nearest_target") if target else None,
    }
    if gas is not None:
        out["distance_from_nearest_gas_station"] = gas.get("distance_from_nearest_gas_station")
        out["count_of_gas_stations_5miles"] = gas.get("count_of_gas_stations_5miles", 0) or 0
    return out
