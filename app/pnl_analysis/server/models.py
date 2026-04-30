from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"
    REVOKED = "REVOKED"


class TaskResponse(BaseModel):
    task_id: str = Field(..., description="Celery task id")
    status: TaskStatus = Field(..., description="Task state")
    message: str = Field(..., description="Human-readable status message")


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ZetaForecastParams(BaseModel):
    """Optional overrides for zeta_modelling wash forecast (passed through Celery payload)."""

    model_config = ConfigDict(extra="ignore")

    margin_per_wash: float = Field(4.0, description="Margin per wash for break-even style fields inside zeta forecast.")
    fixed_monthly_cost: float = Field(50_000.0)
    ramp_up_cost: float = Field(150_000.0)
    scenario: str = Field("Expected", description="Expected | Conservative | Aggressive")
    forecast_months: int = Field(48, ge=12, le=60)
    target_calibration_coverage: float = Field(0.80, ge=0.1, le=0.99)
    forecast_start_date: str = Field("2026-01-01", description="First month of synthetic timeline (ISO date).")
    enable_mature_yoy_control: bool = Field(
        True,
        description="If true, may constrain late-year annual totals to a YoY band only when those years are strictly decreasing YoY; otherwise raw forecast is unchanged.",
    )
    mature_yoy_start_year: int = Field(3, ge=2, le=10, description="Forecast year index (1-based) when YoY band starts.")
    mature_min_yoy: float = Field(0.005, ge=0.0, le=0.5, description="Minimum YoY growth vs prior year total (fraction).")
    mature_max_yoy: float = Field(0.05, ge=0.0, le=1.0, description="Maximum YoY growth vs prior year total (fraction).")
    cluster_distance_policy: str = Field(
        "regional",
        description="regional | fixed. regional uses: Northeast=100, Midwest=150, South=160, West=180 km.",
    )
    max_cluster_distance_km: Optional[float] = Field(
        None,
        gt=0.0,
        description="Used when cluster_distance_policy='fixed'. If omitted, defaults to 100 km.",
    )


class ClusteringV2ProjectionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    address: Optional[str] = Field(None, description="Address (optional if latitude/longitude provided).")
    latitude: Optional[float] = Field(None, description="Latitude.")
    longitude: Optional[float] = Field(None, description="Longitude.")
    method: str = Field("arima", description="Time series method: 'arima', 'holt_winters', or 'blend'.")
    level_model: str = Field("ridge", description="Level model: 'ridge' or 'rf' (if RF checkpoints exist).")
    use_opening_prefix_for_mature_forecast: bool = Field(
        True,
        description="If true, append <2y monthly forecast as context before mature extrapolation.",
    )
    bridge_opening_to_mature_when_prefix: bool = Field(
        True,
        description="If true, bridge month 24→25 when prefix is used.",
    )
    allow_nearest_cluster_beyond_distance_cap: bool = Field(
        False,
        description="If true, allow nearest centroid assignment beyond the 20 km cap.",
    )

    # Optional PnL inputs (to compute revenue/opex/profit summary on top of wash projections).
    wash_prices: Optional[List[float]] = Field(None, description="List of wash tier prices in dollars.")
    wash_pcts: Optional[List[float]] = Field(None, description="List of wash tier user share percentages (0–100).")
    opex_years: Optional[List[float]] = Field(None, description="Operating expense by year: [y1,y2,y3,y4,y5] (or legacy 4-year list).")
    zeta_forecast: Optional[ZetaForecastParams] = Field(
        None,
        description="Optional zeta_modelling parameters (same shape as central form).",
    )

    @model_validator(mode="before")
    @classmethod
    def _location_from_customer_information(cls, data: Any) -> Any:
        """Allow the same nested `customer_information` block as /v1/input-form for this endpoint."""
        if not isinstance(data, dict):
            return data
        data = {**data}
        ci = data.get("customer_information")
        if isinstance(ci, dict):
            if not (str(data.get("address") or "").strip()):
                site = ci.get("site_address")
                if isinstance(site, str) and site.strip():
                    data["address"] = site.strip()
            if data.get("latitude") is None and ci.get("latitude") is not None:
                data["latitude"] = ci["latitude"]
            if data.get("longitude") is None and ci.get("longitude") is not None:
                data["longitude"] = ci["longitude"]
        return data


def _default_menu_packages() -> List["MenuPackageRow"]:
    names = [
        "Basic Package",
        "Menu Package One",
        "Menu Package Two",
        "Menu Package Three",
        "Menu Package Four",
    ]
    return [MenuPackageRow(package_name=n) for n in names]


def _default_acquisition_budget() -> List["AcquisitionBudgetRow"]:
    return [
        AcquisitionBudgetRow(category="land"),
        AcquisitionBudgetRow(category="building"),
        AcquisitionBudgetRow(category="equipment"),
    ]


def _default_bank_debt_allocation() -> List["BankDebtAllocationRow"]:
    return [
        BankDebtAllocationRow(category="land"),
        BankDebtAllocationRow(category="building"),
        BankDebtAllocationRow(category="equipment"),
    ]


def _default_operational_expenses() -> List["OperationalExpenseRow"]:
    return [
        OperationalExpenseRow(category="labor"),
        OperationalExpenseRow(category="utilities"),
        OperationalExpenseRow(category="maintenance"),
    ]


class CustomerInformation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    customer_name: str = ""
    company_name: str = ""
    site_address: str = ""
    express_tunnel: str = ""
    tunnel_count: int = 1
    latitude: Optional[float] = Field(None, description="Optional; use with longitude if no geocodable address.")
    longitude: Optional[float] = Field(None, description="Optional; use with latitude if no geocodable address.")


class LaborRoleHoursWages(BaseModel):
    model_config = ConfigDict(extra="ignore")

    labor_hours: Optional[float] = None
    hourly_wages: Optional[float] = None
    burden_rate: Optional[float] = None


class AttendantsLabor(BaseModel):
    model_config = ConfigDict(extra="ignore")

    number_of_attendants: Optional[float] = None
    labor_hours: Optional[float] = None
    hourly_wages: Optional[float] = None
    burden_rate: Optional[float] = None


class LaborInformation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    manager: LaborRoleHoursWages = Field(default_factory=LaborRoleHoursWages)
    assistant_manager: LaborRoleHoursWages = Field(default_factory=LaborRoleHoursWages)
    attendants: AttendantsLabor = Field(default_factory=AttendantsLabor)


class MenuPackageRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    package_name: str = ""
    price: Optional[float] = None
    customer_percentage: Optional[float] = None
    chemical_cost: Optional[float] = None


class SiteFactors(BaseModel):
    model_config = ConfigDict(extra="ignore")

    area_profile: str = ""
    nearest_competition: str = ""
    type_of_site: str = ""
    site_accessibility: str = ""
    visibility: str = ""
    entrance_stack_up_area: str = ""
    number_of_free_vacuum_slots: Optional[float] = None
    number_of_pay_stations: Optional[float] = None
    traffic_speed: str = ""


class DemographicComponents(BaseModel):
    model_config = ConfigDict(extra="ignore")

    average_household_size: Optional[float] = None
    population_25_65_percent: Optional[float] = None
    households_income_percent: Optional[float] = None
    base_price_of_car_wash: Optional[float] = None


class TrafficInputs(BaseModel):
    model_config = ConfigDict(extra="ignore")

    traffic_count: Optional[float] = None
    year_1_increase_percent: Optional[float] = None
    year_3_increase_percent: Optional[float] = None
    year_5_increase_percent: Optional[float] = None


class AcquisitionBudgetRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    category: str = ""
    total_investment: Optional[float] = None
    owner_percent: Optional[float] = None
    bank_percent: Optional[float] = None


class BankDebtAllocationRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    category: str = ""
    bank_debt_total: Optional[float] = None
    interest_rate: Optional[float] = None
    loan_term_months: Optional[float] = None
    # Keep parity with variants currently sent by clients for some categories.
    total_investment: Optional[float] = None
    owner_percent: Optional[float] = None
    bank_percent: Optional[float] = None


class OperationalExpenseRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    category: str = ""
    percent_of_sales: Optional[float] = None
    year_1_value: Optional[float] = None


class FinancialInputs(BaseModel):
    model_config = ConfigDict(extra="allow")

    acquisition_budget: List[AcquisitionBudgetRow] = Field(default_factory=_default_acquisition_budget)
    bank_debt_allocation: List[BankDebtAllocationRow] = Field(default_factory=_default_bank_debt_allocation)
    operational_expenses: List[OperationalExpenseRow] = Field(default_factory=_default_operational_expenses)
    capex_initial: Optional[float] = Field(None, description="Initial capex ($).")
    zeta_forecast: Optional[ZetaForecastParams] = Field(
        None,
        description="Zeta forecast parameters; preferred placement in updated input form.",
    )
    expense_description: str = ""


class CentralInputFormRequest(BaseModel):
    """
    Full central PnL input form. Optional legacy root keys `address`, `latitude`, `longitude`
    are merged into customer_information for older clients.
    """

    model_config = ConfigDict(extra="allow")

    customer_information: CustomerInformation = Field(default_factory=CustomerInformation)
    labor_information: LaborInformation = Field(default_factory=LaborInformation)
    menu_packages: List[MenuPackageRow] = Field(default_factory=_default_menu_packages)
    site_factors: SiteFactors = Field(default_factory=SiteFactors)
    demographic_components: DemographicComponents = Field(default_factory=DemographicComponents)
    traffic: TrafficInputs = Field(default_factory=TrafficInputs)
    financial_inputs: FinancialInputs = Field(default_factory=FinancialInputs)

    # Legacy optional PnL shortcuts (used by worker when menu_packages rows lack price/pct).
    wash_prices: Optional[List[float]] = Field(None, description="Optional legacy: wash tier prices.")
    wash_pcts: Optional[List[float]] = Field(None, description="Optional legacy: wash tier mix percentages.")
    opex_years: Optional[List[float]] = Field(None, description="Optional: operating expense by year [y1..y5] (legacy [y1..y4] also accepted).")
    capex_initial: Optional[float] = Field(None, description="Optional legacy: initial capex ($).")
    zeta_forecast: Optional[ZetaForecastParams] = Field(
        None,
        description="Optional zeta_modelling (model_1/data_1) forecast parameters; defaults are used if omitted.",
    )

    @model_validator(mode="before")
    @classmethod
    def _merge_legacy_root_location(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        ci = data.get("customer_information")
        addr = data.get("address")
        lat = data.get("latitude")
        lon = data.get("longitude")
        if ci is None and (addr or lat is not None or lon is not None):
            data = {**data}
            data["customer_information"] = {
                "site_address": (addr or "") if isinstance(addr, str) else "",
                "latitude": lat,
                "longitude": lon,
            }
        elif isinstance(ci, dict) and (addr or lat is not None or lon is not None):
            data = {**data}
            merged = {**ci}
            if addr and not str(merged.get("site_address") or "").strip():
                merged["site_address"] = addr
            if lat is not None and merged.get("latitude") is None:
                merged["latitude"] = lat
            if lon is not None and merged.get("longitude") is None:
                merged["longitude"] = lon
            data["customer_information"] = merged
        return data

    @model_validator(mode="after")
    def _require_site_or_coordinates(self) -> CentralInputFormRequest:
        ci = self.customer_information
        site = (ci.site_address or "").strip()
        if site or (ci.latitude is not None and ci.longitude is not None):
            return self
        raise ValueError("Provide customer_information.site_address or both customer_information.latitude and longitude.")

    def to_task_payload(self) -> Dict[str, Any]:
        """Serializable dict for Celery (full nested form plus optional legacy wash/opex fields)."""
        return self.model_dump(mode="json")
