# Weather analysis flow — API requests to run

Base URL: `http://localhost:8001/v1` (or your deployed host).

---

## 1. Start analysis (submit address)

**Request**

```bash
curl -X POST "http://localhost:8001/v1/analyze-site" \
  -H "Content-Type: application/json" \
  -d '{"address": "1600 Amphitheatre Parkway, Mountain View, CA"}'
```

**Response**

```json
{
  "task_id": "<uuid>",
  "status": "PENDING",
  "message": "Site successfully submitted for Analysis"
}
```

Save `task_id` for the next requests.

---

## 2. Poll task status (until SUCCESS)

**Request**

```bash
curl -s "http://localhost:8001/v1/task/fb580552-b4bb-44db-bcf7-f071d9d43c68"
```

**Response (while running)**

```json
{
  "task_id": "<task_id>",
  "status": "STARTED",
  "result": { ... partial result if available ... }
}
```

**Response (when done)**

```json
{
  "task_id": "<task_id>",
  "status": "SUCCESS",
  "result": { ... full result ... }
}
```

Poll every 2–3 seconds until `status` is `SUCCESS` (typically ~10–45s depending on quantile + narratives).

---

## 3. Get full result (optional)

**Request**

```bash
curl -s "http://localhost:8001/v1/result/<task_id>"
```

**Response**

Full analysis: `address`, `lat`, `lon`, `feature_values`, `fetched` (climate, gas, stores, competitors), `quantile_result`, `narratives`.

---

## 4. Get weather (metrics + overall narrative) — one request

**Request**

```bash
curl -s "http://localhost:8001/v1/weather/data-by-task/72756c37-7ebd-44d9-99c5-8e317748a091"
```

This single endpoint returns all 4 weather metrics (value, scale, quartile, summary) **and** the overall narrative (insight, observation, conclusion).

**Response shape**

- **complete** — `true` when full data is ready (quantile result + overall narrative); `false` while partial or still building.
- **success** — `true` when `complete` is true (all data available).
- **metrics** — array of 4 objects, each with: `metric_key`, `display_name`, `subtitle`, `value`, `unit`, `min`, `max`, `quantile_score`, `quantile`, `category`, `summary`, `impact_classification` (no business_impact; frontend handles it). Summaries are dynamic (value, percentile vs car wash sites, quartile, category, and whether higher/lower is better).
- **overall** — `insight` (quantile narrative), `observation`, `conclusion` (overall feature takeaway)

**Weather metrics and labels**

| metric_key              | Display name               | Subtitle / description        | Example value + unit        |
|-------------------------|----------------------------|-------------------------------|-----------------------------|
| dirt-trigger-days       | Dirt Trigger Days Window   | Rainy Days                    | 120 days/year               |
| dirt-deposit-severity   | Dirt Deposit Severity      | Total Annual Snowfall          | 18 cm snowfall/year        |
| comfortable-washing-days| Comfortable Washing Days   | Days 60–80°F                  | 165 days/year               |
| shutdown-risk-days      | Shutdown Risk Days         | Days Below Freezing (< 32°F)   | 35 days/year                |

---

*Optional:* `GET /v1/narratives/<task_id>` returns the raw narrative payload (feature array + overall) if you need it elsewhere.

<details>
<summary>Optional: GET /v1/narratives/ response shape</summary>

```json
{
  "task_id": "<task_id>",
  "narratives": {
    "feature": [
      {
        "feature_key": "weather_rainy_days",
        "metric_key": "dirt-trigger-days",
        "label": "Dirt Trigger Days Window",
        "subtitle": "Rainy Days",
        "value": 120,
        "unit": "days/year",
        "category": "Strong",
        "percentile": 90,
        "wash_q": 4,
        "summary": "...",
        "business_impact": "...",
        "impact_classification": "Strong · 100–150 days"
      },
      ...
    ],
    "overall": {
      "insight": "...",      // overall narrative of quantile predictions (weather’s role, predicted band)
      "observation": "...",  // overall feature summary (how the site benefits from the weather profile)
      "conclusion": "..."    // overall takeaway (weather-driven demand perspective)
    }
  }
}
```

- insight — Overall narrative of **quantile predictions** (e.g. weather’s contribution to site potential, predicted wash band).

---

## Caching and TTL

**Yes, there is caching.** Summary:

| What | Where | TTL |
|------|--------|-----|
| **Partial/full analyse-site result** | Redis key `site_analysis:{task_id}` | **1 day** (86,400 s) |
| **Celery task result** | Celery result backend (Redis) | Default (typically 1 day) |
| **Competitor classification** | DB `car_wash_classifications` (by place_id) | Persistent |
| **USA reference climate** | In-memory (weather reference only) | 24 hours |

**Why answers can be quick:** (1) Same **task_id** — result is read from Redis/Celery, no recompute. (2) **Progressive results** — worker writes to Redis after fetch, then quantile, then narratives; polling returns partial data as soon as each stage completes. (3) Same address again = **new task** (no per-address cache).

**Partial result TTL:** `app/celery/tasks.py` sets `RESULT_CACHE_TTL = 86400`; key `site_analysis:{task_id}` is written with `setex(..., TTL, payload)`.

---

## Quick test script (all in one)

```bash
# 1. Submit
RESP=$(curl -s -X POST "http://localhost:8001/v1/analyze-site" \
  -H "Content-Type: application/json" \
  -d '{"address": "1600 Amphitheatre Parkway, Mountain View, CA"}')
TASK_ID=$(echo "$RESP" | jq -r '.task_id')
echo "Task ID: $TASK_ID"

# 2. Wait for SUCCESS
while true; do
  STATUS=$(curl -s "http://localhost:8001/v1/task/$TASK_ID" | jq -r '.status')
  echo "  Status: $STATUS"
  [ "$STATUS" = "SUCCESS" ] && break
  [ "$STATUS" = "FAILURE" ] && break
  sleep 3
done

# 3. Weather (metrics + overall narrative) — one request
curl -s "http://localhost:8001/v1/weather/data-by-task/$TASK_ID" | jq .
```

Or use the Python test script:

```bash
python scripts/test_weather_api.py
```

Output is also written to `logs/weather_api_test_responses.txt`.
