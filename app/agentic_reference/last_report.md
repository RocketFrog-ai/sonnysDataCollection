# 📊 Car Wash Site Profiling Report

## Overall Classification

**High Performing** (fit score: 34.3%)

Expected annual volume: **96,763** (conservative) — **111,385** (likely) — **142,368** (optimistic)

## Dimension-by-Dimension Analysis

### ✓ Weather: **High Performing** (34.6%)

- **total_sunshine_hours** = 3,350.0 — within the IQR of **High Performing** sites (Q25=3,152.9, Median=3,209.9, Q75=3,415.9)
- **days_pleasant_temp** = 160.0 — within the IQR of **High Performing** sites (Q25=133.4, Median=144.7, Q75=160.9)
- **total_precipitation_mm** = 1,400.0 — within the IQR of **High Performing** sites (Q25=1,157.8, Median=1,301.3, Q75=1,442.4)
- **rainy_days** = 120.0 — within the IQR of **Average Performing** sites (Q25=115.9, Median=124.2, Q75=137.7)
- **total_snowfall_cm** = 5.0 — within the IQR of **Low Performing** sites (Q25=1.3, Median=14.1, Q75=50.8)
- **snowy_days** = 3.0 — within the IQR of **Low Performing** sites (Q25=1.0, Median=7.0, Q75=26.2)
- **days_below_freezing** = 15.0 — within the IQR of **Low Performing** sites (Q25=10.3, Median=41.0, Q75=84.1)
- **avg_daily_max_windspeed_ms** = 16.0 — within the IQR of **Low Performing** sites (Q25=15.1, Median=16.4, Q75=19.1)

### ✓ Traffic: **High Performing** (33.5%)

- **Nearest StreetLight US Hourly-Ttl AADT** = 15,000.0 — within the IQR of **High Performing** sites (Q25=5,919.0, Median=10,973.0, Q75=16,876.0)
- **2nd Nearest StreetLight US Hourly-Ttl AADT** = 12,000.0 — within the IQR of **Low Performing** sites (Q25=3,806.0, Median=9,399.0, Q75=14,586.0)
- **3rd Nearest StreetLight US Hourly-Ttl AADT** = 9,000.0 — within the IQR of **Low Performing** sites (Q25=3,600.0, Median=8,267.0, Q75=14,113.0)
- **Nearest StreetLight US Hourly-ttl_breakfast** = 2,500.0 — within the IQR of **Low Performing** sites (Q25=1,049.0, Median=2,176.0, Q75=3,467.0)
- **Nearest StreetLight US Hourly-ttl_lunch** = 3,000.0 — within the IQR of **High Performing** sites (Q25=1,299.0, Median=2,310.0, Q75=3,373.0)
- **Nearest StreetLight US Hourly-ttl_afternoon** = 3,200.0 — within the IQR of **Low Performing** sites (Q25=1,105.0, Median=2,166.0, Q75=3,224.0)
- **Nearest StreetLight US Hourly-ttl_dinner** = 2,800.0 — within the IQR of **Low Performing** sites (Q25=973.0, Median=2,015.0, Q75=2,850.0)
- **Nearest StreetLight US Hourly-ttl_night** = 1,200.0 — within the IQR of **Low Performing** sites (Q25=432.0, Median=970.0, Q75=1,517.0)
- **Nearest StreetLight US Hourly-ttl_overnight** = 300.0 — within the IQR of **Low Performing** sites (Q25=116.0, Median=291.0, Q75=499.0)
- **nearby_traffic_lights_count** = 14.0 — within the IQR of **Low Performing** sites (Q25=6.0, Median=11.0, Q75=22.0)
- **distance_nearest_traffic_light_1** = 0.1 — within the IQR of **Low Performing** sites (Q25=0.1, Median=0.1, Q75=0.2)
- **distance_nearest_traffic_light_2** = 0.2 — within the IQR of **Average Performing** sites (Q25=0.2, Median=0.3, Q75=0.6)
- **distance_nearest_traffic_light_3** = 0.3 — within the IQR of **Low Performing** sites (Q25=0.3, Median=0.5, Q75=0.8)

### ✓ Competition: **High Performing** (39.9%)

- **competitors_count** = 1.0 — within the IQR of **Low Performing** sites (Q25=0.0, Median=1.0, Q75=1.0)
- **competitor_1_distance_miles** = 3.5 — within the IQR of **Low Performing** sites (Q25=0.3, Median=0.9, Q75=10.0)
- **competitor_1_google_user_rating_count** = 200.0 — above the IQR of **High Performing** sites (Q25=0.0, Median=2.0, Q75=155.0)

### ✗ Infrastructure: **Low Performing** (33.3%)

- **tunnel_length (in ft.)** = 115.0 — within the IQR of **Low Performing** sites (Q25=0.0, Median=102.9, Q75=121.8)
- **total_weekly_operational_hours** = 80.0 — within the IQR of **Low Performing** sites (Q25=75.0, Median=83.0, Q75=84.0)

### ✗ Retail Proximity: **Low Performing** (33.3%)

- **Count of ChainXY VT - Grocery** = 2.0 — within the IQR of **Low Performing** sites (Q25=0.0, Median=1.0, Q75=2.0)
- **Count of ChainXY VT - Mass Merchant** = 1.0 — within the IQR of **Low Performing** sites (Q25=1.0, Median=1.0, Q75=1.0)
- **Sum ChainXY** = 5.0 — within the IQR of **Low Performing** sites (Q25=1.0, Median=3.0, Q75=5.0)
- **count_of_costco_5miles** = 1.0 — above the IQR of **Low Performing** sites (Q25=0.0, Median=0.0, Q75=0.0)
- **distance_from_nearest_walmart** = 1.2 — within the IQR of **Low Performing** sites (Q25=0.4, Median=1.3, Q75=2.5)
- **count_of_walmart_5miles** = 2.0 — above the IQR of **Low Performing** sites (Q25=1.0, Median=1.0, Q75=1.0)

## Overall Verdict

🟡 **PROCEED WITH CAUTION** — Mixed signals across dimensions

- **3/5** dimensions agree on **High Performing**
- Strengths: Weather, Traffic, Competition
- Weaknesses: Infrastructure, Retail Proximity

## Recommendations

- ⚠ **Infrastructure** scored Low — investigate mitigation strategies
- ⚠ **Retail Proximity** scored Low — investigate mitigation strategies
- ✓ Leverage strong dimensions: Weather, Traffic, Competition
- Review competitive landscape on the ground before committing