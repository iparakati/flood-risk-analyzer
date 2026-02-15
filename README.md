# 🌊 Flood Risk Analyzer

Mapping flood insurance coverage gaps across the U.S. using FEMA's public data.

## The Problem

Flood risk is shifting. Climate patterns are changing, development keeps pushing into flood-prone areas, and a lot of properties that need flood insurance don't have it. FEMA publishes detailed data on every NFIP claim and policy going back decades, but it's spread across multiple API endpoints and not easy to work with out of the box.

This project pulls that data together, runs some analysis on where coverage gaps are growing, and puts it in a dashboard so you can actually see what's happening at the county and ZIP level.

## What It Does

- **Ingests NFIP claims and policy data** from the OpenFEMA API (2M+ claims, 80M+ policy records)
- **Enriches with NOAA climate data** and Census tract demographics
- **Identifies coverage gaps** — areas where claims are rising but policy counts aren't keeping up
- **Time-series forecasting** on claim frequency and severity by region
- **Interactive dashboard** with county-level maps and drill-down views

## Tech Stack

```
Python 3.11+
FastAPI          API layer
Streamlit        Dashboard
scikit-learn     ML models (time-series, clustering)
pandas / geopandas   Data processing
Plotly / Folium  Mapping
AWS (S3, EC2)    Deployment
OpenFEMA API     Primary data source
NOAA API         Climate enrichment
```

## Data Sources

All public, no API keys required:

- [OpenFEMA NFIP Claims](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2) — redacted claims with loss amounts, dates, flood zones
- [OpenFEMA NFIP Policies](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-policies-v2) — active and historical policy data
- [FEMA Disaster Declarations](https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2) — every federal disaster declaration since 1953
- [NOAA Climate Data](https://www.ncdc.noaa.gov/cdo-web/) — precipitation, temperature, storm events

## Status

🚧 **In progress** — Building in public. Check back for updates or watch the repo.

## Project Structure

```
src/
  ingestion/          Data ingestion pipeline
    openfema_client.py  OpenFEMA API client with pagination & caching
    noaa_client.py      NOAA Climate Data Online API client
    s3_upload.py        AWS S3 upload utilities
  analysis/           Analytics modules
    coverage_gap.py     Coverage gap model with KMeans clustering
  forecasting/        Time-series forecasting
    timeseries.py       SARIMAX + Prophet forecasting pipeline
  dashboard/          Streamlit frontend
    app.py              Interactive dashboard with Folium maps
  api/                REST API
    main.py             FastAPI endpoints for data, gaps & forecasts
  config.py           Environment & path configuration
notebooks/
  01_exploratory_analysis.ipynb   EDA notebook
tests/                Unit tests
```

## Quick Start

```bash
pip install -r requirements.txt

# Run the dashboard
streamlit run src/dashboard/app.py

# Run the API server
uvicorn src.api.main:app --reload

# Run tests
pytest tests/
```

## Roadmap

- [x] Data ingestion pipeline (OpenFEMA → S3)
- [x] Exploratory analysis notebooks
- [x] Coverage gap model (claims vs. policies by geography)
- [x] Time-series forecasting (Prophet / statsmodels)
- [x] Streamlit dashboard with Folium maps
- [ ] Deploy to AWS
- [ ] Write-up / case study blog post

## License

MIT
