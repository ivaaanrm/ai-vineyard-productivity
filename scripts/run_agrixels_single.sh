docker run --rm -it \
  -e PARAMS_FILE=/workspace/data/input/params_s1.json \
  -v "$(pwd)/data":/workspace/data \
  agrixel-fair:latest


docker run --rm -it \
  -e PARAMS_FILE=/workspace/data/input/params_bulk_era5.json \
  -e CDSAPI_URL=https://cds.climate.copernicus.eu/api \
  -e CDSAPI_KEY=$ERA5_APIKEY \
  -v "$(pwd)/data":/workspace/data \
  agrixel-fair:latest


docker run --rm -it \
  -e PARAMS_FILE=/workspace/data/input/params_bulk_modis.json \
  -e EARTHDATA_USERNAME=$MODIS_USERNAME \
  -e EARTHDATA_PASSWORD=$MODIS_PASSWORD \
  -v "$(pwd)/data":/workspace/data \
  agrixel-fair:latest