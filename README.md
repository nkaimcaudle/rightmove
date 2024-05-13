parallel -j3 --delay 1.32 --bar python get_property_details.py {} :::: codes
head -64 floorplans.csv | parallel -j1 --delay 0.6 --bar curl -s {} --output data/rightmove.floorplan.{/}
