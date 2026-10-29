| run | crops | top-1 | top-5 | loss | per-class top-1 mean / median / min / max | time |
|---|---|---|---|---|---|---|
| lr 0.04 | 1 | 56.54% | 86.58% | 1.642 | 56.5 / 57.0 / 2 / 97% | 26 s |
| lr 0.04, 10-crop | 10 | 57.18% | 87.07% | 1.622 | 57.2 / 59.0 / 4 / 98% | 139 s |
| lr 0.01 | 1 | 56.21% | 86.32% | 1.661 | 56.2 / 57.0 / 4 / 97% | 19 s |

Details for lr 0.04.

15 worst classes:

- 2% building_facade
- 2% river
- 8% mountain
- 13% cafeteria
- 14% valley
- 15% office_building
- 16% botanical_garden
- 16% coast
- 18% house
- 18% ocean
- 19% museum/outdoor
- 19% stadium/football
- 19% yard
- 20% lake/natural
- 21% arch

15 best classes:

- 97% boxing_ring
- 95% cockpit
- 94% basketball_court/indoor
- 94% bamboo_forest
- 93% arena/hockey
- 92% volleyball_court/outdoor
- 92% bowling_alley
- 91% phone_booth
- 90% wind_farm
- 89% windmill
- 89% raceway
- 89% hangar/indoor
- 89% car_interior
- 88% army_base
- 87% underwater/ocean_deep

Most-confused pairs (true -> predicted, % of the true class's images):

- 35% (35) stadium/football -> stadium/soccer
- 32% (32) stadium/football -> football_field
- 27% (27) ice_skating_rink/indoor -> arena/hockey
- 22% (22) assembly_line -> auto_factory
- 22% (22) chemistry_lab -> biology_laboratory
- 22% (22) laundromat -> utility_room
- 22% (22) shower -> bathroom
- 21% (21) airfield -> runway
- 21% (21) downtown -> skyscraper
- 20% (20) classroom -> lecture_room
- 20% (20) football_field -> stadium/football
- 19% (19) beer_hall -> pub/indoor
- 19% (19) conference_center -> legislative_chamber
- 19% (19) jacuzzi/indoor -> swimming_pool/indoor
- 19% (19) shopping_mall/indoor -> atrium/public
- 18% (18) art_studio -> art_gallery
- 18% (18) bus_interior -> train_interior
- 18% (18) dining_hall -> banquet_hall
- 18% (18) drugstore -> pharmacy
- 18% (18) ice_skating_rink/outdoor -> ice_skating_rink/indoor
- 18% (18) office -> office_cubicles
- 17% (17) aqueduct -> viaduct
- 17% (17) canal/natural -> moat/water
- 17% (17) pagoda -> temple/asia
- 16% (16) athletic_field/outdoor -> soccer_field
- 16% (16) baseball_field -> stadium/baseball
- 16% (16) diner/outdoor -> general_store/outdoor
- 16% (16) office -> home_office
- 16% (16) snowfield -> ski_slope
- 16% (16) train_station/platform -> railroad_track

Classes below 30%: 32; above 80%: 36.
