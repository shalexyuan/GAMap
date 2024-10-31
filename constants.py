coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14
}

category_to_id = [
        "chair",
        "bed",
        "plant",
        "toilet",
        "tv_monitor",
        "sofa"
]

category_to_id_gibson = [
        "chair",
        "couch",
        "potted plant",
        "bed",
        "toilet",
        "tv"
]

mp3d_category_id = {
    'void': 1,
    'chair': 2,
    'sofa': 3,
    'plant': 4,
    'bed': 5,
    'toilet': 6,
    'tv_monitor': 7,
    'table': 8,
    'refrigerator': 9,
    'sink': 10,
    'stairs': 16,
    'fireplace': 12
}

# mp_categories_mapping = [4, 11, 15, 12, 19, 23, 6, 7, 15, 38, 40, 28, 29, 8, 17]

mp_categories_mapping = [4, 11, 15, 12, 19, 23, 26, 24, 28, 38, 21, 16, 14, 6, 16]

hm3d_category = [
        "chair",
        "sofa",
        "plant",
        "bed",
        "toilet",
        "tv_monitor",
        "bathtub",
        "shower",
        "fireplace",
        "appliances",
        "towel",
        "sink",
        "chest_of_drawers",
        "table",
        "stairs"
]

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

color_palette = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999]


cat_att={
  "chair": [
    "chair",
    "sitting",
    "chair leg",
    "chair backrest",
    "chair seat",
    "chair armrest"],
  "bed":[
    "bed",
    "sleeping",
    "lying",
    "bed mattress",
    "bed headboard",
    "bed footboard",
    "bed bedframe"
  ],
  "plant":[
    "plant",
    "plant leaf",
    "plant stem",
    "plant pot",
    "plant branch"
  ],
  "toilet": [
    "defecating",
    "urinating",
    "flushing",
    "flush",
    "lid"
  ],
  "tv_monitor": [
    "displaying",
    "watching",
    "tv screen",
    "tv bezel"
  ],
  "sofa": [
    "sitting",
    "sofa cushion",
    "sofa armrest",
    "sofa backrest",
    "sofa upholstery",
    "sofa leg"]
}

# cat_att_prompt={
#     "chair":        'Seems like there is a object with chair leg, chair backrest, chair seat, and chair armrest',
#     "bed":          'Seems like there is a object that can sleep and lie on it with mattress, and bedframe',
#     "plant":        'Seems like there is a object that has leaf stem branch and with pot',
#     "toilet":       'Seems like there is a object that can defecating, urinating, flushing, sitting and has a lid',
#     "tv_monitor":   'Seems like there is a object that has a screen, and can display something and can be watching',
#     "sofa":         'Seems like there is a object that has cushion, armrest, backrest, upholstery, and can be sit',
# }
# cat_att_prompt={
#     "chair":        'Chair: chair leg, chair backrest, chair seat, and chair armrest',
#     "bed":          'Bed: an object that can sleep and lie on it with mattress, and bedframe',
#     "plant":        'Plant: leaf stem branch and pot',
#     "toilet":       'Toilet: can defecating, urinating, flushing, and has a lid',
#     "tv_monitor":   'TV: has a screen, and can display something and can be watching',
#     "sofa":         'Sofa: cushion, armrest, backrest, upholstery',
# }

cat_att_prompt={
    "chair":        'chair',
    "bed":          'bed',
    "plant":        'plant',
    "toilet":       'toilet',
    "tv_monitor":   'tv monitor',
    "sofa":         'sofa',
}

cat_2_multi_att_prompt={
    "chair":        ['chair','leg','backrest','seat','armrest'],
    "bed":          ['bed','sleep','lie','mattress','bedframe'],
    "plant":        ['plant','leaf','stem','branch','pot'],
    "toilet":       ['toilet','bathroom','defecating','urinating','flushing'],
    "tv_monitor":   ['tv monitor','livingroom','bedroom','screen','display'],
    "sofa":         ['sofa','sit','cushion','armrest','backrest'],
}