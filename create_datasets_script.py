
"""

Author: Joseph Morgan
Date: 19/09/2021
Title: create_datasets_scripts.py
Configuration script to create the H5 datasets for each image dataset
"""
from images_to_dataset import CreateDataset

pigeon_dict = {"Alexander": 0,
               "Bertie": 1,
               "Constantine": 2,
               "Edward": 3,
               "Friedrich": 4,
               "George": 5,
               "Haakon": 6,
               "Harald": 7,
               "Henry": 8,
               "James": 9,
               "Nicholas": 10,
               "Olav": 11,
               "Oscar": 12,
               "Paul": 13,
               "Peter": 14,
               "Wilhelm": 15,
               "William": 16}

fish_dict = {"Catherine": 0,
             "Dwayne": 1,
             "Florence": 2,
             "Humphrey": 3,
             "Jack": 4,
             "JP": 5,
             "Ruby": 6,
             "Selwyn": 7,
             "Siobhan": 8}

dataset = CreateDataset(pigeon_dict, "animal_data/PigeonImages")
# dataset = CreateDataset(fish_dict,"animal_data/FishImages")

dataset.create_dataset()
dataset.create_h5df()
