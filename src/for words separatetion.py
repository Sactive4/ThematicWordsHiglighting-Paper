import numpy as np
import json
from work_with_files import import_words

file = open("data/working_with_NN_classifier-Vh2-2.1.1/10000 thematic words EDITED V2.txt", "r", encoding="utf-8")
new_thematic_words = np.asarray(file.read().split("\n"))
file.close()
# file = open("../Archive/New NOT thematic words", "r", encoding="utf-8")
file = open("data/working_with_NN_classifier-Vh2-2.1.1/1000 casual words EDITED V2.txt", "r", encoding="utf-8")
new_not_thematic_words = np.asarray(file.read().split("\n"))
file.close()

new_not_thematic_words = np.setdiff1d(new_not_thematic_words, new_thematic_words)

print(new_not_thematic_words)
print(len(new_not_thematic_words))

with open('science_folder/new_not_thematic_words_2.2.1.json', "w", encoding="utf-8") as file:
    dict_json = {"new_not_thematic_words": list(new_not_thematic_words)}
    file.write(json.dumps(dict_json))

thematic_words_list, casual_words_list = import_words()
thematic_words_list = np.asarray(thematic_words_list)
new_thematic_words = list(np.unique(np.setdiff1d(new_thematic_words, thematic_words_list)))

print(new_thematic_words)
print(len(new_thematic_words))

with open('science_folder/new_thematic_words_2.2.1.json', "w", encoding="utf-8") as file:
    dict_json = {"new_thematic_words": list(new_thematic_words)}
    file.write(json.dumps(dict_json))
