import numpy as np
import itertools

from module_BCHI.config import *

import os,sys

## FUNCTIONS -METRICS/ENCODING

def metric_on_adj_func(points,adjacents,far_distance = 1.732):
    metric_arr=np.zeros((len(points),len(points)))
    for a,b in itertools.product(points,repeat=2):
        if a == b : metric_arr[a,b] = 0
        elif {a,b} in adjacents : metric_arr[a,b] = 1
        else : metric_arr[a,b] = far_distance 
    return lambda x,y : metric_arr[x,y]

def metric_binary_func(x,y):
    return 0 if x==y else 1

def metric_on_adj_arr(points,adjacents,far_distance = 1.732):
    metric_arr=np.zeros((len(points),len(points)))
    for a,b in itertools.product(points,repeat=2):
        if a == b : metric_arr[a,b] = 0
        elif {a,b} in adjacents : metric_arr[a,b] = 1
        else : metric_arr[a,b] = far_distance 
    return metric_arr

metric_binary_arr = np.array([[0,1],[1,0]])

def encoding_adj(encode_info,adjacents):
    return [
        set(map(lambda x : encode_info[x],ele))
        for ele in adjacents
    ]

encoded_metric_func = {
    'strata_race_label': metric_on_adj_func(range(len(entire_race)),
                                       encoding_adj(encode_race,race_adjacent)),
    'strata_sex_label': metric_on_adj_func(range(len(entire_sex)),
                                      encoding_adj(encode_sex,sex_adjacent)),
    'geo_strata_region' : metric_on_adj_func(range(len(entire_region)),
                                        encoding_adj(encode_region,region_adjacent)),
    'geo_strata_poverty' : metric_binary_func,
    'geo_strata_Population' : metric_binary_func,
    'geo_strata_PopDensity' : metric_binary_func,
    'geo_strata_Segregation' : metric_binary_func,
}

encoded_metric_arr = {
    'strata_race_label': metric_on_adj_arr(range(len(entire_race)),
                                       encoding_adj(encode_race,race_adjacent)),
    'strata_sex_label': metric_on_adj_arr(range(len(entire_sex)),
                                      encoding_adj(encode_sex,sex_adjacent)),
    'geo_strata_region' : metric_on_adj_arr(range(len(entire_region)),
                                        encoding_adj(encode_region,region_adjacent)),
    'geo_strata_poverty' : metric_binary_arr,
    'geo_strata_Population' : metric_binary_arr,
    'geo_strata_PopDensity' : metric_binary_arr,
    'geo_strata_Segregation' : metric_binary_arr,
}

def metric_btwn_city_info(X,Y):
    info_dict = {
        0 : 'geo_strata_region',
        1 : 'geo_strata_poverty',
        2 : 'geo_strata_Population',
        3 : 'geo_strata_PopDensity',
        4 : 'geo_strata_Segregation',
    }
    diff = [
        encoded_metric_arr[info_dict[i]][x,y]
        for i, (x,y) in enumerate(zip(X.values,Y.values))
    ]
    return np.linalg.norm(np.array(diff),ord=7)