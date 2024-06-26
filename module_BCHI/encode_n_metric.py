import numpy as np
import itertools

from module_BCHI.config import *

import os,sys

## FUNCTIONS -METRICS/ENCODING

def metric_on_adj(points,adjacents,far_distance = 1.732):
    metric_dict=dict()
    for a,b in itertools.product(points,repeat=2):
        if a == b : metric_dict[(a,b)] = 0
        elif {a,b} in adjacents : metric_dict[(a,b)] = 1
        else : metric_dict[(a,b)] = far_distance 
    return lambda x,y : metric_dict[(x,y)]

def metric_binary(x,y):
    return 0 if x==y else 1

def encoding_adj(encode_info,adjacents):
    return [
        set(map(lambda x : encode_info[x],ele))
        for ele in adjacents
    ]

encoded_metric_dict = {
    'strata_race_label': metric_on_adj(range(len(entire_race)),
                                       encoding_adj(encode_race,race_adjacent)),
    'strata_sex_label': metric_on_adj(range(len(entire_sex)),
                                      encoding_adj(encode_sex,sex_adjacent)),
    'geo_strata_region' : metric_on_adj(range(len(entire_region)),
                                        encoding_adj(encode_region,region_adjacent)),
    'geo_strata_poverty' : metric_binary,
    'geo_strata_Population' : metric_binary,
    'geo_strata_PopDensity' : metric_binary,
    'geo_strata_Segregation' : metric_binary,
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
        encoded_metric_dict[info_dict[i]](x,y)
        for i, (x,y) in enumerate(zip(X.values,Y.values))
    ]
    return np.linalg.norm(np.array(diff),ord=7)