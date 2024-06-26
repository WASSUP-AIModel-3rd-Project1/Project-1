import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import missingno as msno
import seaborn as sns 
import os
import functools, itertools
import argparse, sys
import ipdb

import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("png2x")
# 테마 설정: "default", "classic", "dark_background", "fivethirtyeight", "seaborn"
mpl.style.use("fivethirtyeight")
# 이미지가 레이아웃 안으로 들어오도록 함
mpl.rcParams.update({"figure.constrained_layout.use": True})

import matplotlib.font_manager as fm
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
[fm.FontProperties(fname=font).get_name() for font in font_list if 'D2C' in font]
plt.rc('font', family='D2Coding')
mpl.rcParams['axes.unicode_minus'] = False

## settings
DATASET_DIR = '/home/doeun/code/AI/ESTSOFT2024/workspace/dataset/'
RAW_DATA_DIR = os.path.join(DATASET_DIR,'america_big_cities_health_inventory')
RAW_FILE_NAME = 'BigCitiesHealth.csv'
RSLT_DIR = '/home/doeun/code/AI/ESTSOFT2024/workspace/1.project1_structured/BCHI/processed/'
PVTB_DIR = RSLT_DIR + 'pvtb/'

## FUNCTIONS - DF PROCESS

def cond_check_dict(data=pd.DataFrame,val_dict=dict):
    cond_list=[
        data[col] == val
        for col, val in val_dict.items()
    ]
    return functools.reduce(lambda x,y: x & y, cond_list)

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

## CONSTANTS - ENCODING/METRIC

entire_region = [ "Midwest", "Northeast", "South", "West" ]
region_adjacent = [
    { "Midwest", "Northeast" },
    { "Midwest", "South" },
    { "Midwest", "West" },
    { "West", "South" },
    { "South", "Northeast" }
]
encode_region ={
    "Midwest": 0,
    "Northeast": 1,
    "South": 2,
    "West": 3 
}

entire_race = ['All','White','Black','Hispanic','Asian/PI','Natives']
race_adjacent = [
    {'All', a}
    for a in entire_race
    if a != 'All'
]
encode_race={
    'All' : 0,
    'White' : 1,
    'Black' : 2,
    'Hispanic' : 3,
    'Asian/PI' : 4,
    'Natives' : 5,
}

entire_sex = ['Both','Female','Male']
sex_adjacent = [
    {'Both', a}
    for a in entire_sex
    if a != 'Both'
]
encode_sex = {
    'Both' : 0,
    'Female' : 1,
    'Male' : 2,
}

encode_poverty = {
    'Poorest cities (18%+ poor)' : 0,
    'Less poor cities (<18% poor)' : 1,
}
encode_pop = {
    'Smaller (<1.3 million)' : 0,
    'Largest (>1.3 million)' : 1,
}
encode_popdensity = {
    'Lower pop. density (<10k per sq mi)' : 0,
    'Highest pop. density (>10k per sq mi)' : 1,
}
encode_segregation = {
    'Less Segregated (<50%)' : 0,
    'Highly Segregated (50%+)' : 1,
}

encode_dict ={
    'strata_race_label': encode_race,
    'strata_sex_label': encode_sex,
    'geo_strata_region' : encode_region,
    'geo_strata_poverty' : encode_poverty,
    'geo_strata_Population' : encode_pop,
    'geo_strata_PopDensity' : encode_popdensity,
    'geo_strata_Segregation' : encode_segregation,
}

info_cols = [
    'city_idx',
    'encoded_strata_race_label',
    'encoded_strata_sex_label',
    'encoded_geo_strata_region',
    'encoded_geo_strata_poverty',
    'encoded_geo_strata_Population',
    'encoded_geo_strata_PopDensity',
    'encoded_geo_strata_Segregation',
    'date_label',
]

geo_strat_cols = [
    'geo_strata_region' ,
    'geo_strata_poverty' ,
    'geo_strata_Population' ,
    'geo_strata_PopDensity' ,
    'geo_strata_Segregation' ,
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
        for i, (x,y) in enumerate(zip(X,Y))
    ]
    return np.linalg.norm(np.array(diff),ord=7)

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
        for i, (x,y) in enumerate(zip(X,Y))
    ]
    return np.linalg.norm(np.array(diff),ord=7)

dict_metric = {
    'strata_race_label': metric_on_adj(entire_race,race_adjacent),
    'strata_sex_label': metric_on_adj(encode_sex,sex_adjacent),
    'geo_strata_region' : metric_on_adj(encode_region,region_adjacent),
    'geo_strata_poverty' : metric_binary,
    'geo_strata_Population' : metric_binary,
    'geo_strata_PopDensity' : metric_binary,
    'geo_strata_Segregation' : metric_binary,
}

def make_metric_val(geo_info,geo_cols):
    city_list = list(geo_info.index)
    metric_val= dict()
    for a, b in itertools.product(city_list,repeat=2):
        diff = [
            dict_metric[col](geo_info.loc[a,col],
                             geo_info.loc[b,col])
            for col in geo_cols
            ]
        metric_val[(a,b)] = np.linalg.norm(np.array(diff),ord=7)
    return metric_val

def make_geo_strat_table(df_city:pd.DataFrame,filename='geo_strat_table.csv'):
    geo_strat_cols = list(filter(lambda x : 'geo_strat' in x,df_city.columns))
    df_temp = df_city[geo_strat_cols]
    dup_cond = df_temp.duplicated(keep=False)
    df_geo_strat = df_temp[dup_cond].value_counts().reset_index(drop=False)
    
    save_path = os.path.join(RSLT_DIR,filename)
    df_geo_strat.to_csv(save_path,index=False)
    return df_geo_strat

def search_citytype(geo_strat:pd.Series,df_geo_table):
    geo_strat_dict = geo_strat.to_dict()
    cities = df_geo_table[cond_check_dict(df_geo_table,geo_strat_dict)]
    return list(cities.index)[0]


if __name__ == '__main__' :
    pvtb_name = 'pvtb_city_encoded_ver1.csv'
    pvtb_encoded = pd.read_csv(os.path.join(PVTB_DIR,pvtb_name),index_col=0)
    df_geo_strat = make_geo_strat_table(pvtb_encoded,'geo_strat_encoded.csv')
    
    geo_strat_cols = list(filter(lambda x : 'geo_strat' in x,pvtb_encoded.columns))
    pvtb_encoded['geo_city_type'] = pvtb_encoded[geo_strat_cols].apply(lambda x : search_citytype(x,df_geo_strat),axis=1)
    
    new_cols = ['geo_label_city','geo_city_type'] + list(pvtb_encoded.columns)[1:-1]
    pvtb_renew = pvtb_encoded[new_cols]
    pvtb_renew.to_csv(os.path.join(PVTB_DIR,"pvtb_city_encoded_ver2.csv"),index=False)
    