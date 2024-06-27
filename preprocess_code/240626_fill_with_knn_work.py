import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import missingno as msno
import seaborn as sns 
import functools, itertools
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import argparse
from numba import jit

import os,sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module_BCHI.util import *
from module_BCHI.file_io import *
from module_BCHI.plot import *
from module_BCHI.encode_n_metric import *
from module_BCHI.config import *
from module_BCHI.reg_tool import *

## FUNCTIONS - DF PROCESS

def cond_check_dict(data=pd.DataFrame,val_dict=dict):
    cond_list=[
        data[col] == val
        for col, val in val_dict.items()
    ]
    return functools.reduce(lambda x,y: x & y, cond_list)

vec_metric_dict={
    key : np.vectorize(val)
    for key,val in encoded_metric_dict.items()
}

#def metric_btwn_city_info(X,Y):
#    diff = [
#        vec_metric_dict['geo_strata_region'](int(X[0]),int(Y[0])),
#        vec_metric_dict['geo_strata_poverty'](int(X[1]),int(Y[1])),
#        vec_metric_dict['geo_strata_Population'](int(X[2]),int(Y[2])),
#        vec_metric_dict['geo_strata_PopDensity'](int(X[3]),int(Y[3])),
#        vec_metric_dict['geo_strata_Segregation'](int(X[4]),int(Y[4])),
#    ]
#    return np.linalg.norm(np.array(diff),ord=7)

@jit(nopython = True)
def weigted_metric_city(X,Y,metric_dict_city,weight_norm):
    city_idx, race_idx, sex_idx, date_idx = 0, 1, 2, 8
    diff = [
        vec_metric_dict['strata_race_label'](int(X[race_idx]),int(Y[race_idx])),
        vec_metric_dict['strata_sex_label'](int(X[sex_idx]),int(Y[sex_idx])),
        metric_dict_city[(int(X[city_idx]),int(Y[city_idx]))],
        np.abs(X[date_idx]-Y[date_idx]),
            ]
    return np.linalg.norm(np.array(diff)*weight_norm,ord=5)

########################################################################

# def auto_filename : for dict

########################################################################
from sklearn.neighbors import KNeighborsRegressor
import copy, time
from tqdm import tqdm

import ipdb

def prjct_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_work',default=10)
    parser.add_argument('--step',default=1)
    parser.add_argument('--start_idx',default=0)
    args = parser.parse_args()
    n_work = int(args.n_work) 
    start_idx, step = int(args.start_idx), int(args.step)
    work_setting = start_idx, n_work, step
    
    weight_norm_idx = 1 
    weight_func_idx = 0
    metric = 'custom-{}'.format(weight_norm_idx)
    weight_func = 'custom-{}'.format(weight_func_idx)
    
    knn_set={
        'metric' : metric,
        'n_neigh' : 7, 
        'weight' : 'distance' 
    }
    
    col_select = 'test'
    if col_select == 'else' : target_cols = list(filter(lambda x : x not in cand_cols + test_cols,entire_label))
    if col_select == 'cand' : target_cols = cand_cols + test_cols
    if col_select == 'test' : target_cols = test_cols
    prjct_name = '{}-k{}_{}_{}'.format(*knn_set.values(),col_select)
    
    return prjct_name, knn_set, work_setting, target_cols


#main 함수 만드는 형식으로 코드 리팩토링 하기
# - dict 구조를 좀 더 효율적으로 정리해야 코드도 좀 더 정리될 것 같음
#   - dict_df 와 dict_train_test_split을 통합하거나 : train, val, target 으로 구분
#   - 저장 용량 효율화를 위해 data 전체가 아닌 idx, col 만을 저장하는 것도 방법 
# - 나중에 model 성능 테스트 떄 비교 용이를 위해서 knn_model 저장하는 오류도 해결하면 좋을 듯
#   - knn의 특성을 고려했을 떄 작업자/환경/돌아간 시간 도 저장하는 것이 나중에 발표에 도움될 듯
#전처리 과정에서 geo_strat별로 city_label 부여하고 metric_btwn_city 계산하는 것 넣기 : city label 안봐도 예측 가능하게끔
#저장한 dict 읽어올 때 좀 더 효율적이게 : cond_na, test_df에 대한 정보도 주는 쪽이 좋긴할텐데
#그치만 어차피 복잡한 상황은 아니라 통일되고 있다고 전제해도 될 듯


def make_data_dict(test_df, target_sample,n_strata=10,strata='quantile'):
    dict_data = dict()
    for col in target_sample:
        temp = test_df[info_cols+[col]]
        cond_na = temp.isna().any(axis=1)
        X_spprt, y_spprt = temp.loc[~cond_na,info_cols], temp.loc[~cond_na,col]
        train_X, valid_X, train_y, valid_y= train_test_split_strat_y(X_spprt, y_spprt,
                                                                     n_strata=n_strata,
                                                                     method=strata,
                                                         test_size = 0.2,
                                                         random_state=801,
                                                         ) 
        dict_data[col]={
        'support' : {'X':X_spprt, 'y':y_spprt},
        'train' : {'X':train_X, 'y':train_y},
        'valid' : {'X':valid_X, 'y':valid_y},
        'target' : {'X':temp.loc[cond_na,info_cols], 'y_cond':cond_na}
        }
    return dict_data


@record_time
def knn_reg_process(model:KNeighborsRegressor,dict_data:dict):
    knn_col = copy.deepcopy(model)
    spprt_data = dict_data['support']
    train_data = dict_data['train']
    valid_data = dict_data['valid']
    target_data = dict_data['target']['X']

    knn_col.fit(**train_data)
    y_pred_vlid = knn_col.predict(valid_data['X'])    
    knn_col.fit(**spprt_data)
    y_pred_trgt = knn_col.predict(target_data)
    
    return {'valid' : y_pred_vlid,
            'target': y_pred_trgt}
    
def save_knn_intermid(save_dir,work_name,dict_data,dict_rslt):
    ## save intermd pkl
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    file_name = 'dict_data{}.pkl'.format(work_name)
    save_pkl(save_dir,file_name,dict_data)
    file_name = 'dict_rslt_{}.pkl'.format(work_name)
    save_pkl(save_dir,file_name,dict_rslt)

    print("pkl saved")

def fill_reg_rslt(save_dir,work_name,test_df,dict_data,dict_rslt):
    ## fill missing values
    rslt_form = test_df[info_cols+target_sample]

    for col in target_sample:
        cond = dict_data[col]['target']['y_cond']
        rslt_form.loc[cond,col] = dict_rslt[col]['target']

    file_name = 'pvtb_filled_knn_{}.csv'.format(work_name)
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    rslt_form.to_csv(os.path.join(save_dir,file_name))

def make_city_metric(geo_info:pd.DataFrame):
    city_idx_list = list(geo_info.index)
    dict_rslt= {
    }
    for X,Y in itertools.product(city_idx_list,repeat=2):
        dict_rslt[(X,Y)] = metric_btwn_city_info(geo_info.loc[X],geo_info.loc[Y])
    return dict_rslt

if __name__ == '__main__':
    # initial setting
    prjct_name, knn_set, work_setting, target_cols = prjct_config()
    start_idx, n_work, step = work_setting
    
    geo_name = 'geo_strat_encoded.csv'
    geo_info_path = os.path.join(RSLT_DIR,geo_name)
    geo_strat_info = pd.read_csv(geo_info_path)

    pvtb_name = 'pvtb_city_encoded_ver2.csv'
    pvtb_encoded = pd.read_csv(os.path.join(PVTB_DIR,pvtb_name))
    entire_label = list(pvtb_encoded.columns)[10:]

    weight_norm_cand = [np.array([0.5,0.5,0.4,0.1]),
                        np.array([0.55,0.5,0.35,0.05])]
    weight_func_cand = [lambda x : np.exp2(-(np.power(10*x,2)))/(np.power(x,3)+0.00000001),
                        lambda x : 1/(np.power(x,7)+0.00000001),]
    
    
    metric,n_neigh,weight_func = knn_set.values()

    if 'custom' in metric:
        idx = int(metric.split('-')[-1])
        weight_norm = weight_norm_cand[idx]
        metric_dict_city= make_city_metric(geo_strat_info.drop(columns='count'))
        metric = lambda x,y : weigted_metric_city(x,y,metric_dict_city,weight_norm)
    if 'custom' in weight_func:
        idx = int(weight_func.split('-')[-1])
        weight_func = weight_func_cand[idx]

    # knn regression for each work
    for work_idx in tqdm(range(start_idx,n_work,step)):
        target_sample = target_cols[work_idx::n_work]
        work_name = '{}_{}_{}'.format(prjct_name,n_work,work_idx) 
        print(f'work : {work_name}')

        test_df = pvtb_encoded[info_cols+target_cols]
        dict_data = make_data_dict(test_df,target_sample,n_strata = 5, strata='value')
        model = KNeighborsRegressor(n_neighbors=n_neigh,weights=weight_func,metric=metric[0],algorithm='auto')

        dict_rslt = dict()
        for col in target_sample:
#            ipdb.set_trace()
            rslt,reg_time = knn_reg_process(model,dict_data[col])
            dict_rslt[col] = {**rslt,'time' : reg_time}
            print (col,'/ support_n : {}/ target_n : {}/ time : {:.5f} (sec)'.format(
                len(dict_data[col]['support']['X']),len(dict_data[col]['target']['X']),reg_time)
            )
            
        prjct_dir = os.path.join(RSLT_DIR,f'knn/knn_{prjct_name}')
        if not os.path.exists(prjct_dir): os.mkdir(prjct_dir)
        save_knn_intermid(os.path.join(prjct_dir,'pkl'),
                          work_name,dict_data,dict_rslt)
        ## check score
        dict_score = make_reg_score_dict_cols(dict_data,dict_rslt,print_rslt=True)
        plot_n_save_regrslt(os.path.join(prjct_dir,'PLOT'),
                            work_name,dict_data,dict_rslt,dict_score)
        fill_reg_rslt(os.path.join(prjct_dir,'PROCESSED'),
                      work_name,test_df,dict_data,dict_rslt)
        print("{} : process completed".format(work_name))


#label에는 interpolated 된 value가 들어가지 않도록 주의

'''
########################################################################

# load, score intermd pkl and finish to csv

entire_label = 
target_cols = 

knn_dir = 'knn_else'
knn_path = os.path.join(RSLT_DIR,knn_dir)
work_idx = 0
n_work = 5 

for work_idx in tqdm(range(1,5)):

    work_name = '{}_{}'.format(work_idx, n_work)
    target_sample = target_cols[work_idx::n_work]
    work_name = '{}_{}'.format(work_idx,n_work) 
    print(f'work : {work_name}')

    pvtb_encoded['city_idx'] = pvtb_encoded['geo_label_city'].apply(lambda x : city_list.index(x))
    temp = list(pvtb_encoded.columns)
    pvtb_encoded_whole = pvtb_encoded[['city_idx']+temp[1:-1]]
    test_df = pvtb_encoded[info_cols+target_cols]


    files = list(filter(lambda x : work_name in x,os.listdir(knn_path)))
    file_dict = dict()
    for filename in files:
        if filename[-4:] != '.pkl' : continue
        if 'knn' in filename : continue
        print(filename)
        file_path = os.path.join(knn_path,filename)
        with open (file_path,'rb') as f:
            file_dict[filename[:-5-len(work_name)]] = pickle.load(f)

    dict_rslt = file_dict['dict_rslt']
    target_sample= list(dict_rslt.keys())

    rslt_form = test_df[info_cols+target_sample]

    for col in target_sample:
        cond = test_df[col].isna()
        loaded = dict_rslt[col]['target']
        assert np.sum(cond) == len(loaded)  
        rslt_form.loc[cond,col] = loaded

    file_name = 'pvtb_filled_knn_{}_{}.csv'.format(work_idx,10)
    save_dir = os.path.join(RSLT_DIR,knn_dir,'PROCESSED')
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    rslt_form.to_csv(os.path.join(save_dir,file_name))

    ### check score for loaded data

    dict_rslt=file_dict['dict_rslt']
    dict_train_test=file_dict['dict_train_test']

    dict_df = dict()
    for col in target_sample:
        temp = test_df[info_cols+[col]]
        cond_na = temp.isna().any(axis=1)
        dict_df[col] = {
            'train' : [temp.loc[~cond_na,info_cols], temp.loc[~cond_na,col]],
            'target' : [temp.loc[cond_na,info_cols], cond_na],
        }

    ## check score
    dict_score = make_reg_score_dict_cols(target_sample,dict_df,dict_train_test,dict_rslt,print_plot=True)

    save_dir = os.path.join(RSLT_DIR,knn_dir,'PLOT')
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    fig,axes = scatter_reg_rslt(dict_train_test,dict_rslt)
    file_name = 'reg_scatter_{}.png'.format(work_name)
    fig.savefig(os.path.join(save_dir,file_name))

    fig,axes = plot_reg_score(dict_train_test,dict_rslt,dict_score)
    file_name = 'reg_rslt_{}.png'.format(work_name)
    fig.savefig(os.path.join(save_dir,file_name))

    print("process done")

''' 