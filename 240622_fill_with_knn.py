import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import missingno as msno
import seaborn as sns 
import os
import functools, itertools
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle

## matplotlib setting - py에서 쓸 때는 어디에 써야 함?

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
RAW_DATA_DIR = 'america_big_cities_health_inventory'
RAW_FILE_NAME = 'BigCitiesHealth.csv'
RSLT_DIR = '/home/doeun/code/AI/ESTSOFT2024/workspace/1.project1_structured/BCHI/processed/'
PVTB_DIR = RSLT_DIR + 'pvtb/'

## FUNCTIONS - FILE I/O

import pickle

def save_pkl(save_dir,file_name,save_object):
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    file_path = os.path.join(save_dir,file_name)
    with open(file_path,'wb') as f:
        pickle.dump(save_object,f)

## FUNCTIONS - DF PROCESS

def cond_check_dict(data=pd.DataFrame,val_dict=dict):
    cond_list=[
        data[col] == val
        for col, val in val_dict.items()
    ]
    return functools.reduce(lambda x,y: x & y, cond_list)

## FUNCTIONS - PLOTTING

def choose_split_point(word_len,space,ths):
    # 윗 줄에 space 만큼 공백이 있고, 한 줄의 길이가 ths로 제한 되어있을 때
    # 어떤 지점에서 단어를 끊어줄지 정하기
    # |-------ths-------|
    # |-space-|---------|-space-|------| : word
    #         |-------ths-------|
    print(word_len,space,ths)
    if word_len < ths + space :
        if abs(word_len/2 -ths) <= abs(word_len/2-space) :
            return word_len-ths
        else :
            return word_len - space if word_len < 2 * space else space
    else :
        return ths if word_len - (ths + space) < 0.3 * ths else space

def minimize_seq_idx_np(domain:np.array,func):
    vfunc = np.vectorize(func)
    temp = np.argsort(vfunc(domain))
    return temp[0]

def modify_strlen_ths(last,new,ths=16):
    front = len(last)
    space = ths - (1+front)
    if len(new) < space :
        rslt = [last + ' ' + new]
    else :
        if len(new) < ths:
            rslt = [last, new]
        else:
            cut = choose_split_point(len(new),space-1,ths-1)
            new_h, new_e = new[:cut]+'-', new[cut:]
            if cut < ths-1 :
                rslt = modify_strlen_ths(last+' '+new_h,new_e)
            else :
                rslt = [last] + modify_strlen_ths(new_h,new_e) 
    return rslt

def str_cutter(sentnc, ths = 16):
    words= sentnc.split(' ')
    rslt, pnt = [''], 0
    while pnt < len(words):
        last = '' if len(rslt)==0 else rslt[-1]
        next_ele = modify_strlen_ths(last,words[pnt],ths)
        rslt = rslt[:-1] + next_ele
        pnt += 1
    return '\n'.join(rslt)[1:]

def choose_plot_grid(n:int,r_max=8,c_max=17,res_ths=2):
    #ver2
    r_min = np.ceil(n/c_max)
    sppt = np.arange(r_min,r_max+1) #need error process
    col_nums = np.ceil(n/sppt)
    res = col_nums * sppt -n
    min_idx = np.where((res==np.min(res)) | (res <= res_ths))[0]
    row_cand, col_cand = sppt[min_idx], col_nums[min_idx]
    if len(min_idx) > 1 :
        res = np.abs(row_cand-col_cand)
        i = np.where(res==np.min(res))[0][0]
    else : i = 0
    return int(row_cand[i]), int(col_cand[i])

def pair_plot_feat_hue(fig,axes,data:dict,pair_plot,axis_share=False,hue_label_dict=None, **kwargs):
    #ver2
    if (fig is None) or (axes is None) :
        num_r, num_c = choose_plot_grid(len(data))
        fig, axes = plt.subplots(num_r,num_c,figsize=(21,17),sharex=axis_share,sharey=axis_share)
    for n,key in enumerate(data.keys()):
        ax = axes.flatten()[n]
        plt.setp(ax.get_xticklabels(),ha = 'left',rotation = 90)
        if n >= len(data) : continue
        pair_plot(x=data[key][0], y = data[key][1],ax =ax, **kwargs)
        feat_name = str(key) 
        if hue_label_dict: color = 'b' if hue_label_dict[feat_name] else 'k'
        else : color = 'k'
        ax.set_xlabel(str_cutter(feat_name,20),loc='left',fontsize = 8.3,color=color)
    return fig,axes

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

## FUNCTIONS - DATA- SCORE REG RSLT

def make_reg_score_dict(y_actual,y_pred,base_val):
    rmse_model, rmse_base = np.sqrt(mse(y_actual,y_pred)), np.sqrt(mse([base_val]*len(y_actual),y_actual))
    msle_model, msle_base = 0, 0 #msle(valid_y,y_pred) : negtive value error occurs but i don't know why #msle(valid_y,[train_y.mean()]*len(valid_y)) :
    mape_model, mape_base = mape(y_actual,y_pred), mape([base_val]*len(y_actual),y_actual)
    r2_model, r2_base = r2_score(y_actual,y_pred), 0
    
    return {
        'rmse' : [rmse_model, rmse_base],
        'msle' : [msle_model, msle_base],
        'mape' : [mape_model, mape_base],
        'r2_score' : [r2_model,r2_base]
    }

def print_reg_score_dict(name,dict_score):
    print('{}\nr2 score : {:.5f}'.format(name,dict_score['r2_score'][0]))
    print('rmse_model : {:.5f} / rmse_base : {:.5f}\t'.format(*dict_score['rmse']),
          'mape_model : {:.5f} / mape_base : {:.5f}\t'.format(*dict_score['mape']),
          'msle_model : {:.5f} / msle_base : {:.5f}'.format(*dict_score['msle']))

def make_reg_score_dict_cols(target_sample,dict_df,dict_train_test,dict_rslt,print_plot=False):
    dict_score = dict()
    for col in target_sample:
        train_y = dict_df[col]['train'][1]
        valid_y = dict_train_test[col][3]
        y_pred = dict_rslt[col]['valid']
        dict_score[col] = make_reg_score_dict(valid_y,y_pred,np.mean(train_y))
        if print_plot :
            print_reg_score_dict(col,dict_score[col])
            print('-'*150)
    return dict_score

## FUNCTIONS - DATA- PLOT REG RSLT
from itertools import repeat, chain

def scatter_reg_rslt(dict_train_test,dict_rslt): #set_iput
    target_sample = list(dict_rslt.keys())
    data_plot ={
        col : (dict_train_test[col][3], dict_rslt[col]['valid'])
        for col in target_sample 
    }
    data_line = {
        col : (dict_train_test[col][3],dict_train_test[col][3])
        for col in target_sample 
    }
    fig,axes = plt.subplots(3,3,figsize=(12,12))
    fig,axes = pair_plot_feat_hue(fig=fig,axes=axes,data=data_line,
    #fig,axes = pair_plot_feat_hue(fig=None,axes=None,data=data_line,
                                  pair_plot=sns.lineplot,lw=0.3)
    #fig.set_size_inches(12,8, forward=True)
    fig,axes = pair_plot_feat_hue(fig=fig,axes=axes,data=data_plot,
                                  pair_plot=sns.scatterplot,s=5,alpha=0.65)

    for n,key in enumerate(data_plot.keys()):
        ax = axes.flatten()[n]
        ax.set_ylabel('')
        ax.set_title(key,fontsize=12)
        ax.set_xlabel('rmse_model : {:.3f} | rmse_base : {:.3f}\nr2 score : {:.3f} {:>35}\n'.format(*dict_score[key]['rmse'],dict_score[key]['r2_score'][0],
            f'n = {len(dict_train_test[key][0])}'), fontsize=10, ha ='left')

    for ax in axes.flatten():
        plt.setp(ax.get_yticklabels(),rotation = 0, fontsize = 9)
        plt.setp(ax.get_xticklabels(),ha ='center',rotation = 0, fontsize = 9)
    
    return fig,axes

def plot_reg_score(dict_train_test,dict_rslt,dict_score):
    data_plot ={
        col : (dict_train_test[col][3], dict_rslt[col]['valid'])
        for col in target_sample 
    }
    fig,axes = plt.subplots(len(data_plot),3,figsize=(15,20))
    for n, col in enumerate(data_plot.keys()):
        #quo, rem = divmod(n,3)
        ax1, ax2, ax3 = axes[n][0], axes[n][1], axes[n][2]
        test_y, y_pred = data_plot[col]
        train_y = dict_train_test[col][2]

        sns.histplot(test_y,label='actual',ax=ax1,alpha=0.5)
        sns.histplot(y_pred,label='knn_pred',ax=ax1,alpha=0.5)
        ax1.legend(fontsize=9)

        sns.histplot(test_y-np.mean(train_y),ax=ax2, label = 'baseline',alpha=0.5)
        sns.histplot(test_y-y_pred,ax=ax2, label = 'knn_pred',alpha=0.5)
        ax2.legend(fontsize=9)

        df_score = pd.DataFrame(dict_score[col]).T[[1,0]]
        xs = list(chain.from_iterable(repeat(val,2) for val in df_score.index))
        ax3r = ax3.twinx()
        sns.barplot(x=xs[:4],y=list(df_score.values.reshape(-1))[:4],
                    hue = ['knn_pred','base']*2,ax=ax3,alpha=0.65,legend=False)
        sns.barplot(x=xs[4:],y=list(df_score.values.reshape(-1))[4:],
                    hue = ['knn_pred','base']*2,ax=ax3r,alpha=0.8,legend=False)
        ax3.set_yscale('log')
        ax3r.set_ylim([0.0,1.15])
        ax3r.bar_label(ax3r.containers[0], fontsize=8, fmt='%.4f')
        ax3r.bar_label(ax3r.containers[1], fontsize=8, fmt='%.4f')
        ax3.grid(False)
        ax3r.grid(False)
        ax3r.set_yscale('linear')    
        ax3.bar_label(ax3.containers[0], fontsize=8, fmt='%.4f')
        ax3.bar_label(ax3.containers[1], fontsize=8, fmt='%.4f')
        #ax3.axvline()

        ax1.set_title(str_cutter(col,50),fontsize=15,loc='left',ha='left')
        ax2.xaxis.set_label_coords(-0.02, -0.15)
        ax2.set_xlabel('rmse_model : {:.3f} | rmse_base : {:.3f}\nr2 score : {:.3f} {:>48}\n'.format(*dict_score[col]['rmse'],dict_score[col]['r2_score'][0],
            f'n = {len(dict_train_test[col][0])}'), fontsize=10,ha ='left')
        #ax2.xaxis.set_label_position('left')
        ax1.set_xlabel('')
        ax3.set_xlabel('')
        ax1.set_ylabel('count',fontsize =10) 
        ax2.set_ylabel('count',fontsize =10) 
        plt.setp(ax3.get_yticklabels(),rotation = 0, fontsize = 9, color='#333333')
        plt.setp(ax3r.get_yticklabels(),rotation = 0, fontsize = 9)

    for ax in axes.flatten():
        plt.setp(ax.get_yticklabels(),rotation = 0, fontsize = 9)
        plt.setp(ax.get_xticklabels(),ha ='center',rotation = 0, fontsize = 9)

    return fig, axes

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

vec_metric_dict={
    key : np.vectorize(val)
    for key,val in encoded_metric_dict.items()
}

def metric_btwn_city_info_for_vec(X,Y):
    diff = [
        vec_metric_dict['geo_strata_region'](int(X[0]),int(Y[0])),
        vec_metric_dict['geo_strata_poverty'](int(X[1]),int(Y[1])),
        vec_metric_dict['geo_strata_Population'](int(X[2]),int(Y[2])),
        vec_metric_dict['geo_strata_PopDensity'](int(X[3]),int(Y[3])),
        vec_metric_dict['geo_strata_Segregation'](int(X[4]),int(Y[4])),
    ]
    return np.linalg.norm(np.array(diff),ord=7)

weight_norm = np.array([0.5,0.5,0.4,0.1])

def weigted_metric_spprt(vec_metric,X,Y):
    city_idx= range(2,7)
    race_idx, sex_idx, date_idx = 0, 1, 7
    diff = [
        vec_metric['strata_race_label'](int(X[race_idx]),int(Y[race_idx])),
        vec_metric['strata_sex_label'](int(X[sex_idx]),int(Y[sex_idx])),
        vec_metric['city_info'](X[city_idx],Y[city_idx]),
        np.abs(X[date_idx]-Y[date_idx]),
            ]
    return np.linalg.norm(np.array(diff)*weight_norm,ord=5)

dict_metric = {
    'strata_race_label': metric_on_adj(entire_race,race_adjacent),
    'strata_sex_label': metric_on_adj(encode_sex,sex_adjacent),
    'geo_strata_region' : metric_on_adj(encode_region,region_adjacent),
    'geo_strata_poverty' : metric_binary,
    'geo_strata_Population' : metric_binary,
    'geo_strata_PopDensity' : metric_binary,
    'geo_strata_Segregation' : metric_binary,
}

geo_name = 'geo_strat_info.csv'
geo_info_path = os.path.join(RSLT_DIR,geo_name)
geo_strat_info = pd.read_csv(geo_info_path, index_col=0)

pvtb_name = 'pvtb_city_encoded_ver1.csv'
pvtb_encoded = pd.read_csv(os.path.join(PVTB_DIR,pvtb_name),index_col=0)

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

metric_val = make_metric_val(geo_strat_info,geo_strat_cols)
dict_metric['geo_label_city'] = lambda x,y : metric_val[(x,y)]

weight_norm = np.array([0.5,0.5,0.4,0.1])

city_list = list(geo_strat_info.index)

def weigted_metric_city(X,Y,weight_norm):
    X_city, Y_city = city_list[int(X[0])], city_list[int(Y[0])]
    race_idx, sex_idx, date_idx = 1, 2, 8
    diff = [
        vec_metric_dict['strata_race_label'](int(X[race_idx]),int(Y[race_idx])),
        vec_metric_dict['strata_sex_label'](int(X[sex_idx]),int(Y[sex_idx])),
        dict_metric['geo_label_city'](X_city,Y_city),
        np.abs(X[date_idx]-Y[date_idx]),
            ]
    return np.linalg.norm(np.array(diff)*weight_norm,ord=5)

test_col = [
    'Deaths | Premature Death',                                  
    'Deaths | Injury Deaths',                                    
    'Cancer | All Cancer Deaths',                                
    'Cardiovascular Disease | Cardiovascular Disease Deaths',    
    'Deaths | Deaths from All Causes',                           
    'Substance Use | Drug Overdose Deaths',                      
    'Cardiovascular Disease | Heart Disease Deaths',             
    'Diabetes and Obesity | Diabetes Deaths',                    
    'Cancer | Lung Cancer Deaths',                               
    'Life Expectancy at Birth | Life Expectancy',                
    'Mental Health | Suicide',                                   
    'Deaths | Motor Vehicle Deaths',                             
    'Deaths | Gun Deaths (Firearms)',                            
    'Respiratory Infection | Pneumonia or Influenza Deaths',     
    'Substance Use | Opioid Overdose Deaths',                    
    'Cancer | Colorectal Cancer Deaths',                         
    'Crime Incidents | Homicides',                               
]

col_cand_list=[
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Deaths | Premature Death',
    'Income Inequality | Household Income Inequality',
    'Income | Households with Higher-Incomes',
    'Life Expectancy at Birth | Life Expectancy',
    'Substance Use | Adult Smoking',
    'Births | Preterm Births',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Cardiovascular Disease | Heart Disease Deaths',
    'Deaths | Infant Deaths',
    'Deaths | Premature Death',
    'Income Inequality | Household Income Inequality',
    'Income | Households with Higher-Incomes',
    'Income | Per-capita Household Income',
    'Language and Nativity | Primarily Speak English',
    'Life Expectancy at Birth | Life Expectancy',
    'Substance Use | Adult Smoking',
    'Cancer | All Cancer Deaths',
    'Cancer | Breast Cancer Deaths',
    'Cancer | Lung Cancer Deaths',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Cardiovascular Disease | Heart Disease Deaths',
    'Language and Nativity | Primarily Speak Spanish',
    'Births | Teen Births',
    'Cancer | All Cancer Deaths',
    'Cancer | Breast Cancer Deaths',
    'Cancer | Lung Cancer Deaths',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Cardiovascular Disease | Heart Disease Deaths',
    'Heat and Disasters | Community Social Vulnerability to Climate Disasters',
    'Income-related | Unemployment',
    'Language and Nativity | Primarily Speak Spanish',
    'Mental Health | Suicide',
    'Race/Ethnicity | Minority Population',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Deaths | Premature Death',
    'Income Inequality | Household Income Inequality',
    'Income | Households with Higher-Incomes',
    'Life Expectancy at Birth | Life Expectancy',
    'Substance Use | Adult Smoking',
    'Births | Preterm Births',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Cardiovascular Disease | Heart Disease Deaths',
    'Deaths | Infant Deaths',
    'Deaths | Premature Death',
    'Income Inequality | Household Income Inequality',
    'Income | Households with Higher-Incomes',
    'Income | Per-capita Household Income',
    'Language and Nativity | Primarily Speak English',
    'Life Expectancy at Birth | Life Expectancy',
    'Substance Use | Adult Smoking',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Deaths | Premature Death',
    'Income Inequality | Household Income Inequality',
    'Income | Households with Higher-Incomes',
    'Life Expectancy at Birth | Life Expectancy',
    'Substance Use | Adult Smoking',
    'Births | Preterm Births',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Cardiovascular Disease | Heart Disease Deaths',
    'Deaths | Infant Deaths',
    'Deaths | Premature Death',
    'Income Inequality | Household Income Inequality',
    'Income | Households with Higher-Incomes',
    'Income | Per-capita Household Income',
    'Language and Nativity | Primarily Speak English',
    'Life Expectancy at Birth | Life Expectancy',
    'Substance Use | Adult Smoking',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Deaths | Premature Death',
    'Income Inequality | Household Income Inequality',
    'Income | Households with Higher-Incomes',
    'Life Expectancy at Birth | Life Expectancy',
    'Substance Use | Adult Smoking',
    'Births | Preterm Births',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Cardiovascular Disease | Heart Disease Deaths',
    'Deaths | Infant Deaths',
    'Deaths | Premature Death',
    'Income Inequality | Household Income Inequality',
    'Income | Households with Higher-Incomes',
    'Income | Per-capita Household Income',
    'Language and Nativity | Primarily Speak English',
    'Life Expectancy at Birth | Life Expectancy',
    'Substance Use | Adult Smoking',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Cardiovascular Disease | Heart Disease Deaths',
    'Deaths | Motor Vehicle Deaths',
    'Deaths | Premature Death',
    'Diabetes and Obesity | Diabetes Deaths',
    'Life Expectancy at Birth | Life Expectancy',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Cardiovascular Disease | Heart Disease Deaths',
    'Deaths | Gun Deaths (Firearms)',
    'Deaths | Motor Vehicle Deaths',
    'Deaths | Premature Death',
    'Diabetes and Obesity | Diabetes Deaths',
    'Language and Nativity | Foreign Born Population',
    'Life Expectancy at Birth | Life Expectancy',
    'Mental Health | Adult Mental Distress',
    'Racial Segregation Indices | Racial Segregation, White and Non-White',
    'Respiratory Infection | COVID-19 Deaths',
    'Respiratory Infection | New Tuberculosis Cases',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Cardiovascular Disease | Heart Disease Deaths',
    'Deaths | Motor Vehicle Deaths',
    'Deaths | Premature Death',
    'Diabetes and Obesity | Diabetes Deaths',
    'Life Expectancy at Birth | Life Expectancy',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Cardiovascular Disease | Heart Disease Deaths',
    'Deaths | Gun Deaths (Firearms)',
    'Deaths | Motor Vehicle Deaths',
    'Deaths | Premature Death',
    'Diabetes and Obesity | Diabetes Deaths',
    'Language and Nativity | Foreign Born Population',
    'Life Expectancy at Birth | Life Expectancy',
    'Mental Health | Adult Mental Distress',
    'Racial Segregation Indices | Racial Segregation, White and Non-White',
    'Respiratory Infection | COVID-19 Deaths',
    'Respiratory Infection | New Tuberculosis Cases',
    'Deaths | Gun Deaths (Firearms)',
    'Deaths | Premature Death',
    'Life Expectancy at Birth | Life Expectancy',
    'Mental Health | Suicide',
    'Substance Use | Drug Overdose Deaths',
    'Substance Use | Opioid Overdose Deaths',
    'Crime Incidents | Homicides',
    'Deaths | Deaths from All Causes',
    'Deaths | Gun Deaths (Firearms)',
    'Deaths | Motor Vehicle Deaths',
    'Deaths | Premature Death',
    'Food Access | Limited Supermarket Access',
    'Language and Nativity | Foreign Born Population',
    'Life Expectancy at Birth | Life Expectancy',
    'Mental Health | Suicide',
    'Sexually Transmitted Disease | New Gonorrhea Cases',
    'Substance Use | Drug Overdose Deaths',
    'Substance Use | Opioid Overdose Deaths',
    'Active Transportation | Walking to Work',
    'Deaths | Deaths from All Causes',
    'Deaths | Premature Death',
    'Education | Preschool Enrollment',
    'Lead Poisoning | Child Lead Levels 5+ mcg/dL',
    'Transportation | Public Transportation Use',
    'Active Transportation | Walking to Work',
    'Deaths | Deaths from All Causes',
    'Deaths | Gun Deaths (Firearms)',
    'Deaths | Infant Deaths',
    'Deaths | Premature Death',
    'Education | Preschool Enrollment',
    'Heat and Disasters | Longer Summers',
    'Income-related | Service Workers',
    'Language and Nativity | Primarily Speak Chinese',
    'Lead Poisoning | Child Lead Levels 5+ mcg/dL',
    'Mental Health | Suicide',
    'Transportation | Public Transportation Use',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Crime Incidents | Homicides',
    'Deaths | Deaths from All Causes',
    'Deaths | Gun Deaths (Firearms)',
    'Deaths | Injury Deaths',
    'Life Expectancy at Birth | Life Expectancy',
    'Births | Low Birthweight',
    'Births | Preterm Births',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Cardiovascular Disease | Heart Disease Deaths',
    'Crime Incidents | Homicides',
    'Crime Incidents | Violent Crime',
    'Deaths | Deaths from All Causes',
    'Deaths | Gun Deaths (Firearms)',
    'Deaths | Injury Deaths',
    'Deaths | Maternal Deaths',
    'Life Expectancy at Birth | Life Expectancy',
    'Mental Health | Adult Mental Distress',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Deaths | Deaths from All Causes',
    'Education | College Graduates',
    'Income Inequality | Income Inequality',
    'Lead Poisoning | Housing Lead Risk',
    'Racial Segregation Indices | Racial Segregation, White and Black',
    'Cardiovascular Disease | Cardiovascular Disease Deaths',
    'Deaths | Deaths from All Causes',
    'Deaths | Racial Disparity in Police Killings',
    'Disability | People with Disabilities',
    'Education | College Graduates',
    'Income Inequality | Income Inequality',
    'Income-related | Service Workers',
    'Lead Poisoning | Housing Lead Risk',
    'Population | Single-Parent Families',
    'Race/Ethnicity | Minority Population',
    'Racial Segregation Indices | Racial Segregation, White and Black',
    'Racial Segregation Indices | Racial Segregation, White and Non-White',
    'Active Transportation | Riding Bike to Work',
    'Education | Preschool Enrollment',
    'Housing | Owner Occupied Housing',
    'Population | Population Density',
    'Race/Ethnicity | Minority Population',
    'Racial Segregation Indices | Racial Segregation, White and Black',
    'Active Transportation | Riding Bike to Work',
    'Education | Preschool Enrollment',
    'Housing | Homeless, Total',
    'Housing | Owner Occupied Housing',
    'Housing | Renters vs. Owners',
    'Income Inequality | Income Inequality',
    'Population | Population Density',
    'Race/Ethnicity | Minority Population',
    'Racial Segregation Indices | Racial Segregation, White and Black',
    'Racial Segregation Indices | Racial Segregation, White and Hispanic',
    'Transportation | Lack of Car',
    'Transportation | Longer Driving Commute Time',
    'Deaths | Injury Deaths',
    'Health Insurance | Uninsured, All Ages',
    'Housing | Homeless, Total',
    'Mental Health | Adult Mental Distress',
    'Sexually Transmitted Disease | Syphilis Prevalence',
    'Sexually Transmitted Disease | Syphilis, Newborns',
    'Deaths | Injury Deaths',
    'Health Insurance | Uninsured, All Ages',
    'Health Insurance | Uninsured, Child',
    'Housing | Homeless, Children',
    'Housing | Homeless, Total',
    'Housing | Vacant Housing and Homelessness',
    'Lead Poisoning | Child Lead Levels 5+ mcg/dL',
    'Mental Health | Adult Mental Distress',
    'Population | Children',
    'Sexually Transmitted Disease | Syphilis Prevalence',
    'Sexually Transmitted Disease | Syphilis, Newborns',
]

########################################################################

# def auto_filename : for dict
# transform checkking and making dir to decorator

########################################################################
from sklearn.neighbors import KNeighborsRegressor
import copy, time
from tqdm import tqdm


cand_cols = sorted(list(set(col_cand_list)))
entire_label = list(pvtb_encoded.columns)[9:]
print(len(entire_label))
target_cols = list(filter(lambda x : x not in cand_cols,entire_label))
prjct_name = 'else'

n_work = 10

for work_idx in tqdm(range(1,n_work,2)):

    target_sample = target_cols[work_idx::n_work]
    work_name = '{}_{}_{}'.format(prjct_name,work_idx,n_work) 
    print(f'work : {work_name}')

    pvtb_encoded['city_idx'] = pvtb_encoded['geo_label_city'].apply(lambda x : city_list.index(x))
    temp = list(pvtb_encoded.columns)
    pvtb_encoded_whole = pvtb_encoded[['city_idx']+temp[1:-1]]
    test_df = pvtb_encoded[info_cols+target_cols]

    #main 함수 만드는 형식으로 코드 리팩토링 하기
    # - dict 구조를 좀 더 효율적으로 정리해야 코드도 좀 더 정리될 것 같음
    #   - dict_df 와 dict_train_test_split을 통합하거나 : train, val, target 으로 구분
    #   - 저장 용량 효율화를 위해 data 전체가 아닌 idx, col 만을 저장하는 것도 방법 
    # - 나중에 model 성능 테스트 떄 비교 용이를 위해서 knn_model 저장하는 오류도 해결하면 좋을 듯
    #   - knn의 특성을 고려했을 떄 작업자/환경/돌아간 시간 도 저장하는 것이 나중에 발표에 도움될 듯
    #전처리 과정에서 geo_strat별로 city_label 부여하고 metric_btwn_city 계산하는 것 넣기 : city label 안봐도 예측 가능하게끔
    #저장한 dict 읽어올 때 좀 더 효율적이게 : cond_na, test_df에 대한 정보도 주는 쪽이 좋긴할텐데
    #그치만 어차피 복잡한 상황은 아니라 통일되고 있다고 전제해도 될 듯
    #param 추가 
    
    dict_df = dict()
    for col in target_cols:
        temp = test_df[info_cols+[col]]
        cond_na = temp.isna().any(axis=1)
        dict_df[col] = {
            'train' : [temp.loc[~cond_na,info_cols], temp.loc[~cond_na,col]],
            'target' : [temp.loc[cond_na,info_cols], cond_na],
        }

    #train/validation
    dict_train_test = {
        col: train_test_split(*dict_df[col]['train'],
                             test_size = 0.2,
                             random_state=801) #check how to using stratify option
        for col in target_cols}

    knn_metric_weighted = lambda x,y : weigted_metric_city(x,y,weight_norm)
    model = KNeighborsRegressor(n_neighbors=7,weights='distance',metric=knn_metric_weighted,algorithm='auto')

    dict_knn, dict_rslt = dict(), dict()
    for col in target_sample:
        knn_col = copy.deepcopy(model)
        train_X, train_y = dict_df[col]['train']
        _,valid_X,_,valid_y = dict_train_test[col]
        target_X, cond_na = dict_df[col]['target']

        start = time.time()
        knn_col.fit(train_X,train_y)
        y_pred_trgt = knn_col.predict(target_X)
        pred_start = time.time()
        y_pred_vlid = knn_col.predict(valid_X)
        end = time.time()

        print (col,f'/ train_n : {len(train_X)}/ target_n : {len(target_X)}/ time : {end-start:.5f} (sec)')
        dict_knn[col], dict_rslt[col] = knn_col, {'target':y_pred_trgt,
                                                  'valid':y_pred_vlid,
                                                  'pred_time' : end-pred_start}

    ## save intermd pkl
    knn_dir = f'knn_{prjct_name}'
    save_dir = os.path.join(RSLT_DIR,knn_dir)

    file_name = 'dict_train_test_{}.pkl'.format(work_name)
    save_pkl(save_dir,file_name,dict_train_test)
    file_name = 'dict_rslt_{}.pkl'.format(work_name)
    save_pkl(save_dir,file_name,dict_rslt)
    #file_name = 'dict_knn_{}.pkl'.format(work_name)
    #save_pkl(save_dir,file_name,dict_knn)
    #_pickle.PicklingError: Can't pickle <function <lambda> at 0x7f08f804d8b0>: attribute lookup <lambda> on __main__ failed

    print("pkl saved")

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

    print("plot completed")

    ## fill missing values
    rslt_form = test_df[info_cols+target_sample]

    for col in target_sample:
        cond = dict_df[col]['target'][1]
        rslt_form.loc[cond,col] = dict_rslt[col]['target']

    file_name = 'pvtb_filled_knn_{}.csv'.format(work_name)
    save_dir = os.path.join(RSLT_DIR,knn_dir,'PROCESSED')
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    rslt_form.to_csv(os.path.join(save_dir,file_name))

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