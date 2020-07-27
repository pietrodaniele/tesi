
import lightgbm as lgb
import numpy as np
import pandas as pd
import itertools
import os
import seaborn as sns
from dataclasses import asdict
from sklearn.model_selection import train_test_split as split

from load_data import RecoTrue, PARTICLES, prefix, get_df, all_reco_true
from definitions import PARTICLE_ORIGIN, PARTICLE_TYPE, CONVERSION_TYPE

print('Loading data...')
# loading file with and without pile up
pileup = True

all_df = {reco_true: get_df(reco_true, pileup) for reco_true in all_reco_true}
all_df_no_pileup = {reco_true: all_df[reco_true].query('%sis_pileup==False' % prefix[reco_true.reco]) for reco_true in all_reco_true}
# merging files. reco_el_truth_el, reco_ph_truth_el in unico file. Lo stesso per truth_ph
# true el
p = all_df_no_pileup[RecoTrue('photon', 'photon')].query('ph_is_ambiguous==True') # query verifica che i dati soddisfino una certa caratteristica
e = all_df_no_pileup[RecoTrue('electron', 'photon')].query('el_is_ambiguous==True')
df_true_photon = pd.merge(p, e, suffixes=('_ph', '_el'),
                          left_on=['EventNumber', 'ambiguos_index'],
                          right_on=['EventNumber', 'index'],
                          how='outer'
                         )
df_true_photon = df_true_photon.set_index(['EventNumber', 'index_ph'])
# true ph
p = all_df[RecoTrue('photon', 'electron')].query('ph_is_ambiguous==True')
e = all_df[RecoTrue('electron', 'electron')].query('el_is_ambiguous==True')
df_true_electron = pd.merge(e, p, suffixes=('_ph', '_el'),
                          right_on=['EventNumber', 'ambiguos_index'],
                          left_on=['EventNumber', 'index'])
df_true_electron = df_true_electron.set_index(['EventNumber', 'index_el'])
# merging two truth file
#data = pd.concat ([ df_true_photon , df_true_electron ] , names=['sample'], sort = False )
data = pd.concat({'el': df_true_electron.reset_index(), 'ph': df_true_photon.reset_index()}, names=['sample'], sort = False)
data = data.reset_index().drop('level_1', axis=1)
print('Data loaded')
# scelgo le caratteristiche che mi servono
data = data[[
                # particle truth
                'sample' ,
                # Electron Features
                'el_track_ep' , 'el_tracketa' , 'el_trackpt' , 'el_trackz0' ,
                'el_trkPixelHits' , 'el_trkSiHits' , 'el_refittedTrack_qoverp' ,
                # Photon Features
                'ph_convtrk1nPixHits' , 'ph_convtrk1nSCTHits' ,
                'ph_convtrk2nPixHits' , 'ph_convtrk2nSCTHits' , 'ph_zconv' ,
                'ph_Rconv' , 'ph_pt1conv' , 'ph_pt2conv' , 'ph_ptconv'
]]




# LightGBM Dataset Preparation
x_train , x_test , y_train , y_test = split (
                                                data.drop(columns=['sample']) ,
                                                data['sample'] ,
                                                test_size = 0.2 ,
                                            )

# create dataset for lightgbm
lgb_train = lgb.Dataset(x_train, y_train)
lgb_test = lgb.Dataset(x_test, y_test, reference = lgb_train)


# specify your configurations as a dict
params = {

}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_test,
                early_stopping_rounds=5)

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)'''
