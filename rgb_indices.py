#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2022-02-10
Purpose: RGB indices extraction
"""

import argparse
import os
import sys
import rasterio
import numpy as np
import glob
# import matplotlib.pyplot as plt
import cv2
import tifffile as tifi
import pandas as pd
import geopandas as gpd
import multiprocessing
import re
from datetime import datetime
from numpy import inf
import warnings
warnings.filterwarnings('ignore')


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Rock the Casbah',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # parser.add_argument('positional',
    #                     metavar='str',
    #                     help='A positional argument')

    parser.add_argument('-i',
                        '--input_dir',
                        help='Input directory',
                        metavar='str',
                        type=str,
                        required=True)

    parser.add_argument('-d',
                        '--date',
                        help='Collection date.',
                        metavar='str',
                        type=str,
                        required=True)

    parser.add_argument('-od',
                        '--output_dir',
                        help='Output directory name.',
                        metavar='str',
                        type=str,
                        default='rgb_indices')

    parser.add_argument('-of',
                        '--output_filename',
                        help='Output file name.',
                        metavar='str',
                        type=str,
                        default='rgb_indices')

    return parser.parse_args()


# --------------------------------------------------
def get_paths(directory):

    ortho_list = []

    for root, dirs, files in os.walk(directory):
        for name in files:
            if '.tif' in name:
                ortho_list.append(os.path.join(root, name))

    if not ortho_list:

        raise Exception(f'ERROR: No compatible images found in {directory}.')


    print(f'Items to process: {len(ortho_list)}')

    return ortho_list


# --------------------------------------------------
def create_tgi(r_band, g_band, b_band):
    
    tgi = (g_band.astype(float)-(0.39*r_band.astype(float))-(0.61*b_band.astype(float)))
    # tgi[tgi < 0] == np.nan
    mean, median, q1, q3, var, sd = get_stats(tgi)

    return tgi, mean, median, q1, q3, var, sd


# --------------------------------------------------
def get_stats(img):
    
    # img = img[~np.isnan(img)]

    mean = np.mean(img) #- 273.15
    median = np.percentile(img, 50)

    q1 = np.percentile(img, 25)
    q3 = np.percentile(img, 75)

    var = np.var(img)
    sd = np.std(img)

    return mean, median, q1, q3, var, sd


# --------------------------------------------------
def find_date_string(dir_path):
    try:
        match = re.search(r'\d{4}-\d{2}-\d{2}', dir_path)
        date = datetime.strptime(match.group(), '%Y-%m-%d').date()

    except:
        print('Error: Cannot find scan/flight date. Make sure input directory has a date in the following format YYYY-MM-DD.')

    return date


# --------------------------------------------------
def open_image(img_path):

    img = tifi.imread(img_path)
    
    return img


# --------------------------------------------------
def split_bands(img):
    try:
        r, g, b, _ = cv2.split(img)
        # b, g, r, _ = cv2.split(img)
    except:
        r, g, b, = cv2.split(img)
        # b, g, r, = cv2.split(img)

    return r, g, b


# --------------------------------------------------
def blur_image(img):

    blur_img = cv2.blur(img, (40, 40))

    return blur_img
    

# --------------------------------------------------
def add_fieldbook_data(df, fb_df):
    
    fb_df.columns = fb_df.columns.str.lower()
    fb_df = fb_df.set_index('plot')#.drop(drop_list, axis=1)
    out_df = fb_df.join(df)
    
    return out_df


# --------------------------------------------------
def kmeans_img(img):

    # img = img[~np.isnan(img)]

    pixel_vals = img.reshape((-1,1))
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.0)
    k = 2
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 500, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((img.shape))    
    low_thresh = np.unique(segmented_image)[:1][0]
    upp_thresh = np.unique(segmented_image)[1:2][0]
    # img[segmented_image > upp_thresh] = np.nan

    # mean, median, q1, q3, var, sd = get_stats(img)

    return segmented_image, low_thresh, upp_thresh #mean, median, q1, q3, var, sd


# --------------------------------------------------
def get_detection_df(date, cyverse_public_link='https://data.cyverse.org/dav-anon/iplant/projects/phytooracle/season_11_sorghum_yr_2020/level_3/stereoTop/season_11_clustering.csv'):
    detection_csv = pd.read_csv(cyverse_public_link).convert_dtypes()
    detection_csv['plot'] = detection_csv['plot'].astype(str).str.zfill(4)
    detection_csv = detection_csv.set_index('plot')
    detection_csv = detection_csv[detection_csv['date']==str(date)]

    return detection_csv


# --------------------------------------------------
def get_plot_number(plot_path):
    
    plot = os.path.basename(plot_path).split('_')[0]
    
    return plot


# --------------------------------------------------
def remove_infinite(arr):

    arr[~np.isfinite(arr)] = np.nan

    return arr


# --------------------------------------------------
def do_SNDVI(r,g,b):
    try:
        sndvi = (g.astype(float) - r.astype(float)) / ((g.astype(float)+r.astype(float))-3)
        sndvi = remove_infinite(sndvi)
    except:
        sndvi = np.nan
    # sndvi = replace_inf(sndvi)
    return  np.nanmean(sndvi)


# --------------------------------------------------
def do_vdvi(r,g,b):
    try:
        vdvi = ((2-g.astype(float))-r.astype(float)-b.astype(float))/((2-g.astype(float))+r.astype(float)+b.astype(float))-4
        vdvi = remove_infinite(vdvi)
    except:
        vdvi = np.nan
    # vdvi = replace_inf(vdvi)
    return  np.nanmean(vdvi)


# --------------------------------------------------
def do_ari1b(r,g,b):
    try:
        ari1b = (1/g.astype(float)) - (1/r.astype(float))
        ari1b = remove_infinite(ari1b)
    except:
        ari1b = np.nan
    # ari1b = replace_inf(ari1b)
    return  np.nanmean(ari1b)


# --------------------------------------------------
def do_bi(r,g,b):
    try:
        bi = np.sqrt(((r.astype(float)**2) + (g.astype(float)**2))/2)
        bi = remove_infinite(bi)
    except:
        bi = np.nan
    # bi = replace_inf(bi)
    return  np.nanmean(bi)


# --------------------------------------------------
def do_ci(r,g,b):
    try:
        ci = (r.astype(float)-g.astype(float))/(r.astype(float)+g.astype(float))
        ci = remove_infinite(ci)
    except: 
        ci = np.nan
    # ci = replace_inf(ci)
    return  np.nanmean(ci)


# --------------------------------------------------
def do_rgri(r,g,b):
    try:
        rgri = (r.astype(float)/g.astype(float))
        rgri = remove_infinite(rgri)
    except:
        rgri = np.nan
    # rgri = replace_inf(rgri)
    return  np.nanmean(rgri)


# --------------------------------------------------
def do_exr(r,g,b):
    try:
        exr = 2*g.astype(float)*r.astype(float)-b.astype(float)
        exr = remove_infinite(exr)
    except:
        exr = np.nan
    # exr = replace_inf(exr)
    return  np.nanmean(exr)


# # --------------------------------------------------
# def do_exr(r,g,b):
#     try:
#         exr = 2*g.astype(float)*r.astype(float)-b.astype(float)
#     except:
#         exr = np.nan
#     # exr = replace_inf(exr)
#     return  np.nanmean(exr)

# --------------------------------------------------
def do_exgr(r,g,b):
    try:
        exg = do_EXG(r,g,b)
        exr = do_exr(r,g,b)
        exgr = exg - exr
        # exgr = remove_infinite(exgr)

    except:
        exgr = np.nan
    return  exgr


# --------------------------------------------------
def do_si(r,g,b):
    try:
        si = (r.astype(float)+g.astype(float)+b.astype(float))/3
        si = remove_infinite(si)
    except:
        si = np.nan
    # si = replace_inf(si)
    return  np.nanmean(si)


# --------------------------------------------------
def replace_inf(x):
    x[x == -inf] = 0
    x[x == inf] = 0
    x[np.isnan(x)] = 0
    # print('unique and length before')
    # print(np.unique(x))
    # print(len(x))
    x = x[x!=0.0]

    # print('unique and length after')
    # print(np.unique(x))
    # print(len(x))


    return x


# --------------------------------------------------
def do_GR(r,g,b):
    try:
        rg = r.astype(float)/g.astype(float)
        rg = remove_infinite(rg)
    except:
        rg = np.nan
    # rg = replace_inf(rg)
    
    return  np.nanmean(rg)


# --------------------------------------------------
def do_GRVI(r,g,b):
    try:
        grvi = (g.astype(float)-r.astype(float))/(g.astype(float)+r.astype(float))
        grvi = remove_infinite(grvi)
    except:
        grvi = np.nan
    # grvi = replace_inf(grvi)
    return  np.nanmean(grvi)


# --------------------------------------------------
def do_RGBVI(r,g,b):
    try:
        rgbvi = (g.astype(float)**2)-(b.astype(float)*r.astype(float))/(g.astype(float)**2)+(b.astype(float)*r.astype(float))
        rgbvi = remove_infinite(rgbvi)
    except:
        rgbvi = np.nan
    # rgbvi = replace_inf(rgbvi)
    return  np.nanmean(rgbvi)


# --------------------------------------------------
def do_MGRVI(r,g,b):
    try:
        mgrvi = (g.astype(float)**2) - (r.astype(float)**2) / (g.astype(float)**2) + (r.astype(float)**2)
        mgrvi = remove_infinite(mgrvi)
    except:
        mgrvi = np.nan
    # mgrvi = replace_inf(mgrvi)
    return  np.nanmean(mgrvi)


# --------------------------------------------------
def do_VARI(r,g,b):
    try:
        vari = (g.astype(float)-r.astype(float))/(g.astype(float)+r.astype(float)-b.astype(float))
        vari = remove_infinite(vari)
    except:
        vari = np.nan
    # vari = replace_inf(vari)
    return  np.nanmean(vari)


# --------------------------------------------------
def do_BGI2(r,g,b):
    try:
        bgi2 = b.astype(float)/g.astype(float)
        bgi2 = remove_infinite(bgi2)
    except:
        bgi2 = np.nan
    # bgi2 = replace_inf(bgi2)
    return  np.nanmean(bgi2)


# --------------------------------------------------
def do_VEG(r,g,b):
    try:
        a = 0.667
        veg = g.astype(float)/(r.astype(float)^a)*(b.astype(float)^(1-a))
        veg = remove_infinite(veg)
    except:
        veg = np.nan
    # veg = replace_inf(veg)
    return  np.nanmean(veg)


# --------------------------------------------------
def do_GLI(r,g,b):
    try:
        gli = (2*g.astype(float) - r.astype(float) - b.astype(float))/(2*g.astype(float) + r.astype(float) + b.astype(float))
        gli = remove_infinite(gli)
    except:
        gli = np.nan
    # gli = replace_inf(gli)
    return  np.nanmean(gli)


# --------------------------------------------------
def do_EXG(r,g,b):
    try:
        exg = 2*g.astype(float) -r.astype(float) -b.astype(float)
        exg = remove_infinite(exg)
    except:
        exg = np.nan
    # exg = replace_inf(exg)
    return  np.nanmean(exg)


# --------------------------------------------------
def do_NGBDI(r,g,b):
    try:
        ngbdi = (g.astype(float)-b.astype(float))/(g.astype(float)+b.astype(float))
        ngbdi = remove_infinite(ngbdi)
    except:
        ngbdi = np.nan
    # ngbdi = replace_inf(ngbdi)
    return  np.nanmean(ngbdi)


# --------------------------------------------------
def do_RGBV12(r,g,b):
    try:
        rgbv12 = (g.astype(float)-r.astype(float))/b.astype(float)
        rgbv12 = remove_infinite(rgbv12)
    except:
        rgbv12 = np.nan
    # rgbv12 = replace_inf(rgbv12)
    return  np.nanmean(rgbv12)


# --------------------------------------------------
def do_RGBV13(r,g,b):
    try:
        rgbv13 = (g.astype(float)+b.astype(float))/r.astype(float)
        rgbv13 = remove_infinite(rgbv13)
    except:
        rgbv13 = np.nan
    # rgbv13 = replace_inf(rgbv13)
    return  np.nanmean(rgbv13)


# --------------------------------------------------
def do_TGI(r, g, b):
    try:
        tgi = (g.astype(float)-(0.39*r.astype(float))-(0.61*b.astype(float)))
#         tgi = remove_infinite(tgi)
    except:
        tgi = np.nan

    return np.nanmean(tgi)


# --------------------------------------------------
def clip_individual_plants(plot, img, detection_csv, date):
    indices_dict = {}

    for i, row in detection_csv.loc[plot].iterrows():

        min_x = int(row['min_x'])
        max_x = int(row['max_x'])
        min_y = int(row['min_y'])
        max_y = int(row['max_y'])
        plant_name = row['plant_name']

        if img.shape[2]:
            crop = img[min_y:max_y,min_x:max_x,:]

        else:
            crop = img[min_y:max_y,min_x:max_x]

        r, g, b = split_bands(crop)
        #tgi, mean, median, q1, q3, var, sd = create_tgi(r, g, b)

        indices_dict[plant_name] = {
            'date': date,
            'plot': plot,
            'plant_name': plant_name,
            'tgi': do_TGI(r, g, b)
#             'gr': do_GR(r,g,b), 
#             'grvi': do_GRVI(r,g,b),
#             'rgbvi': do_RGBVI(r,g,b),
#             'mgrvi': do_MGRVI(r,g,b),
#             'vari': do_VARI(r,g,b),
#             'bgi': do_BGI2(r,g,b),
#             'gli': do_GLI(r,g,b),
#             'exg': do_EXG(r,g,b),
#             'ngbdi': do_NGBDI(r,g,b),
#             'rgbv12': do_RGBV12(r,g,b),
#             'rgbv13': do_RGBV13(r,g,b),
#             'sndvi': do_SNDVI(r,g,b),
#             'vdvi': do_vdvi(r,g,b),
#             'ari1b': do_ari1b(r,g,b),
#             'bi': do_bi(r,g,b), 
#             'ci': do_ci(r,g,b),
#             'rgri': do_rgri(r,g,b),
#             'exr': do_exr(r,g,b),
#             'exgr': do_exgr(r,g,b),
#             'si': do_si(r,g,b)
        }
    df = pd.DataFrame.from_dict(indices_dict, orient='index')

    try:
        df = df.reset_index()
        df = df.drop('index', axis=1)
    except:
        df = df.reset_index()

    return df
    

# --------------------------------------------------
def merge_detection_phenotype(df, detection_csv):

    df['plot'] = df['plot'].astype(str)
    df['plant_name'] = df['plant_name'].astype(str)
    df['date'] = df['date'].astype(str)
    df = df.set_index(['date', 'plot', 'plant_name'])

    detection_csv = detection_csv.reset_index()
    detection_csv['plot'] = detection_csv['plot'].astype(str)
    detection_csv['plant_name'] = detection_csv['plant_name'].astype(str)
    detection_csv['date'] = detection_csv['date'].astype(str)
    detection_csv = detection_csv.reset_index().set_index(['date', 'plot', 'plant_name'])

    result_df = detection_csv.join(df)

    return result_df
    

# --------------------------------------------------
def clean_dataframe(final_df):

    final_df = final_df.reset_index().drop(['level_0', 'level_0 ', 'Unnamed: 0', 'index'], axis=1, errors='ignore')

    return final_df


# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    args = get_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Get date and list images to process
    plot_list = get_paths(args.input_dir)
    detection_csv = get_detection_df(date=args.date)
    result_list = []

    for plot_path in plot_list:

        try:
            plot = get_plot_number(plot_path)
            print(f'Processing {plot}.')
            img = open_image(plot_path)
            df = clip_individual_plants(plot, img, detection_csv, args.date)
            result_list.append(df)

            print('Processed successfully.')
        except:
            print(f'ERROR: Cannot process {plot}.')
            pass
            
    result_df = pd.concat(result_list)
    final_df = merge_detection_phenotype(result_df, detection_csv)
    final_clean_df = clean_dataframe(final_df)
    final_clean_df.to_csv(os.path.join(args.output_dir, '_'.join([args.date, args.output_filename])+'.csv'), index=False)


# --------------------------------------------------
if __name__ == '__main__':
    main()
