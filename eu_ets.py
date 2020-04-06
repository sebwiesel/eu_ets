#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 07:04:36 2020

@author: sebastian
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

st.title('EU Emissions Trading System')

DATA_URL = r'https://raw.githubusercontent.com/sebwiesel/eu_ets/master/data_03.csv'

st.markdown(
'''
The EU emissions trading system (EU ETS) is a cornerstone of the EU's policy
to combat climate change and its key tool for reducing greenhouse gas emissions
cost-effectively. It is the world's first major carbon market and remains the
biggest one. This Streamlit app hosted on Heroku shows all installations that
fall within the scope of the EU ETS. Use the slider to pick a specific year
and look at how the chart changes. You can also switch between 4 different data
points and scale the size of the data bars. The height of a data bar indicates
the absolute amount of the selected data point. The colour or a data bars
indicates the percentage change between the start of the data and the selected
year. Red indicates an increase green a decrease.

Follow the link and take a look at the source code.
[See source code](https://github.com/sebwiesel/eu_ets/blob/master/eu_ets.py)
''')

@st.cache(allow_output_mutation=True)
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows, sep='|', error_bad_lines=False)
    # Streamlit is a lowercase world so ...
    #lowercase = lambda x: str(x).lower()
    #data.rename(lowercase, axis='columns', inplace=True)
    #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


#https://bsou.io/posts/color-gradients-with-python
def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)



def make_colors(df, value, center):

    df['RANK'] = df.groupby(['YEAR'])[value].rank(method='first')
    # Creates bins for ...
    # High is bad
    df.loc[df[value] >= center, 'BIN'] = df[df[value] >= center].groupby(['YEAR'])['RANK'].transform(
        lambda x: pd.qcut(x, 5, labels=range(5,10)))
    # Low is good
    df.loc[df[value] < center, 'BIN'] = df[df[value] < center].groupby(['YEAR'])['RANK'].transform(
        lambda x: pd.qcut(x, 5, labels=range(0,5)))
    # Start color is good
    gradient_obj = linear_gradient('#26c929','#cc0000', n=10)

    # Create map dictionaries. 
    r_dict = dict(enumerate(gradient_obj['r']))
    g_dict = dict(enumerate(gradient_obj['g']))    
    b_dict = dict(enumerate(gradient_obj['b']))
    
    df['R'] = df['BIN'].map(r_dict)
    df['G'] = df['BIN'].map(g_dict)
    df['B'] = df['BIN'].map(b_dict)
    


# This text element lets the user know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(150000)
# Notify the user that the data was successfully loaded.

data_load_state.text('Loading data...done!')
#data = data[['LON','LAT', 'VERIFIED_EMISSIONS', 'INSTALLATION_NAME', 'YEAR']]

make_colors(data, value='VERIFIED_EMISSIONS_PCT_CHANGE', center=0)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

metric = st.sidebar.selectbox(label='Data Point',
                              options=['VERIFIED_EMISSIONS',
                                       'ALLOCATION',
                                       'ALLOCATION_RESERVE',
                                       'ALLOCATION_TRANSITIONAL'],
                              index=0)

year = st.sidebar.slider(label='Year',
                         min_value=2008,
                         max_value=2018,
                         value=2018,
                         step=1)

scale = st.sidebar.slider(label='Scale',
                         min_value=1,
                         max_value=100,
                         value=50,
                         step=1)

lower_percentile, upper_percentile = st.sidebar.slider(label='Percentile',
                                                       min_value=0,
                                                       max_value=100,
                                                       value=(0,100),
                                                       step=1)

col_tooltip = {
    'html': '''
            ACCOUNT HOLDER: {ACCOUNT_HOLDER_NAME}<br>
            INSTALLATION: {INSTALLATION_NAME}<br>
            VERIFIED EMISSIONS: {VERIFIED_EMISSIONS} tC02e<br>
            Change since 2008: {VERIFIED_EMISSIONS_PCT_CHANGE}<br>
            ALLOCATION: {ALLOCATION}<br>
            Change since 2008: {ALLOCATION_PCT_CHANGE}<br>            
            ALLOCATION RESERVE: {ALLOCATION_RESERVE}<br>
            Change since 2008: {ALLOCATION_RESERVE_PCT_CHANGE}<br>
            ALLOCATION TRANSITIONAL: {ALLOCATION_TRANSITIONAL}<br>
            Change since 2008: {ALLOCATION_TRANSITIONAL_PCT_CHANGE}<br>
            ACTIVITY TYPE: {ACTIVITY_TYPE}<br>
            COUNTRY: {COUNTRY}<br>            
            ''',
    'style': {
        'background': 'grey',
        'color': 'white',
        'font-family': '"Helvetica Neue", Arial',
        'z-index': '10000',
    },
}

# -----------------------------------------------------------------------------
# Set filters. 
# -----------------------------------------------------------------------------

data_year = data[data['YEAR']==year]

mask_perc = (
            (data_year[metric] >=
             np.percentile(data_year[metric], lower_percentile)) &
            (data_year[metric] <=
             np.percentile(data_year[metric], upper_percentile)))

data_perc = data_year[mask_perc]

filtered_data = data_perc
filtered_data.sort_values(metric, ascending=False, inplace=True)


st.subheader('Map of all Installations')

midpoint = (np.average(filtered_data['LAT']),
            np.average(filtered_data['LON']))


col_lyer = pdk.Layer(
    'ColumnLayer',
    data = filtered_data,
    get_position = ['LON', 'LAT'],
    get_elevation = '{0}'.format(metric),
    elevation_scale = 1 / scale,
    radius = 5000,
    get_fill_color = ['R', 'G', 'B', 255],
    pickable = True,
    auto_highlight = True,
    opacity = .5
    )

st.write(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state={
        'latitude': midpoint[0],
        'longitude': midpoint[1],
        'min_zoom': 2,
        'max_zoom': 10,
        'zoom': 3,
        'pitch': 40.5,
        'bearing': -27.36
    },
    layers = [col_lyer],
    tooltip = col_tooltip
))

if st.checkbox('Show Raw Data'):
    st.subheader('Raw Data for Selection')
    st.write(filtered_data)
