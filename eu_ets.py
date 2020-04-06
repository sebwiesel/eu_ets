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

#DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

st.markdown(
"""
This is a demo of a Streamlit app that shows the Uber pickups
geographical distribution in New York City. Use the slider
to pick a specific hour and look at how the charts change.
[See source code](https://github.com/streamlit/demo-uber-nyc-pickups/blob/master/app.py)
""")

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

st.sidebar.title('Sidebar Title')

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
                         value=1,
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
            ALLOCATION: {ALLOCATION}<br>
            ALLOCATION RESERVE: {ALLOCATION_RESERVE}<br>
            ALLOCATION TRANSITIONAL: {ALLOCATION_TRANSITIONAL}<br>
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



# # If the user doesn't want to select which features to control, these will be used.
# default_control_features = ['Young','Smiling','Male']
# if st.sidebar.checkbox('Show advanced options'):
#     # Randomly initialize feature values. 
#     features = get_random_features(feature_names, seed)
#     # Let the user pick which features to control with sliders.
#     control_features = st.sidebar.multiselect( 'Control which features?',
#         sorted(features), default_control_features)
# else:
#     features = get_random_features(feature_names, seed)
#     # Don't let the user pick feature values to control.
#     control_features = default_control_features

# # Insert user-controlled values from sliders into the feature vector.
# for feature in control_features:
#     features[feature] = st.sidebar.slider(feature, 0, 100, 50, 5)

# # Generate a new image from this feature vector (or retrieve it from the cache).
# with session.as_default():
#     image_out = generate_image(session, pg_gan_model, tl_gan_model,
#             features, feature_names)


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

# legend = """
#                 <style>
#                 .bdot {{
#                 height: 15px;
#                 width: 15px;
#                 background-color: Blue;
#                 border-radius: 50%;
#                 display: inline-block;
#                 }}
#                 .gdot {{
#                 height: 15px;
#                 width: 15px;
#                 background-color: #4DFF00;
#                 border-radius: 50%;
#                 display: inline-block;
#                 }}
#                 </style>
#                 </head>
#                 <body>
#                 <div style="text-align:left">
#                 <h3>Legend</h3>
#                 <span class="bdot"></span>  {} - {}<br>
#                 <span class="gdot"></span>  &#62;{} - {}
#                 </div>
#                 </body>
#                 """.format(round(min_val), round((max_val - min_val) / 2), round((max_val - min_val) / 2), round(max_val))

# st.markdown(legend, unsafe_allow_html=True)


if st.checkbox('Show raw data', False):
    st.subheader('Raw data by ... year minute')
    st.write(filtered_data)
    
    
    
# # -----------------------------------------------------------------------------
# # Define Layers
# # -----------------------------------------------------------------------------
# hex_layer = pdk.Layer(
#     'HexagonLayer',
#     data=data,
#     get_position=['lon', 'lat'],
#     radius=100,
#     elevation_scale=4,
#     elevation_range=[0, 1000],
#     pickable=True,
#     extruded=True,
# )

# sct_layer = pdk.Layer(
#     'ScatterplotLayer',
#     data=data,
#     get_position=['lon', 'lat'],
#     auto_highlight=True,
#     get_radius=10000,          # Radius is given in meters
#     get_fill_color=[180, 0, 200, 140],  # Set an RGBA value for fill
#     pickable=True)