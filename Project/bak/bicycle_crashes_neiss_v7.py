#!/usr/bin/env python
# coding: utf-8

# # Bicycle Crash analysis NEISS data 1999 to 2018
# ## National Electronic Injury Surveillance System
# 
# "CPSC’s National Electronic Injury Surveillance System (NEISS) is a national probability sample of hospitals in the U.S. and its territories. Patient information is collected from each NEISS hospital for every emergency visit involving an injury associated with consumer products."
# 
# https://catalog.data.gov/dataset/cpscs-national-electronic-injury-surveillance-system-neiss
# https://www.cpsc.gov/cgibin/NEISSQuery/home.aspx
# 
# 

# Imports

import calendar
from datetime import datetime
import ipywidgets as widgets
from ipywidgets import interact, fixed
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import re
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import seaborn as sns
import sys
import xlrd


# In[2]:


#get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')


# - Selection of geographic areas called primary sampling units (PSU) that are defined within sampling strata. 
# 
# https://www.cdc.gov/nchs/nhis/singleton_psu.htm

# In[ ]:





# In[3]:


class Neiss:
    DICT_CATEGORY_TRANSLATOR = {}

    def __init__(self, df):
        self.df = df
        #self.df['Age'] = self.translateAge(self.df['Age'])

    def setColumnCodeDictionary(column_dictionary):
        Neiss.DICT_CATEGORY_TRANSLATOR = column_dictionary
        
    def getDataFrame(self):
        return self.df

    def getColumnDictionary(column_name):
        return Neiss.DICT_CATEGORY_TRANSLATOR[column_name]

    def extractRowsWhereColumnContainsString(self, column_name, search_values, case=False, translate_codes=False):
        '''
        Search the column specified by column_name for the strings listed in search_values.
        Args:
            column_name (str): The name of the column to search for the strings
            search_values (list): A list containing the strings to search for
            
        Return:
            A dataframe containing the rows where the search was successful
        '''
        # https://kanoki.org/2019/03/27/pandas-select-rows-by-condition-and-string-operations/
        
        
        df = self.df[self.df[column_name].str.contains('|'.join(search_values), case=case)]
        if True == translate_codes:
            df = Neiss.translateCodes(df)
        return df
    
    def extractRowsWhereColumnsContainCode(self, column_names, search_values, translate_codes=False):
        '''
        Search the column specified by column_name for the strings listed in search_values.
        Args:
            column_names (list(str)): Alist containing the names of columns to search for the strings
            search_values (list): A list containing the codes to search for
            
        Return:
            A dataframe containing the rows where the search was successful
        '''
        df = pd.DataFrame()
        for column_name in column_names:
            df_filter = self.df[column_name].isin(search_values)
            if df.empty:
                df = self.df[df_filter]
            else:
                df.append(self.df[df_filter])

        if True == translate_codes:
            df = Neiss.translateCodes(df)
        return df

    def lookupCodeFor(category, lookup_name):
        inverse = {v: k for k, v in Neiss.DICT_CATEGORY_TRANSLATOR[category].items()}
        return inverse[lookup_name]
    
    def translateAge(self, df):
        return [age if age < 200 else (age - 200)/12 for age in df['Age']]

    def translateCodes(df):
        return df.replace( Neiss.DICT_CATEGORY_TRANSLATOR)


# ## Read the pickled neissYYYY.xlsx file
# 
# xlsx_to_pckl.ipynb is used to create the pickled file
# - If the neiss_data.pckl file exists read it as the data file.
# - Otherwise, raise an exception.



neiss_pathname = os.getcwd() + '/data/NEISS'

pckl_fname = neiss_pathname + '/neiss_data.pckl'
if os.path.exists(pckl_fname):
    print("Reading {}  ... ".format(pckl_fname), end="")
    neiss_df = pickle.load( open( pckl_fname, "rb" ) )
    print("done!")
else:
    raise Exception('ERROR:  {} does not exist'.format(pckl_fname))


# In[5]:


neiss_df.shape


# In[6]:


neiss_df.head()


# In[7]:


def getColumnCodeDictionary(fname):
    d1 = {}
    with xlrd.open_workbook(fname) as workbook:
        sheet_names = workbook.sheet_names()

        for sheet in sheet_names:
            xl_sheet = workbook.sheet_by_name(sheet)

            d2 = {}
            for row_idx in range(0, xl_sheet.nrows):
                code = int(xl_sheet.cell_value(row_idx, 0))
                value = xl_sheet.cell_value(row_idx, 1)
                d2[code] = value
            d1[sheet] = d2
    return d1

column_codes_fname = neiss_pathname + '/column_codes.xlsx'
column_dictionary = getColumnCodeDictionary(column_codes_fname)
Neiss.setColumnCodeDictionary(column_dictionary)


# - Code to take the Neiss dictionaries for column codes and write them out to the column_codes.xlsx file.

# In[8]:


neiss = Neiss(neiss_df)


# ## Generates a graph showing injuries per year

# In[9]:


def showInjuriesPerYear(df, date_name, stat_name, sup_title):
    '''
    Show the accumulated number of injuries per year
    
    Args:
        date_name (str): The name of the column containing the datetime instances
        stat_name (str): The name of the column to include in the graph (one line per unique item will be displayed)
        sup_title (str): The super position title to display
    
    Returns:
        None
    '''
    stat_dict = Neiss.getColumnDictionary(stat_name)

    df1 = df.xs([stat_name, date_name], axis='columns')
    df2 = df1.copy()
    df2['_count'] = 1
    df2['_year'] = [t.year for t in df2[date_name]]

    df3 = pd.pivot_table(
        df2, values='_count', index=['_year'], columns=[stat_name],
        aggfunc=np.sum, fill_value=0)

    fig, ax = plt.subplots(figsize=(18,6))
    fig.suptitle(sup_title)
    df3.columns=[stat_dict[key] for key in sorted(stat_dict.keys(), reverse=False)]
    df3.plot(ax=ax, lw=2)
    ax.set_ylabel('Injuries')
    ax.set_xlabel('Year')


# ## Generates a graph showing injuries per week (1 ... 52) of the year

# In[10]:


def showInjuriesPerWeekOfTheYear(df, date_name, stat_name, sup_title):
    '''
    Show the accumulated number of injuries week within the year
    
    Args:
        date_name (str): The name of the column containing the datetime instances
        stat_name (str): The name of the column to include in the graph (one line per unique item will be displayed)
        sup_title (str): The super position title to display
    
    Returns:
        None
    '''
    stat_dict = Neiss.getColumnDictionary(stat_name)

    df1 = df.xs([stat_name, date_name], axis='columns')
    df2 = df1.copy()
    df2['_count'] = 1
    df2['_week'] = [t.week for t in df2[date_name]]

    df3 = pd.pivot_table(
        df2, values='_count', index=['_week'], columns=[stat_name],
        aggfunc=np.sum, fill_value=0)

    fig, ax = plt.subplots(figsize=(18,6))
    fig.suptitle(sup_title)
    df3.columns=[stat_dict[key] for key in sorted(stat_dict.keys(), reverse=False)]
    df3.plot(ax=ax, lw=2)
    ax.set_ylabel('Injuries')
    ax.set_xlabel('Week Of The Year')
    ax.set_xticks(range(1,52))


# ## Generates a graph of aggregated injuries per month

# In[11]:


def showInjuriesPerMonthOfTheYear(df, date_name, stat_name, sup_title):
    '''
    Show the accumulated number per month
    
    Args:
        date_name (str): The name of the column containing the datetime instances
        stat_name (str): The name of the column to include in the graph (one line per unique item will be displayed)
        sup_title (str): The super position title to display
    
    Returns:
        None
    '''
    stat_dict = Neiss.getColumnDictionary(stat_name)

    df1 = df.xs([stat_name, date_name], axis='columns')
    df2 = df1.copy()
    df2['_count'] = 1
    df2['_month'] = [t.month for t in df2[date_name]]
    
    
    df3 = pd.pivot_table(
        df2, values='_count', index=['_month'], columns=[stat_name],
        aggfunc=np.sum, fill_value=0)
    df3.index = calendar.month_name[1:]

    fig, ax = plt.subplots(figsize=(18,6))
    fig.suptitle(sup_title)
    df3.columns=[stat_dict[key] for key in sorted(stat_dict.keys(), reverse=False)]
    df3.plot(ax=ax, lw=2)
    ax.set_ylabel('Injuries')
    ax.set_xlabel('Month')


# ## Generates a graph a graph of aggregated injuries per weekday

# In[12]:


def showInjuriesPerWeekOfTheYear(df, date_name, stat_name, sup_title):
    '''
    Show the accumulated number per weekday
    
    Args:
        date_name (str): The name of the column containing the datetime instances
        stat_name (str): The name of the column to include in the graph (one line per unique item will be displayed)
        sup_title (str): The super position title to display
    
    Returns:
        None
    '''
    stat_dict = Neiss.getColumnDictionary(stat_name)

    df1 = df.xs([stat_name, date_name], axis='columns')
    df2 = df1.copy()
    df2['_count'] = 1
    df2['_week_day'] = [t.weekday() for t in df2[date_name]]

    df3 = pd.pivot_table(
        df2, values='_count', index=['_week_day'], columns=[stat_name],
        aggfunc=np.sum, fill_value=0)

    #df3.index = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
    df3.index = calendar.day_name
    
    fig, ax = plt.subplots(figsize=(18,6))
    fig.suptitle(sup_title)
    df3.columns=[stat_dict[key] for key in sorted(stat_dict.keys(), reverse=False)]
    df3.plot(ax=ax, lw=2)
    ax.set_ylabel('Injuries')
    ax.set_xlabel('Weekday')


# In[13]:


df = neiss.getDataFrame()
date_name, stat_name = 'Treatment_Date', 'Sex'

showInjuriesPerYear(df, date_name, stat_name, 'Injuries Per Year')
showInjuriesPerWeekOfTheYear(df, date_name, stat_name, 'Injuries Per 52 Weeks of the Year')
showInjuriesPerMonthOfTheYear(df, date_name, stat_name, 'Injuries Per Month of the Year')
showInjuriesPerWeekOfTheYear(df, date_name, stat_name, 'Injuries Per Week of the Year')


# https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
# 

# In[14]:


import scipy.stats as scs

class PearsonChiSquared:
    def __init__(self, df):
        self.df = df

    def getColumns(self):
        return self.df.columns
    
    def chiSquared(df, column_name1, column_name2):
        dependent = False
        table = df.xs([column_name1, column_name2], axis='columns')
        #print(table)

        #stat, p, dof, expected = chi2_contingency(table, lambda_="log-likelihood")
        stat, p, dof, expected = chi2_contingency(table)

        # interpret test-statistic
        prob = 0.95
        critical = chi2.ppf(prob, dof)
        if abs(stat) >= critical:
            dependent = True
        else:
            dependent = False

        # interpret p-value
        if p <= 1 - prob:
            dependent = True
        else:
            dependent = False

        #print('abs(stat) >= critical: {} p <= prob: {} p:{}'.format(abs(stat) >= critical, p <= prob, p))
        return dependent, 1 - p

    def getCorrMatrixDataframe(self, categorical):
        len_categorical = len(categorical)
        dfDst = pd.DataFrame(
            [[0] * len_categorical] * len_categorical,
            index = categorical,
            columns = categorical)

        dfSrc = self.df.copy()
        dfSrc = dfSrc + 1  # Make sure there are no zero entries for correlation
        #print('len(dfSrc)=', len(dfSrc))

        dependent, prob = False, 0
        for row_name in categorical:
            for col_name in categorical:
                if row_name == col_name:
                    prob = 1.0
                else:
                    dependent, prob = PearsonChiSquared.chiSquared(dfSrc, row_name, col_name)
                    if False == dependent:
                        prob = 0

                dfDst.loc[row_name][col_name] = prob
        return dfDst
    


# In[16]:


# contingency table
df = neiss_df.copy()
categorical = ['Sex', 'Race', 'Body_Part', 'Diagnosis', 'Disposition', 'Location',
               'Fire_Involvement', 'Product_1', 'Product_2', 'PSU']   # 'Stratum', 
df = df.xs(categorical, axis=1)
df.dropna(inplace=True)
df['Race'] = [int(x) for x in df['Race']]
df = df[0:2000]

pcs1 = PearsonChiSquared(df)
df1 = pcs1.getCorrMatrixDataframe(categorical)
#print(df1)

# Mask Upper triangle
# np.triu_indices_from(arr):  Return the indices for the upper-triangle of arr.
mask = np.zeros_like(df1)
mask[np.triu_indices_from(mask)] = True
mask

fig, ax = plt.subplots(figsize=(10, 10))

#sns.heatmap(df1, annot=True, mask=mask, fmt="d", linewidths=0.4, ax=ax)
sns.heatmap(df1, annot=True, linewidths=0.4, ax=ax)

# Fix the top and bottom margins of the heatmap
bottom_y, top_y = plt.ylim() 
bottom_y += 0.5 
top_y -= 0.5 
plt.ylim(bottom_y, top_y)

plt.show() 


# In[28]:


sns.pairplot(df)


# In[72]:


def correlations(dfCorrelations, xSel, ySel, plotType):
    '''
    Show the relational (scatter) plot of two columns
    
    Args:
        dfCorrelations (pd.DataFrame): The dataframe to show the relational plot on
        xSel (str): The name of the first selected column in the dataframe
        ySel (str): The name of the second selected column in the dataframe
    '''
    if xSel == ySel:
        print("Identity {}={}".format(xSel, ySel))
    else:
        dfSelCorr = dfCorrelations.xs([xSel, ySel], axis='columns')
        #showHeatmap([xSel, ySel], df2)
        print(dfSelCorr)

        try:
            xdict = Neiss.getColumnDictionary(xSel)
            xticks = np.sort(dfSelCorr[xSel].unique())
            xlabels = [xdict[x] if x in xdict else x for x in xticks]
        except Exception as e:
            print(e)
            
        try:
            ydict = Neiss.getColumnDictionary(ySel)
            yticks = np.sort(dfSelCorr[ySel].unique())
            ylabels = [ydict[y] if y in ydict else y for y in yticks]
        except Exception as e:
            print(e)

        print()
        print('xsel =', xSel)
        print('xticks = {} len={}'.format(xticks, len(xticks)))
        print('xlabels =', xlabels)

        print()
        print('ysel =', ySel)
        print('yticks = {} len={}'.format(yticks, len(yticks)))
        print('ylabels =', ylabels)

        fig, ax = plt.subplots(figsize=(len(xticks) + 1, len(yticks) + 1))

        if 'swarm' == plotType:
            sns.catplot(x=xSel, y=ySel, data=dfSelCorr, ax=ax, kind='swarm');
            plt.setp(ax, xticks=xticks, xticklabels=xlabels, yticks=yticks, yticklabels=ylabels)
        else: # default to scatter if unknown
            sns.scatterplot(x=xSel, y=ySel, data=dfSelCorr, ax=ax);
            plt.setp(ax, xticks=xticks, xticklabels=xlabels, yticks=yticks, yticklabels=ylabels)
            
        plt.close(2) # close the second empty graph


# In[73]:


from IPython.display import display
button = widgets.Button(description="Click Me!")
output = widgets.Output()

wDropDownSel1 = widgets.Dropdown(
    options=categorical,
    value=categorical[0],
    description='xCategory',
    disabled=False,
)

wDropDownSel2 = widgets.Dropdown(
    options=categorical,
    value=categorical[1],
    description='yCategory',
    disabled=False,
)

wDropDownSel3 = widgets.Dropdown(
    options=['swarm', 'scatter'],
    value='swarm',
    description='plot_type',
    disabled=False,
)

wHBox = widgets.HBox([wDropDownSel1, wDropDownSel2, wDropDownSel3])
wVBox = widgets.VBox([wHBox, button, output])

display(wVBox)

def on_button_clicked(b):
    with output:
        correlations(df, wDropDownSel1.value, wDropDownSel2.value, wDropDownSel3.value)

button.on_click(on_button_clicked)

