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

# In[1]:


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


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-whitegrid')


# - Selection of geographic areas called primary sampling units (PSU) that are defined within sampling strata. 
# 
# https://www.cdc.gov/nchs/nhis/singleton_psu.htm

# In[ ]:





# In[3]:


class UniqueId():
    '''
    A class for creating a column within a dataframe that represents a unique ID
    '''
    def addIdColumn(df, columnName):
        '''
        Create a new column in the dataframe with unique IDs for the values in columnName
        
        Args:
        df         (pd.DataFrame):  The dataframe containing the olumn
        columnName          (str):  The name of the column to create the unique IDs for
        '''
        unique = pd.Series(pd.unique(df[columnName].sort_values(ascending=True)))
        df[columnName + '_id'] = [unique[unique == code].index[0] for code in df[columnName]]

#dfX = pd.DataFrame([2, 1], columns = ['ohs'])
#UniqueId.addIdColumn(dfX, 'ohs')
#print(dfX)


# In[4]:


def getColumnCodeDictionary(fname):
    '''
    Read an Excel file containing the codes used and the appropriate code equivalences
    
    Args:
    fname   (str):  The excel file containing the translations from code to human readable equivalents 
    '''
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


# In[5]:


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
- If the neiss_data.pckl file exists read it as the data file.
- Otherwise, raise an exception.

See xlsx_to_pckl.ipynb for the creation of the file
# In[6]:


neiss_pathname = os.getcwd() + '/data/NEISS'

pckl_fname = neiss_pathname + '/neiss_data.pckl'
if os.path.exists(pckl_fname):
    print("Reading {}  ... ".format(pckl_fname), end="")
    neiss_df = pickle.load( open( pckl_fname, "rb" ) )
    print("done!")
else:
    raise Exception('ERROR:  {} does not exist'.format(pckl_fname))


# In[7]:


neiss_df.shape


# In[8]:


neiss_df.head()


# In[ ]:





# In[ ]:





# - Read the column code dictionary from the Excel file and add it to the Neiss object

# In[9]:


column_codes_fname = neiss_pathname + '/column_codes.xlsx'
column_dictionary = getColumnCodeDictionary(column_codes_fname)
Neiss.setColumnCodeDictionary(column_dictionary)


# - Code to take the Neiss dictionaries for column codes and write them out to the column_codes.xlsx file.

# In[10]:


neiss = Neiss(neiss_df)


# In[11]:


neiss_df.shape


# ## Generates a graph showing injuries per year

# In[12]:


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

# In[13]:


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

# In[14]:


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

# In[15]:


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


# In[16]:


df = neiss.getDataFrame()
date_name, stat_name = 'Treatment_Date', 'Sex'

#showInjuriesPerYear(df, date_name, stat_name, 'Injuries Per Year')
#showInjuriesPerWeekOfTheYear(df, date_name, stat_name, 'Injuries Per 52 Weeks of the Year')
#showInjuriesPerMonthOfTheYear(df, date_name, stat_name, 'Injuries Per Month of the Year')
#showInjuriesPerWeekOfTheYear(df, date_name, stat_name, 'Injuries Per Week of the Year')


# https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
# 

# In[17]:


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
    


# In[18]:


# contingency table
df = neiss_df.copy()
categorical = ['Sex', 'Race', 'Body_Part', 'Diagnosis', 'Disposition', 'Location',
               'Fire_Involvement', 'Product_1', 'Product_2', 'PSU', 'Stratum']
df = df.xs(categorical, axis=1)
df.dropna(inplace=True)
df['Race'] = [int(x) for x in df['Race']]
df = df[0:500]
UniqueId.addIdColumn(df, 'Stratum')
#df.rename(columns={'Stratum_id':'Stratum_code'}, inplace=True)
categorical[categorical.index('Stratum')] = 'Stratum_id'
df.drop('Stratum', inplace=True, axis='columns')
print(df)
print(categorical)

pcs1 = PearsonChiSquared(df)
dfCorrMatrix = pcs1.getCorrMatrixDataframe(categorical)
#print(dfCorrMatrix)

# Mask Upper triangle
# np.triu_indices_from(arr):  Return the indices for the upper-triangle of arr.
#mask = np.zeros_like(dfCorrMatrix)
#mask[np.triu_indices_from(mask)] = True
#mask

fig, ax = plt.subplots(figsize=(10, 10))

#sns.heatmap(dfCorrMatrix, annot=True, mask=mask, fmt="d", linewidths=0.4, ax=ax)
sns.heatmap(dfCorrMatrix, annot=True, linewidths=0.4, ax=ax)

# Fix the top and bottom margins of the heatmap
bottom_y, top_y = plt.ylim() 
bottom_y += 0.5 
top_y -= 0.5 
plt.ylim(bottom_y, top_y)

plt.show() 


# In[19]:


# sns.pairplot(df)


# In[29]:


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
        
        dfSelCorrId = dfSelCorr.copy()
        UniqueId.addIdColumn(dfSelCorrId, xSel)
        UniqueId.addIdColumn(dfSelCorrId, ySel)
        dfSelCorrId.drop([xSel, ySel], axis=1, inplace=True)
        
        #print(dfSelCorr)

        def getColumnDictionary(df, sel):
            try:
                xdict = Neiss.getColumnDictionary(sel)
            except:
                xdict = {x : x for x in df[sel].unique()}
            return xdict

        try:
            xdict = getColumnDictionary(dfSelCorr, xSel)
            xticks = np.sort(dfSelCorr[xSel].unique())
            xlabels = [xdict[x] if x in xdict else x for x in xticks]
            
            # Now 'reindex' the xticks
            xticks = list(range(0, len(xlabels)))
        except Exception as e:
            print(e)
            raise
            
        try:
            ydict = getColumnDictionary(dfSelCorr, ySel)
            yticks = np.sort(dfSelCorr[ySel].unique())
            ylabels = [ydict[y] if y in ydict else y for y in yticks]

            # Now 'reindex' the yticks
            yticks = list(range(0, len(ylabels)))
        except Exception as e:
            print(e)
            raise
            
        #fig, ax = plt.subplots(figsize=(len(xticks) + 1, len(yticks) + 1 >> 1))
        fig, ax = plt.subplots()
        fig.set_size_inches(10,10)

        xSel = xSel + '_id'
        ySel = ySel + '_id'
        if 'swarm' == plotType:
            chart = sns.catplot(x=xSel, y=ySel, data=dfSelCorrId, ax=ax, kind='swarm', col=xSel);
            plt.setp(ax, xticklabels=xlabels, yticklabels=ylabels, yticks=yticks, xticks=xticks)

            #g = sns.FacetGrid(dfSelCorrId, row=xSel, col=ySel)
            #g.map(sns.swarmplot, xSel, ySel)
            #g.set_xticklabels(rotation=90)
        else: # default to scatter if unknown
            chart = sns.scatterplot(x=xSel, y=ySel, data=dfSelCorrId, ax=ax);
            #plt.setp(ax, xticks=xticks, yticks=yticks, yticklabels=ylabels)
            chart.set_yticklabels(labels=ylabels )
            chart.set(yticks=yticks, xticks=xticks)

            chart.set_xticklabels(rotation=90, ha='right', labels=xlabels )
            #chart.set_xticks(xticks)
    
        plt.close(2) # close the second empty graph
        #plt.setp(ax)


# In[31]:


from IPython.display import display
button = widgets.Button(description="Click Me!")
output = widgets.Output()

xDropDownSel1 = widgets.Dropdown(
    options=categorical,
    value=categorical[0],
    description='xCategory',
    disabled=False,
)

yDropDownSel2 = widgets.Dropdown(
    options=categorical,
    value=categorical[1],
    description='yCategory',
    disabled=False,
)

typeDropDownSel3 = widgets.Dropdown(
    options=['swarm', 'scatter'],
    value='swarm',
    description='plot_type',
    disabled=False,
)

wHBox = widgets.HBox([xDropDownSel1, yDropDownSel2, typeDropDownSel3])
wVBox = widgets.VBox([wHBox, button, output])

display(wVBox)

def on_button_clicked(b):
    with output:
        correlations(df, xDropDownSel1.value, yDropDownSel2.value, typeDropDownSel3.value)

button.on_click(on_button_clicked)


# In[22]:


high_correlations = [(y, x) if (y != x) and (dfCorrMatrix[y][x] > 0.95) else np.NaN 
 for y in dfCorrMatrix.index 
     for x in dfCorrMatrix.columns]

high_correlations = pd.Series(high_correlations)
high_correlations.dropna(inplace=True)

print(high_correlations)
#for y, x in high_correlations:
#    print('{}, {}'.format(x, y))
#    correlations(df, x, y, 'swarm')



# In[ ]:





# In[ ]:




