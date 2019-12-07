#!/usr/bin/env python
# coding: utf-8

# File:  neiss_backend.py

# # Bicycle Crash analysis NEISS data 1999 to 2018 backend
# ## National Electronic Injury Surveillance System
# 
# "CPSC’s National Electronic Injury Surveillance System (NEISS) is a national probability sample of hospitals in the U.S. and its territories. Patient information is collected from each NEISS hospital for every emergency visit involving an injury associated with consumer products."
# 
# https://catalog.data.gov/dataset/cpscs-national-electronic-injury-surveillance-system-neiss
# https://www.cpsc.gov/cgibin/NEISSQuery/home.aspx
# 

# - Selection of geographic areas called primary sampling units (PSU) that are defined within sampling strata.
#
# https://www.cdc.gov/nchs/nhis/singleton_psu.htm

# Imports

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
import scipy.stats as scs
import seaborn as sns
import sys
import xlrd




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
                code = xl_sheet.cell_value(row_idx, 0)
                value = xl_sheet.cell_value(row_idx, 1)
                d2[code] = value
            d1[sheet] = d2
    return d1


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
        '''
        Retrieve the column category translator for the specified column_name

        Args:
            column_name (str):  The name of the column in the dataframe to retrieve the dictionary for.

        Returns:
            The column category translator in the form of a dictionary for the specified column
        '''
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

    def lookupCodeFor(column_header, name_of_variable_in_column):
        '''
        Args:
          column_header  (str):  The name of the column containing the variable to be translated
          name_of_variable_in_column  (str):  The name of the variable in the 'column_header' column
          
        Returns:
          The code representing the variable
        '''
        inverse = {v: k for k, v in Neiss.DICT_CATEGORY_TRANSLATOR[column_header].items()}
        return inverse[name_of_variable_in_column]

    def translateAge(self, df):
        return [age if age < 200 else (age - 200)/12 for age in df['Age']]

    def translateCodes(df):
        return df.replace(Neiss.DICT_CATEGORY_TRANSLATOR)


class PearsonChiSquared:
    '''
    https://machinelearningmastery.com/chi-squared-test-for-machine-learning/
    '''
    def __init__(self, df):
        self.df = df

    def getColumns(self):
        return self.df.columns
    
    def chiSquared(df, column_name1, column_name2):
        dependent = False
        table = df.xs([column_name1, column_name2], axis='columns')
        #print(table)

        # The function takes an array as input representing the contingency table for the 
        # two categorical variables. It returns the calculated statistic and p-value for 
        # interpretation as well as the calculated degrees of freedom and table of expected 
        # frequencies.
        stat, p, dof, expected = scs.chi2_contingency(table)

        # interpret test-statistic
        #
        # We can interpret the statistic by retrieving the critical value from the chi-squared 
        # distribution for the probability and number of degrees of freedom.
        #
        # For example, a probability of 95% can be used, suggesting that the finding of the test 
        # is quite likely given the assumption of the test that the variable is independent. If 
        # the statistic is less than or equal to the critical value, we can fail to reject this 
        # assumption, otherwise it can be rejected.
        prob = 0.95
        critical = scs.chi2.ppf(prob, dof)
        if abs(stat) >= critical:
            dependent = True
        else:
            dependent = False

        # interpret p-value
        # We can also interpret the p-value by comparing it to a chosen significance level, which 
        # would be 5%, calculated by inverting the 95% probability used in the critical value 
        # interpretation.
        alpha = 1 - prob
        if p <= 1 - alpha:
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
            columns = categorical,
            dtype=np.float64)

        dfSrc = self.df.copy()
        #dfSrc = dfSrc + 1  # Make sure there are no zero entries for correlation
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

def correlations(codeIdTranslator, xSel, ySel, plotType):
    '''
    Show the relational (scatter) plot of two columns
    
    Args:
        codeIdTranslator (CodeIdTranslatorDataFrame): ID to/from Code translator
        xSel (str): The name of the first selected column in the dataframe
        ySel (str): The name of the second selected column in the dataframe
    '''
    # codeIdTranslator.setState(newState='code')
    #dfCorrelations = codeIdTranslator.getDataFrame()

    currentState = codeIdTranslator.getState()
    codeIdTranslator.setState('id')
    dfCorrelationsId = codeIdTranslator.getDataFrame().copy()
    codeIdTranslator.setState('code')
    dfCorrelationsCode = codeIdTranslator.getDataFrame().copy()
    codeIdTranslator.setState(currentState)

    dfCorrelationsId   = dfCorrelationsId.xs([xSel, ySel], axis='columns')
    dfCorrelationsCode = dfCorrelationsCode.xs([xSel, ySel], axis='columns')

    if xSel == ySel:
        print("Identity {}={}".format(xSel, ySel))
    else:
        def getColumnDictionary(df, sel):
            try:
                _dict = Neiss.getColumnDictionary(sel)
            except:
                # Revert to using the values themselves if the dictionary is not valid
                print('Reverting dictionary')
                _dict = {x : x for x in df[sel].unique()}
            return _dict


        try:
            xticksCode = np.sort(dfCorrelationsCode[xSel].unique())
            xdict = getColumnDictionary(dfCorrelationsCode, xSel)
            xlabels = [xdict[x] if x in xdict else x for x in xticksCode]
            xticks = list(range(1, len(xlabels) + 1))  # Now 'reindex' the xticks
        except Exception as e:
            print(e)
            raise

        try:
            yticksCode = np.sort(dfCorrelationsCode[ySel].unique())
            ydict = getColumnDictionary(dfCorrelationsCode, ySel)
            ylabels = [ydict[x] if x in ydict else x for x in yticksCode]
            yticks = list(range(1, len(ylabels) + 1))  # Now 'reindex' the xticks
        except Exception as e:
            print(e)
            raise

        #fig, ax = plt.subplots(figsize=(len(xticks) + 1, len(yticks) + 1 >> 1))
        fig, ax = plt.subplots()
        fig.set_size_inches(10,10)

        if 'swarm' == plotType:
            g = sns.catplot(x=xSel, y=ySel, data=dfCorrelationsId, ax=ax, kind='swarm', col=xSel, edgecolor='gray')
            #plt.setp(ax, xticklabels=xlabels, yticklabels=ylabels, yticks=yticks, xticks=xticks)

            ax.set_xticks(xticks, minor=False)
            ax.set_xticklabels(xlabels, rotation='vertical')
            ax.set_yticks(yticks, minor=False)
            ax.set_yticklabels(ylabels)
            #g = sns.FacetGrid(dfSelCorrId, row=xSel, col=ySel)
            #g.map(sns.swarmplot, xSel, ySel)
            #g.set_xticklabels(rotation=90)
        else: # default to scatter if unknown
            #g = sns.scatterplot(x=xSel, y=ySel, data=dfSelCorrId, ax=ax, hue=xSel);
            #g.set(yticks=yticks, xticks=xticks)
            #g.set_xticklabels(rotation=90, ha='right', labels=xlabels )
            #g.set_yticklabels(labels=ylabels )

            g = sns.relplot(x=xSel, y=ySel, data=dfCorrelationsId, ax=ax, hue=xSel, col=ySel);

            ax.set_xticks(xticks, minor=False)
            ax.set_xticklabels(xlabels, rotation='vertical')
            ax.set_yticks(yticks, minor=False)
            ax.set_yticklabels(ylabels)
    
        plt.close(2) # close the second empty graph
        #plt.setp(ax)
