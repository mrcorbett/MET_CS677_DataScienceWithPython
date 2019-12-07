#!/usr/bin/env python
# coding: utf-8

# File:  time_based_graphs.py

# Imports
import calendar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class TimeBasedGraphs():
    def __init__(self, df, column_dictionary, date_name, stat_name):
        self.df = df
        self.column_dictionary = column_dictionary
        self.date_name = date_name
        self.stat_name = stat_name

    def showInjuriesPerYear(self, sup_title='Injuries Per Year'):
        '''
        Show the accumulated number of injuries per year
        
        Args:
            column_dictionary (dict):  The column category translator in the form of a dictionary for stat_name
            sup_title (str): The super position title to display

        Returns:
            None
        '''
        stat_dict = self.column_dictionary

        df1 = self.df.xs([self.stat_name, self.date_name], axis='columns')
        df2 = df1.copy()
        df2['_count'] = 1
        df2['_year'] = [t.year for t in df2[self.date_name]]

        df3 = pd.pivot_table(
            df2, values='_count', index=['_year'], columns=[self.stat_name],
            aggfunc=np.sum, fill_value=0)

        fig, ax = plt.subplots(figsize=(18,6))
        fig.suptitle(sup_title)
        df3.columns=[stat_dict[key] for key in sorted(stat_dict.keys(), reverse=False)]
        df3.plot(ax=ax, lw=2)
        ax.set_ylabel('Injuries')
        ax.set_xlabel('Year')
        plt.show()

    def showInjuriesPerWeekOfTheYear(self, sup_title='Injuries per week (1 ... 52) of the year'):
        '''
        Show the accumulated number of injuries week (1 ... 52) of the year

        Args:
            sup_title (str): The super position title to display
        
        Returns:
            None
        '''
        stat_dict = self.column_dictionary

        df1 = self.df.xs([self.stat_name, self.date_name], axis='columns')
        df2 = df1.copy()
        df2['_count'] = 1
        df2['_week'] = [t.week for t in df2[self.date_name]]

        df3 = pd.pivot_table(
            df2, values='_count', index=['_week'], columns=[self.stat_name],
            aggfunc=np.sum, fill_value=0)

        fig, ax = plt.subplots(figsize=(18,6))
        fig.suptitle(sup_title)
        df3.columns=[stat_dict[key] for key in sorted(stat_dict.keys(), reverse=False)]
        df3.plot(ax=ax, lw=2)
        ax.set_ylabel('Injuries')
        ax.set_xlabel('Week Of The Year')
        ax.set_xticks(range(1,52))
        plt.show()


    def showInjuriesPerMonthOfTheYear(self, sup_title='Injuries Per Month of the Year'):
        '''
        Show the accumulated number of injuries per month
        
        Args:
            sup_title (str): The super position title to display
        
        Returns:
            None
        '''
        stat_dict = self.column_dictionary

        df1 = self.df.xs([self.stat_name, self.date_name], axis='columns')
        df2 = df1.copy()
        df2['_count'] = 1
        df2['_month'] = [t.month for t in df2[self.date_name]]
        
        
        df3 = pd.pivot_table(
            df2, values='_count', index=['_month'], columns=[self.stat_name],
            aggfunc=np.sum, fill_value=0)
        df3.index = calendar.month_name[1:]

        fig, ax = plt.subplots(figsize=(18,6))
        fig.suptitle(sup_title)
        df3.columns=[stat_dict[key] for key in sorted(stat_dict.keys(), reverse=False)]
        df3.plot(ax=ax, lw=2)
        ax.set_ylabel('Injuries')
        ax.set_xlabel('Month')
        plt.show()


    def showInjuriesPerWeekdayOfTheYear(self, sup_title='Injuries Per Weekday of the Year'):
        '''
        Show the accumulated number of injuries per weekday
        
        Args:
            sup_title (str): The super position title to display
        
        Returns:
            None
        '''
        stat_dict = self.column_dictionary

        df1 = self.df.xs([self.stat_name, self.date_name], axis='columns')
        df2 = df1.copy()
        df2['_count'] = 1
        df2['_week_day'] = [t.weekday() for t in df2[self.date_name]]

        df3 = pd.pivot_table(
            df2, values='_count', index=['_week_day'], columns=[self.stat_name],
            aggfunc=np.sum, fill_value=0)

        #df3.index = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
        df3.index = calendar.day_name
        
        fig, ax = plt.subplots(figsize=(18,6))
        fig.suptitle(sup_title)
        df3.columns=[stat_dict[key] for key in sorted(stat_dict.keys(), reverse=False)]
        df3.plot(ax=ax, lw=2)
        ax.set_ylabel('Injuries')
        ax.set_xlabel('Weekday')
        plt.show()

    def show(self):
        self.showInjuriesPerYear()
        self.showInjuriesPerWeekOfTheYear()
        self.showInjuriesPerMonthOfTheYear()
        self.showInjuriesPerWeekdayOfTheYear()

