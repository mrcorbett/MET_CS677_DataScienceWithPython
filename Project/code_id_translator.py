import numpy as np
import pandas as pd

class CodeIdTranslator():
    def __init__(self):
        self.dfCodeTranslator = pd.DataFrame()
        self.state = 'notFit'

    def fit(self, df, categories):
        '''
        Provide the ability to convert categorical columns into linear id based columns
        
        Args:
          df         (pd.DataFrame):   The dataframe to be operated on. Assumed to be in 'code' mode.
          categories    (list(str)):  A list of strings representing the column names containing categorical data
          state         (int): The starting state of the dataframe
        '''
        multiLevelCategories = [ categories, ['codeToId', 'idToCode'] ]
        indices = pd.MultiIndex.from_product(
            multiLevelCategories, names=['Category', 'Function'])   # Uses the dot-product to create indices

        
        self.dfCodeTranslator = pd.DataFrame(index = indices, columns = [])

        for index, category in enumerate(multiLevelCategories[0]):
            unique_codes = np.sort(df[category].unique())

            # Generate the id (1 based) to code dictionary and insert it into a temporary dataframe
            idToCode = {index + 1 : value for index, value in enumerate(unique_codes)}
            dfIdToCode = pd.DataFrame(index=[[category], ['idToCode']], columns=['translator'])
            dfIdToCode.loc[(category, 'idToCode'), 'translator'] = [idToCode] # for some reason the dict has to be in a list

            # Generate the code to id dictionary and insert it into a temporary dataframe
            codeToId = {v: k for k, v in idToCode.items()}
            dfCodeToId = pd.DataFrame(index=[[category], ['codeToId']], columns=['translator'])
            dfCodeToId.loc[(category, 'codeToId'), 'translator'] = [codeToId] # for some reason the dict has to be in a list

            # Add the two temporary dataframes to the dfCodeTranslator dataframe
            self.dfCodeTranslator = pd.concat([self.dfCodeTranslator, dfIdToCode, dfCodeToId], sort=False)
        self.dfCodeTranslator.dropna(inplace=True)
        self.dfCodeTranslator = self.dfCodeTranslator.unstack(level=0)
        self.dfCodeTranslator = self.dfCodeTranslator.stack()
        self.state = 'code'

    def _transform(self, df, function, colName, newColName=None):
        '''
        Transform a column in the dataframe (potentially into a new column_
        
        Args:
            df         (pd.DataFrame):  The dataframe to use.
            function   (str): One of:  {'codeToId', 'idToCode'}
            colName    (str): The name of the column to be transformed
            newColName (str): If None, the original colName will be overwritten.  Otherwise, this specifies the new column name
        '''
        if None == newColName:
            newColName = colName

        d = self.dfCodeTranslator.loc[(function, colName)][0][0]
        df[newColName] = [d[x] for x in df[colName]]

    def transformColumn(self, df, colName, newColName=None):
        '''
        Transform a column in the dataframe (potentially into a new column_
        
        Args:
            df         (pd.DataFrame):  The dataframe to use.
            colName    (str): The name of the column to be transformed
            newColName (str): If None, the original colName will be overwritten.  Otherwise, this specifies the new column name
        '''
        if self.state == 'code':
            self._transform(df, 'codeToId', colName, newColName)
            self.state = 'id'
        else:
            raise Exception('Invalid state={} for selected function={}'.format(self.state, function))

    def transformColumns(self, df, columnNames):
        '''
        Transform all columns in the dataframe specified by the categories variable in the constructor.

        Args:
            df         (pd.DataFrame):  The dataframe to use.
            colNames   (str):  A list of the names of the columns in the dataframe to transform.
        '''
        if self.state == 'code':
            for colName in columnNames:
                self.transformColumn(df, colName)
            self.state = 'id'
        else:
            raise Exception('Invalid state={} for selected function={}'.format(self.state, function))

    def inverseTransformColumn(self, df, colName, newColName=None):
        '''
        Inverse transform a column in the dataframe (potentially into a new column_
        
        Args:
            df         (pd.DataFrame):  The dataframe to use.
            colName    (str): The name of the column to be transformed
            newColName (str): If None, the original colName will be overwritten.  Otherwise, this specifies the new column name
        '''
        if self.state == 'id':
            self._transform(df, 'idToCode', colName, newColName)
            self.state = 'code'
        else:
            raise Exception('Invalid state={} for selected function={}'.format(self.state, function))

    def inverseTransformColumns(self, df, columnNames):
        '''
        Transform all columns in the dataframe specified by the categories variable in the constructor._

        Args:
            df         (pd.DataFrame):  The dataframe to use.  Or, an empty dataframe to use the one associated with the class.
            colNames   (str):  A list of the names of the columns in the dataframe to inverse transform.
        '''
        if self.state == 'id':
            for colName in columnNames:
                self.inverseTransformColumn(df, colName)
            self.state = 'code'
        else:
            raise Exception('Invalid state={} for selected function={}'.format(self.state, function))
 
    def __str__(self):
        '''
        Returns:
          The dataframe used for the translation converted to a string.
        '''
        return str(self.dfCodeTranslator)


class CodeIdTranslatorDataFrame(CodeIdTranslator):
    def __init__(self, df, columnNames):
        '''
        Provide the ability to convert categorical columns into linear id based columns
        
        Args:
          df         (pd.DataFrame):   The dataframe to be operated on
          columnNames    (list(str)):  A list of strings representing the column names containing categorical data
        '''
        super().__init__()
        self.df = df
        self.columnNames = columnNames
        super().fit(df, columnNames)        

    def getDataFrame(self):
        return self.df

    def transformColumn(self, colName, newColName=None):
        '''
        Transform a column in the dataframe (potentially into a new column_
        
        Args:
            colName    (str): The name of the column to be transformed
            newColName (str): If None, the original colName will be overwritten.  Otherwise, this specifies the new column name
        '''
        if self.state == 'code':
            super()._transform(self.df, 'codeToId', colName, newColName)
            self.state = 'id'
        else:
            raise Exception('Invalid state={}'.format(self.state))
 
    def transformColumns(self):
        '''
        Transform all columns in the dataframe specified by the categories variable in the constructor._
        '''
        if self.state == 'code':
            for colName in self.columnNames:
                super()._transform(self.df, 'codeToId', colName)
            self.state = 'id'
        else:
            raise Exception('Invalid state={}'.format(self.state))

    def inverseTransformColumn(self, colName, newColName=None):
        '''
        Inverse transform a column in the dataframe (potentially into a new column_
        
        Args:
            colName    (str): The name of the column to be transformed
            newColName (str): If None, the original colName will be overwritten.  Otherwise, this specifies the new column name
        '''
    
        if self.state == 'id':
            self._transform(self.df, 'idToCode', colName, newColName)
            self.state = 'code'
        else:
            raise Exception('Invalid state={}'.format(self.state))

    def inverseTransformColumns(self):
        '''
        Transform all columns in the dataframe specified by the categories variable in the constructor._
        '''
        if self.state == 'id':
            for colName in self.columnNames:
                self._transform(self.df, 'idToCode', colName)
            self.state = 'code'
        else:
            raise Exception('Invalid state={}'.format(self.state))

    def setState(self, newState):
        '''
        Set the current state to either code or id

        Args:
            newState (str):  One of {'code', 'id}
        '''
        if (newState == self.state) and ((self.state == 'code') or (self.state == 'id')):
            return
        elif newState == 'code':
            self.inverseTransformColumns()
        elif newState == 'id':
            self.transformColumns()
        else:
            raise Exception('State is invalid = {}'.format(self.state))

    def getState(self):
        return self.state
