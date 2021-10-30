import pandas as pd
import os
from os import path

def to_csv_bulk(data: pd.DataFrame, size: int, refreshOutput: bool=True, outfile:str=None):

    if path.exists(outfile) and refreshOutput == False: 
        
        data.to_csv(path_or_buf=outfile,header=False,mode='a')

    else:

        data.to_csv(path_or_buf=outfile,header=True)


def read_csv_chunks(inputfile: str, records:int)->pd.DataFrame:

    """
    Keyword arguments:
    inputfile -- the file to load from the operating syste.
    records -- the number of records to load into memory at a given time

    """
    df = pd.DataFrame()
    for chunk in pd.read_csv(filepath_or_buffer=inputfile,chunksize=records):
        df.append(other=chunk)
    return df
