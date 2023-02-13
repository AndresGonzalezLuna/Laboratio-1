# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 01:05:19 2023

@author: 18and
"""

#%% Inversion Pasiva

    #%% Importamos librerias
import pandas as pd
import yfinance as yf
import os


import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

np.random.seed(123)

# Turn off progress printing 
solvers.options['show_progress'] = False

    #%% Funcion para sacar ticker prices 

def info_yahoo(dates,tickers):
    # Se baja la información de YahooFinance
    start_d = dates
    yahoo_data = yf.download(tickers, start= start_d, group_by="close", interval='1d')
    return yahoo_data

    #%% Function para leer archivio inicial
def initial_file_reader(path):
    
    data = pd.read_csv(path, skiprows=2)
    
    return data
    #%% Transformamos data para quedarnos con columnas que queremos
    
def transform_data(data, monto_inv_inicial, comision):
    
    df = data[['Ticker', 'Peso (%)', 'Precio']]
    df = df.loc[0:36,:]
    df['Precio'] = df['Precio'].str.replace(',', '')
    df['Acciones'] = ((monto_inv_inicial*(df['Peso (%)'].astype(float)/100))/df['Precio'].astype(float)).astype(int)
    
    df['Valor'] = df.Precio.astype(float) * df.Acciones
    df['Comisiones']=df.Precio.astype(float) * df.Acciones * comision
    
    return df

    #%% Checamos las fechas de los archivos de cierrre para usar esas fechas
    
def date_checker(path):
    
    archivos = os.listdir(path)
    
    fechas_cierre = []

    for a in archivos:
        
        fecha = a.split('_')[-1].split('.')[0][:4] + '-' + a.split('_')[-1].split('.')[0][4:6:] + '-' + a.split('_')[-1].split('.')[0][-2:]
        
        fechas_cierre.append(fecha)
        
    fecha_inicial = fechas_cierre[0]
    fechas_cierre = fechas_cierre[1:]
    
    return fechas_cierre, fecha_inicial

    #%% Limpiamos los tickers del Df inicial
    
def ticker_cleaner(df):
    
    tickers = list(df['Ticker'])

    for a in range(len(tickers)):
        
        if list(tickers[a])[-1] == '*':
            
            tickers[a] = tickers[a][:-1]
            
        elif tickers[a] == "LIVEPOLC.1":
            
            tickers[a] = 'LIVEPOL1'
            
        else:
            tickers[a] = tickers[a]
            
    return tickers
        
    #%% Descargamos los datos para cada activo, y agregamos los valores a la fecha de cierre a un diccionario de diccionarios
    
def dictionary_maker(tickers, fecha_inicial, df, fechas_cierre, comision):
    
    diccionario_tickers = {}

    for i in range(len(tickers)):
        
        atributos = {}
        
        # Sacamos los precios del ticker
        ticker = (tickers[i]+'.MX')
        start_d = fecha_inicial
        yahoo_data = yf.download(ticker, start= start_d, group_by="close", interval='1d')
        
        # Preguntar cuales usar Adj Close o Close
        num_acciones = df.loc[(df.Ticker == list(df['Ticker'])[i])]['Acciones'][i]
        
        try:
            atributos[fecha_inicial] = (yahoo_data.loc[fecha_inicial, 'Adj Close']*num_acciones)*(1-comision)
        except:
            atributos[fecha_inicial] = 0
            print("Error: ", tickers[i])
        
        for a in range(len(fechas_cierre)):
    
            try:
                valor = (yahoo_data.loc[fechas_cierre[a], 'Adj Close']*num_acciones)
            except:
                valor = 0
                print("Error: ", tickers[i])
    
            atributos[fechas_cierre[a]]= valor
            
        diccionario_tickers[tickers[i]] = atributos
        
    return diccionario_tickers

    #%% Convertimos el diccionairo a un df para hacerlo como se pide en el proyecto
def final_df_maker(diccionario_tickers, fecha_inicial):
    
    master_data_frame = pd.DataFrame(diccionario_tickers)
    master_data_frame['Capital'] = master_data_frame.sum(axis=1)
    
    dataframe = pd.DataFrame(master_data_frame['Capital'])
    dataframe['rend']= dataframe.Capital.pct_change()
    dataframe['rend_acum']=0
    
    #Sacamos el rendmiento acumulado
    for i in dataframe.index:
        
        dataframe.loc[i,'rend_acum'] = (dataframe.loc[i,'Capital']/dataframe.loc[fecha_inicial,'Capital'])-1
        
    # Renombramos como pide el proyceto
    
    df_pasiva = dataframe.copy()
    
    return df_pasiva


#%% Inversion activa
    
    #%%  Portafolio optimo
def optimal_portfolio(returns):
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


    #%% dataframe portafolio optimo
    
def price_getter(tickers, start_d, end_d, period):
    
    # Creamos un dataframe con data de dos anos antes para cada ticker

    diccionario_tickers = {}
    
    for i in range(len(tickers)):
        
        atributos = {}
        
        # Sacamos los precios del ticker
        ticker = (tickers[i]+'.MX')
        #start_d = "2020-01-28"
        #end_d = "2021-01-28"
        yahoo_data = yf.download(ticker, start= start_d, end = end_d, group_by="close", interval=period)
        
    
        
        for a in yahoo_data.index:
    
            try:
                valor = (yahoo_data.loc[a, 'Adj Close'])
            except:
                valor = 0
                print("Error: ", tickers[i])
    
            atributos[a]= valor
            
        diccionario_tickers[tickers[i]] = atributos
        
        
    master_data_frame = pd.DataFrame(diccionario_tickers)
    quitar = ['SITESB.1', 'MXN', 'NMKA']
    master_data_frame.drop(quitar, inplace=True, axis=1)
    
    ret_log = np.log(1 + master_data_frame.pct_change()).dropna()
    
    return master_data_frame, ret_log


def table_maker(pesos, master_data_frame, comision, monto_inv_inicial):
    
    df_pesos = pd.DataFrame(list(pesos), columns = ['Weights'],
                 index = list(master_data_frame.columns))

    df_activa = master_data_frame.T
    df_activa['Pesos (%)'] = df_pesos['Weights']
    fecha_inicio = df_activa.columns[-2]
    df_activa = df_activa[[fecha_inicio,  'Pesos (%)']]
    
    df_activa.reset_index()
    
    df_activa['Num_acciones'] = ((monto_inv_inicial * df_activa['Pesos (%)'])/df_activa[fecha_inicio]).astype(int)
    df_activa['Valor_inicial'] = df_activa['Num_acciones']*df_activa[fecha_inicio]-(df_activa['Num_acciones']*comision)
    
    return df_activa

def rebalanceador(master_data_frame_final, df_activa, comision):
    
    df_transacciones = pd.DataFrame(columns = ['timestamp', 'ticker','titulos_totales', 'titulos_compra', 'comision', 'comision_acum'])
    
    for i in master_data_frame_final.columns:
    
        for a in range(len(master_data_frame_final)):
        
            try:
    
                ret = (master_data_frame_final.loc[master_data_frame_final.index[a+1], i]/master_data_frame_final.loc[master_data_frame_final.index[a], i])-1
    
            except:
                ret = 0
    
            if ret >= 0.05:
    
                acciones_comprar = int(df_activa.loc[i, 'Num_acciones']*0.05)
    
                transaccion = {'timestamp': master_data_frame_final.index[a+1], 'ticker': i, 'titulos_totales': df_activa.loc[i, 'Num_acciones'], 'titulos compra': acciones_comprar, 'comision': acciones_comprar*comision,'comision_acum': 0}
    
                df_activa.loc[i, 'Num_acciones']=int(df_activa.loc[i, 'Num_acciones']*1.05)
    
            elif ret <= -0.05:
    
                acciones_comprar = int(df_activa.loc[i, 'Num_acciones']*0.05)
    
                transaccion = {'timestamp': master_data_frame_final.index[a+1], 'ticker': i,'titulos_totales': df_activa.loc[i, 'Num_acciones'], 'titulos compra': -1*acciones_comprar, 'comision': acciones_comprar*comision,'comision_acum': 0}
    
                df_activa.loc[i, 'Num_acciones']=int(df_activa.loc[i, 'Num_acciones']*0.95)
    
            df_transacciones = df_transacciones.append(transaccion, ignore_index=True)
    
            #print(master_data_frame_final.index[a])

    return df_activa, df_transacciones
    
        
        
    
    
