# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 01:25:17 2023

@author: 18and
"""

import functions

#%% Inversion pasiva

    #%% Importamos DF inicial y transformamos
from data import data

df = functions.transform_data(data = data, monto_inv_inicial = 1000000, comision=0.00125)

    #%% Sacamos las fechas de cierre de mes de los archivos

fechas_cierre = functions.date_checker('files/')[0]
fecha_inicial = functions.date_checker('files/')[1]

    #%% Limpiamos los tickers

tickers = functions.ticker_cleaner(df)

    #%% Descargamos los datos para cada activo, y agregamos los valores a la fecha de cierre a un diccionario de diccionarios

diccionario_tickers = functions.dictionary_maker(tickers=tickers, fecha_inicial=fecha_inicial, df=df, fechas_cierre=fechas_cierre, comision=0.00125)

    #%% Convertimos el diccionairo a un df para hacerlo como se pide en el proyecto

df_pasiva = functions.final_df_maker(diccionario_tickers, fecha_inicial)