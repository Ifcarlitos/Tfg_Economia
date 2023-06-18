import pandas as pd

#Analisi de la demanda de energia electrica en España en 2019
ruta_Demanda_2019 = 'C:/Users/crome/Desktop/TFG_ECO/TFG_Entrega_Parcial_Carlos_Romero_1518680/Demanda/demanda_2019.xlsx'

#Lectura del fichero excel
df_Demanda_2019 = pd.read_excel(ruta_Demanda_2019)

#    Column1     Column2     Column3     Column4     Column5     Column6  ...   Column361          Column362   Column363          Column364   Column365          Column366
# 0            01/ene/19   02/ene/19   03/ene/19   04/ene/19   05/ene/19  ...   26/dic/19          27/dic/19   28/dic/19          29/dic/19   30/dic/19          31/dic/19
# 1  Demanda  582,949806  742,199407  787,764963 

#Eliminacion de la primera columna
df_Demanda_2019 = df_Demanda_2019.drop(df_Demanda_2019.columns[0], axis = 1)

#convertir columanas en filas
df_Demanda_2019 = df_Demanda_2019.transpose()

#renombrar columnas, columna 1 es la fecha y columna 2 es la demanda
df_Demanda_2019 = df_Demanda_2019.rename(columns = {0: 'Fecha', 1: 'Demanda'})

# sacar fecha de la primera columna y dividirla en año, mes y dia, la fecha esta en 01/ene/19
df_Demanda_2019['Año'] = df_Demanda_2019['Fecha'].str.split('/').str[2]
df_Demanda_2019['Mes'] = df_Demanda_2019['Fecha'].str.split('/').str[1]
df_Demanda_2019['Dia'] = df_Demanda_2019['Fecha'].str.split('/').str[0]

# convertir a entero
df_Demanda_2019['Año'] = df_Demanda_2019['Año'].astype(int)
df_Demanda_2019['Dia'] = df_Demanda_2019['Dia'].astype(int)

# pasar de mes ene, feb, mar, etc a numero
df_Demanda_2019['Mes'] = df_Demanda_2019['Mes'].replace(['ene'], 1)
df_Demanda_2019['Mes'] = df_Demanda_2019['Mes'].replace(['feb'], 2)
df_Demanda_2019['Mes'] = df_Demanda_2019['Mes'].replace(['mar'], 3)
df_Demanda_2019['Mes'] = df_Demanda_2019['Mes'].replace(['abr'], 4)
df_Demanda_2019['Mes'] = df_Demanda_2019['Mes'].replace(['may'], 5)
df_Demanda_2019['Mes'] = df_Demanda_2019['Mes'].replace(['jun'], 6)
df_Demanda_2019['Mes'] = df_Demanda_2019['Mes'].replace(['jul'], 7)
df_Demanda_2019['Mes'] = df_Demanda_2019['Mes'].replace(['ago'], 8)
df_Demanda_2019['Mes'] = df_Demanda_2019['Mes'].replace(['sep'], 9)
df_Demanda_2019['Mes'] = df_Demanda_2019['Mes'].replace(['oct'], 10)
df_Demanda_2019['Mes'] = df_Demanda_2019['Mes'].replace(['nov'], 11)
df_Demanda_2019['Mes'] = df_Demanda_2019['Mes'].replace(['dic'], 12)

#año en formato YYYY no YY
df_Demanda_2019['Año'] = df_Demanda_2019['Año'] + 2000

#pasar demanda de string a float
df_Demanda_2019['Demanda'] = df_Demanda_2019['Demanda'].str.replace(',', '.')
df_Demanda_2019['Demanda'] = df_Demanda_2019['Demanda'].astype(float)

#eliminar columna fecha
df_Demanda_2019 = df_Demanda_2019.drop(['Fecha'], axis = 1)

DatosDemanda = df_Demanda_2019