import pandas as pd

#Analisi de la estructura de energia electrica en España en 2019
ruta_Estructura_2019 = 'C:/Users/crome/Desktop/TFG_ECO/TFG_Entrega_Parcial_Carlos_Romero_1518680/estructura/estructura_2019.xlsx'

#Lectura del fichero excel
df_Estructura_2019 = pd.read_excel(ruta_Estructura_2019)

# pasar filas a columnas
df_Estructura_2019 = df_Estructura_2019.transpose()

#primera fila y primera columna poner fecha
df_Estructura_2019.iloc[0,0] = 'Fecha'

#primera fila como cabecera
new_header = df_Estructura_2019.iloc[0] #
df_Estructura_2019 = df_Estructura_2019[1:]
df_Estructura_2019.columns = new_header

# sacar fecha de la primera columna y dividirla en año, mes y dia, la fecha esta en 01/ene/19
df_Estructura_2019['Año'] = df_Estructura_2019['Fecha'].str.split('/').str[2]
df_Estructura_2019['Mes'] = df_Estructura_2019['Fecha'].str.split('/').str[1]
df_Estructura_2019['Dia'] = df_Estructura_2019['Fecha'].str.split('/').str[0]

#convertir a entero
df_Estructura_2019['Año'] = df_Estructura_2019['Año'].astype(int)
df_Estructura_2019['Dia'] = df_Estructura_2019['Dia'].astype(int)

# pasar de mes ene, feb, mar, etc a numero
df_Estructura_2019['Mes'] = df_Estructura_2019['Mes'].replace(['ene'], 1)
df_Estructura_2019['Mes'] = df_Estructura_2019['Mes'].replace(['feb'], 2)
df_Estructura_2019['Mes'] = df_Estructura_2019['Mes'].replace(['mar'], 3)
df_Estructura_2019['Mes'] = df_Estructura_2019['Mes'].replace(['abr'], 4)
df_Estructura_2019['Mes'] = df_Estructura_2019['Mes'].replace(['may'], 5)
df_Estructura_2019['Mes'] = df_Estructura_2019['Mes'].replace(['jun'], 6)
df_Estructura_2019['Mes'] = df_Estructura_2019['Mes'].replace(['jul'], 7)
df_Estructura_2019['Mes'] = df_Estructura_2019['Mes'].replace(['ago'], 8)
df_Estructura_2019['Mes'] = df_Estructura_2019['Mes'].replace(['sep'], 9)
df_Estructura_2019['Mes'] = df_Estructura_2019['Mes'].replace(['oct'], 10)
df_Estructura_2019['Mes'] = df_Estructura_2019['Mes'].replace(['nov'], 11)
df_Estructura_2019['Mes'] = df_Estructura_2019['Mes'].replace(['dic'], 12)

#año en formato YYYY no YY
df_Estructura_2019['Año'] = df_Estructura_2019['Año'] + 2000

#borrar columna fecha
df_Estructura_2019 = df_Estructura_2019.drop(['Fecha'], axis=1)

#valors nulos a 0
df_Estructura_2019 = df_Estructura_2019.fillna(0)

#renplazar - por 0
df_Estructura_2019 = df_Estructura_2019.replace('-', 0)

#convertir a float los valores de todas las columnas 

#primero reemplazar , por . para poder convertir a float
df_Estructura_2019['Hidráulica'] = df_Estructura_2019['Hidráulica'].str.replace(',', '.')
df_Estructura_2019['Turbinación bombeo'] = df_Estructura_2019['Turbinación bombeo'].str.replace(',', '.')
df_Estructura_2019['Nuclear'] = df_Estructura_2019['Nuclear'].str.replace(',', '.')
df_Estructura_2019['Carbón'] = df_Estructura_2019['Carbón'].str.replace(',', '.')
df_Estructura_2019['Fuel + Gas'] = df_Estructura_2019['Fuel + Gas'].str.replace(',', '.')
df_Estructura_2019['Motores diésel'] = df_Estructura_2019['Motores diésel'].str.replace(',', '.')
df_Estructura_2019['Turbina de gas'] = df_Estructura_2019['Turbina de gas'].str.replace(',', '.')
df_Estructura_2019['Turbina de vapor'] = df_Estructura_2019['Turbina de vapor'].str.replace(',', '.')
df_Estructura_2019['Ciclo combinado'] = df_Estructura_2019['Ciclo combinado'].str.replace(',', '.')
df_Estructura_2019['Hidroeólica'] = df_Estructura_2019['Hidroeólica'].str.replace(',', '.')
df_Estructura_2019['Eólica'] = df_Estructura_2019['Eólica'].str.replace(',', '.')
df_Estructura_2019['Solar fotovoltaica'] = df_Estructura_2019['Solar fotovoltaica'].str.replace(',', '.')
df_Estructura_2019['Solar térmica'] = df_Estructura_2019['Solar térmica'].str.replace(',', '.')
df_Estructura_2019['Otras renovables'] = df_Estructura_2019['Otras renovables'].str.replace(',', '.')
df_Estructura_2019['Cogeneración'] = df_Estructura_2019['Cogeneración'].str.replace(',', '.')
df_Estructura_2019['Residuos no renovables'] = df_Estructura_2019['Residuos no renovables'].str.replace(',', '.')
df_Estructura_2019['Residuos renovables'] = df_Estructura_2019['Residuos renovables'].str.replace(',', '.')
df_Estructura_2019['Generación total'] = df_Estructura_2019['Generación total'].str.replace(',', '.')

#convertir a float
df_Estructura_2019['Hidráulica'] = df_Estructura_2019['Hidráulica'].astype(float)
df_Estructura_2019['Turbinación bombeo'] = df_Estructura_2019['Turbinación bombeo'].astype(float)
df_Estructura_2019['Nuclear'] = df_Estructura_2019['Nuclear'].astype(float)
df_Estructura_2019['Carbón'] = df_Estructura_2019['Carbón'].astype(float)
df_Estructura_2019['Fuel + Gas'] = df_Estructura_2019['Fuel + Gas'].astype(float)
df_Estructura_2019['Motores diésel'] = df_Estructura_2019['Motores diésel'].astype(float)
df_Estructura_2019['Turbina de gas'] = df_Estructura_2019['Turbina de gas'].astype(float)
df_Estructura_2019['Turbina de vapor'] = df_Estructura_2019['Turbina de vapor'].astype(float)
df_Estructura_2019['Ciclo combinado'] = df_Estructura_2019['Ciclo combinado'].astype(float)
df_Estructura_2019['Hidroeólica'] = df_Estructura_2019['Hidroeólica'].astype(float)
df_Estructura_2019['Eólica'] = df_Estructura_2019['Eólica'].astype(float)
df_Estructura_2019['Solar fotovoltaica'] = df_Estructura_2019['Solar fotovoltaica'].astype(float)
df_Estructura_2019['Solar térmica'] = df_Estructura_2019['Solar térmica'].astype(float)
df_Estructura_2019['Otras renovables'] = df_Estructura_2019['Otras renovables'].astype(float)
df_Estructura_2019['Cogeneración'] = df_Estructura_2019['Cogeneración'].astype(float)
df_Estructura_2019['Residuos no renovables'] = df_Estructura_2019['Residuos no renovables'].astype(float)
df_Estructura_2019['Residuos renovables'] = df_Estructura_2019['Residuos renovables'].astype(float)
df_Estructura_2019['Generación total'] = df_Estructura_2019['Generación total'].astype(float)

#valores nulos a 0
df_Estructura_2019 = df_Estructura_2019.fillna(0)

DatosEstructura = df_Estructura_2019
