# <h1 align=center>**PROYECTO DATATHON**

# <h1 align=center>**`Preprocesamiento y uso de herramientas de Machine Learning`**</h1>  

Este proyecto individual hace énfasis en la aplicación de métodos de pre-procesamiento de Datos y su posterior aplicación en modelos de Machine Learning para la resolución de un problema en específico

# <h2> **Objetivos**</h1>
- Realizar labores de Preprocesamiento de Datos. 
- Aplicar métodos de aprendizaje supervisado y/o no supervisado para generar un set de predicciones

## **Tareas a Ejecutar**

1.Las Tareas e pre-procesamiento de datos en este proyecto se pueden resumir en: 
- Eliminación de duplicados
- Eliminación de datos atipicos
- Imputación de valores Nulos
- Selección de Features
- Creación de Pipeline (aún en ejecución) 
    
2. Las tareas en la plicación de un Modelo supervisado y/o no supervisado se resumen en:
- Selección del modelo
- Optimización del modelo (selección de hyperparametros)

## **Archivos del repositorio**
- **`Datasets`**: (./Datasets/) En esta carpeta se encuentran los archivos de los datos fuente utilizados para realizar el proyecto, y también el archivo que se creó con los resultados de preprocesamiento tanto para el set de entrenamiento como para el set de testeo.
- **`Predicciones`**: (./ETL.ipby/) Archivo CSV con la predicción de los datos.
- **`ETL`**: (./ETL.ipby/) Script para realizar el pre-procesamiento de los datos.
- **`modelo`**: (/modelo.ipynb) Script para ejecutar modelos supervisados 

## `1.ETL(Preprocesamiento)`
Con el archivo ETL.py se realizaron las tareas de pre-procesamiento de datos. Haciendo uso de pandas se realizaron las tareas de pre-procesamiento necesaarias: 

+ Eliminación de valores duplicados: Se eliminaron valores duplicados mediante el uso de la columna (**`image_url`**)

+ Limpieza de datos de ubicación geógrafica (**`[longitud,latitud] `**) usando valores encontrados en la literatura se definieron los siguientes intervalos (**`latitud=[19°,66°] `**) (**`longitud= [-155°,-68°]`**)

+ Imputación de nulos para la ubicación géografica, usando para ello los valores promedio de longitud y latitud para cada estado. Esto se facilita ya que el campo estado no cuenta con valores nulos.

```
df_mean_lat = pd.DataFrame()
df_mean_long= pd.DataFrame()

# Calcular el promedio para longitud y latitud según cada estado
y se convierten en diccionarios que luego serviran para el mapeo del Dataframe

df_mean_lat = df_train.groupby('state')['lat'].mean()
df_mean_lat=df_mean_lat.T
def_mean_lat_dict = df_mean_lat.to_dict()

df_mean_long = df_train.groupby('state')['long'].mean()
df_mean_long=df_mean_long.T
def_mean_long_dict = df_mean_long.to_dict()

#Ejecución del mapeo

filterlat= df_train.loc[:,'lat'].isna()
filterlong = df_train.loc[:,'long'].isna()
df_train.loc[filterlat, 'lat'] = df_train.loc[filterlat,"state"].map(def_mean_lat_dict)
df_train.loc[filterlong, 'long'] = df_train.loc[filterlong,"state"].map(def_mean_long_dict)
```

+ Para el campo (**`sqfeet`**) se eliminaron los valores (**`"0"`**) , teniendo en cuenta que estos registros representaban el 0,017% del campo muestral. Además se removieron Outliers siguiendo la regla 3 sigma

+ Para el campo (**`price`**) se eliminaron los valores (**`"0"`**) , teniendo en cuenta que estos registros representaban el 0,3% del campo muestral. Además se removieron Outliers siguiendo la regla 3 sigma

+ Para los campos (**`[beds,baths]`**) se removieron Outliers siguiendo la regla 3 sigma

+ Para el campo (**`laundry_options`**) se han explorado hasta ahora dós métodos: Imputación de valores nulos mediante valores que se presentan con mayor frecuencia para cada tipo de propiedad. Esto se facilita ya que en la categoria (**`type`**) no se encuentran valores nulos

```
df_typeLaundry=df_train.groupby('type')['laundry_options'].value_counts()

dic_type_laundry = {'apartment':'laundry on site',
                    'assisted living':'laundry on site',
                    'condo':'w/d in unit',
                    'cottage/cabin':'no laundry on site',
                    'duplex':'w/d hookups',
                    'flat':'w/d in unit',
                    'townhouse':'w/d in unit',
                    'manufactured':'w/d hookups',
                    'loft':'w/d in unit',
                    'land':'w/d in unit',
                    'in-law':'w/d in unit',
                    'house':'w/d hookups'}

#Ahora se puede Hacer el mapeo para eliminar los valores nulos 
filter_laundry = df_train.loc['laundry_options'].isna()
df_train.loc[filter_laundry, 'laundry_options'] = df_train.loc[filter_laundry,"type"].map(dic_type_laundry)
```
El segundo método que aun tiene errores y por eso no se hizo uso de él hasta el momento sería el reemplazo mediante extracción de datos del campo (**`description`**) previa estandarización del mismo (poner minusculas, extraer cáracteres especiales) y con la busqueda de determinados patrones ejemplo 
```
pattern_l1 = r'\s+(w/d in unit|w/d hookups|laundry on site|laundry in bldg|no laundry on site)+\s'

pattern_l2 = r'\s+(w/d in unit|w/d hookups|laundry on site|laundry in bldg|no laundry on site)'

pattern_l3 = r'(w/d in unit|w/d hookups|laundry on site|laundry in bldg|no laundry on site)+\s'

#Ahora se puede Hacer el mapeo para eliminar los valores nulos 
filter = df_train.loc['laundry_options'].isna()
df_train.loc[filter2, 'laundry_options'] = df_train.loc[filter,"text_clean"].str.extract(pattern_l2, expand = False)
```

+ Para el campo (**`parking_options**) se han explorado hasta ahora dós métodos: Imputación de valores nulos mediante valores que se presentan con mayor frecuencia para cada tipo de propiedad. Esto se facilita ya que en la categoria (**`type`**) no se encuentran valores nulos

```
df_typeParking=df_train.groupby('type')['parking_options'].value_counts()

dict_type_parking = {'apartment':"off-street parking",
                    'assisted living':'off-street parking',
                    'condo':'attached garage',
                    'cottage/cabin':'off-street parking',
                    'duplex':'off-street parking',
                    'flat':'off-street parking',
                    'townhouse':'carport',
                    'loft':'off-street parking',
                    'land':'off-street parking',
                    'in-law':'off-street parking',
                    'house':'attached garage',
                    'manufactured':'off-street parking'}

#Ahora se puede acer el mapeo
filter_parking = df_train.loc[:,'parking_options'].isna()
df_train.loc[filter_parking , 'parking_options'] = df_train.loc[filter_parking ,'type'].map(dict_type_parking)
```
El segundo método que aún tiene errores y por eso no se hizo uso de él hasta el momento sería el reemplazo mediante extracción de datos del campo (**`description`**) previa estandarización del mismo (poner minusculas, extraer cáracteres especiales) y con la busqueda de determinados patrones ejemplo 

```
pattern_l1 = r'\s+(off-street parking|attached garage|carport|detached garage|street parking|no parking|valet_parking)+\s'

pattern_l2 = r'\s+(off-street parking|attached garage|carport|detached garage|street parking|no parking|valet_parking)'

pattern_l3 = r'(off-street parking|attached garage|carport|detached garage|street parking|no parking|valet_parking)+\s'

filter3= df_train.loc[:,'parking_options'].isna()
df_train.loc[filter3, 'parking_options'] = df_train.loc[filter3,"text_clean"].str.extract(pattern_l3, expand = False)
```

## `2.Selección de Features`
Usando un mapa de correlaciones  y agrupandolos por grupos, se puede evidenciar que los parametros **`[sqfeet,beds,baths,smoking_allowed,laundry_options,long]`** muestran mejor correlación. Estos serán los parametros que se usaran en los módelos de machine learning

## `3.Ejecución de modelo Machine learning`

Se han usado hasta ahora los siguientes modelos:

-  Arbol de Decisión : obteniendo un RECALL= 0.8989 y un AVERAGE de 0.8951