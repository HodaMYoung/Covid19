""" The main objective of this code is to demonstarte how to create a choropleth map
 with two well-known Python libraries, i.e. GeoPandas and folium for current global status
 of Covid-19. The Covid-19 dataset was extracted from WHO status report, while the .shp
 used here for creation of folium choropleth maps were obtained from Natural Earth website.
 The details and further information about the procedure of extracting data and .shp file 
 can be found in the following websites:
 
https://github.com/HodaMYoung/World-GeoJson-File/CountriesGeoJsonData.py
https://github.com/HodaMYoung/Covid19

In order to be able to merge the extracted data from WHO and .shp file, modifications
are required due to inconsistency in the format of countries' names. For instance, Vietnam
was presented as Viet Nam  in WHO report. Another example is South Korea where in the
WHO report refered as 'Republic of Korea'.

The variables of interest to be graphed as choropleth map are:
1- Total Cases
2- Total Deaths
3- New Cases
4- New Deaths
5- Cases per Population in 100,000,(CPP)
6- Deaths per Population in 100,000(DPP)
7- Case Fatality Rate(CFR) %: Number of Deaths/ Number of Cases*100

In order to calculate the last three variables basic information from countries, i.e.
poulation was obtained from wikipedia website. The procedure of how to extract this dataset 
can be found in the following website:

https://gist.github.com/HodaMYoung

Nonetheless, the format of countries' names in Wikipedia websites are occasionally different
from the WHO reports. In order to merge these two datasets modifications are requiered. 

Following stages were applied to the datasets during modification process:

1- Removing dupliactions of countries from WiKipedia extracted dataframes. The reason for
duplications is that some countries like Turkey and Russia are located partially in two continents,
while internationally each country only belongs to one continent. Moreover,the wikipedia lists
primarly consider the countries geopraphical location. For instance, Israel is located in
Middle East while it is considered a European country in the international community.

2- Combining three territories and countries, i.e. Bonaire,Sint Eustatius,Saba to match the WHO
report. Bonaire,Sint Eustatius,Saba are three different entetities in North America, while in WHO
report their results have been reported together.

3-Combining the Venezuela and one of its territories where they are considered as
separated enteties  under two different continents,i.e. North America and South America in Wikipedia.

4- General revision of countries' names for both Wikipedia and WHO report.

It should be noted that the accuracy of calculated data were slightly compromised by considering
references like Wikipedia. But the aim is to demonstrate the general idea of how to create choropleth map."""
#importing required libraries:
import pandas as pd
import numpy as np
import geopandas as gdp
import matplotlib.pyplot as plt
import folium
import os
import json
import matplotlib.font_manager as font_manager
import shapefile
#reading and revising the datasets
#reading datasets
dfCon=pd.read_csv('CountriesWikiAllContinents.csv')
dfCon.drop(columns=['Unnamed: 0'],inplace=True)
df=pd.read_csv('CRWHO158.csv')
df.drop(columns=['Unnamed: 0'],inplace=True)
#Checking names differences between WHO report and Wikipedia
WHO_Not_Wiki_b=df[~df['Country'].isin(dfCon['Country'].unique())]
Wiki_Not_WHO_b=dfCon[~dfCon['Country'].isin(df['Country'].unique())]
#wikipedia revision
""" The revision process contains following stages:
1- Counting the frequency of each country in the dataset, by .value_count() attribute.
2- Finding repeated countries.
3- Sorting and grouping the list of duplicated countries in a manner that will be easier to
eleminate the unwanted repetition by .sort_value() and .groupby attribute.
4- For the numeric values of duplicated countries, the highest values were kept, since those values
refelect the more recent reference in wikipedia.
5- Correcting inforamtion directly by assigning the more accurate values,Ex. Israel corresponding
continent.
6- Combining the data for Bonaire,Sint Eustatius,Saba countries while keeping the order the same as
WHO.
7- Combining Venezuela and its territory.
8- General name revision.
"""
dfCon_count=pd.DataFrame(dfCon['Country'].value_counts())
dfCon_count=pd.DataFrame(dfCon['Country'].value_counts())
dfCon_count.reset_index(inplace=True)
dfCon_count.columns=['Country','Count']
dfCon_Doubles=dfCon_count[dfCon_count['Count']>1]
duplicate_lists=dfCon_Doubles['Country'].unique()
dfCon_duplicates=dfCon[dfCon['Country'].isin(duplicate_lists)]                                          
dfCon_duplicates['Continent'].replace('North ','',regex=True,inplace=True)                                           
dfCon_duplicates.sort_values(by=['Country','Continent'],inplace=True) 
dfCon_R1=dfCon_duplicates.groupby(by=['Country'])[['Area','Population','Density']].max() 
dfCon_R2=dfCon_duplicates.groupby(by=['Country'])[['Capital','Continent']].agg(lambda x:(',').join(x))
dfCon_R1.reset_index(inplace=True)
dfCon_R2.reset_index(inplace=True)
dfCon_R2['Continent']=dfCon_R2['Continent'].str.split(',',expand=True,n=2)[1]
dfCon_R2['Capital']=dfCon_R2['Capital'].str.split(',',expand=True,n=2)[1]
dfCon_R=dfCon_R1.merge(dfCon_R2)
ndxIS=dfCon[dfCon['Country']=='Israel'].index.to_list()
dfCon.iloc[ndxIS,-1]='Europe'
dfCon_n=pd.concat([dfCon,dfCon_R],axis=0)
dfCon_n.drop_duplicates(subset=['Country'],keep='last',inplace=True)
dfCon_n.reset_index(drop=True,inplace=True)
dfCon_n['Country'].replace('Bailiwick of ','',regex=True,inplace=True)
dfCon_n['Country'].replace('The Gambia','Gambia',regex=True,inplace=True)
dfCon_n['Country'].replace('Czech Republic','Czechia',regex=True,inplace=True)
dfCon_n['Country'].replace('United States',\
                           'United States of America',regex=True,inplace=True)
dfCon_n['Country'].replace('Tanzania',\
                           'United Republic of Tanzania',regex=True,inplace=True)
dfCon_n['Country'].replace('United States of America Virgin Islands',\
                           'United States Virgin Islands',regex=True,inplace=True)
dfCon_n['Country'].replace('Cape Verde','Cabo Verde',regex=True,inplace=True)
ndxVNZLs=dfCon_n[dfCon_n['Country'].str.contains('Venezuela')].index.to_list()
VNZLs=dfCon_n.iloc[ndxVNZLs,:]
VNZLs_APD=pd.DataFrame(VNZLs[['Area','Population']].sum()).transpose()
VNZLs_APD['Density']=VNZLs_APD['Population']/VNZLs_APD['Area']
ndxVNZLo=dfCon_n[dfCon_n['Country']=='Venezuela'].index.to_list()
VNZLs_APD['Country']=dfCon_n.iloc[ndxVNZLo[0],0]
VNZLs_APD['Continent']=dfCon_n.iloc[ndxVNZLo[0],-1]
VNZLs_APD['Capital']=dfCon_n.iloc[ndxVNZLo[0],-2]
VNZLs_n=VNZLs_APD[['Country','Area','Population','Density','Capital','Continent']]
dfCon_n.drop(index=ndxVNZLs,inplace=True)
dfCon_n.reset_index(drop=True,inplace=True)
dfCon_n=pd.concat([dfCon_n,VNZLs_n],axis=0)
dfCon_n.reset_index(drop=True,inplace=True)
ndxBSES=dfCon_n[dfCon_n['Country'].isin(['Bonaire','Sint Eustatius','Saba'])].\
         index.to_list()
ndxBSES_order=[ndxBSES[0],ndxBSES[2],ndxBSES[1]]
BSES=dfCon_n.iloc[ndxBSES_order,:]
BSES_APD=BSES[['Area','Population','Continent']].\
          groupby(by=['Continent'])[['Area','Population']].sum()
BSES_APD.reset_index(inplace=True)
BSES_APD['Density']=BSES_APD['Population']/BSES_APD['Area']
BSES_CC=BSES[['Country','Continent','Capital']].\
          groupby(by=['Continent'])[['Country','Capital']].\
          agg(lambda x:(',').join(x))
BSES_CC.reset_index(inplace=True)
BSES_n=BSES_APD.merge(BSES_CC)
BSES_n=BSES_n[['Country','Area','Population','Density','Capital','Continent']]
dfCon_n=pd.concat([dfCon_n,BSES_n],axis=0)
dfCon_n.drop(index=ndxBSES_order,inplace=True)
dfCon_n.reset_index(drop=True,inplace=True)
#General revision of country names in WHO reports
""" The revision process contains following stages:
1- Removing all spaces and unwanted characters.
2- General name revision.
"""
NameR1=df['Country'].str.split('(',expand=True,n=2)
df['Country']=NameR1[0]
NameR2=df['Country'].str.split('[',expand=True,n=2)
df['Country']=NameR2[0]
df['Country']=[x[0:len(x)-1] if x[-1]==' ' else x for x in df['Country']]
df['Country'].replace('Viet Nam','Vietnam',regex=True,inplace=True)
df['Country'].replace('Brunei Darussalam','Brunei',regex=True,inplace=True)
df['Country'].replace('Republic of Korea','South Korea',regex=True,inplace=True)
df['Country'].replace('Republic of Moldova','Moldova',regex=True,inplace=True)
df['Country'].replace('Russian Federation','Russia',regex=True,inplace=True)
df['Country'].replace('Bonaire, Sint Eustatius and Saba',\
                      'Bonaire,Sint Eustatius,Saba',regex=True,inplace=True)
df['Country'].replace('Timor-Leste','East Timor',regex=True,inplace=True)
df['Country'].replace('Côte d’Ivoire','Ivory Coast',regex=True,inplace=True)
df['Country'].replace('Congo','Republic of the Congo',regex=False,inplace=True)
df['Country'].replace('Syrian Arab Republic','Syria',regex=True,inplace=True)
df['Country'].replace('Occupied Palestinian Territory','Palestine',\
                      regex=True,inplace=True)
df['Country'].replace('Bahamas','The Bahamas',regex=True,inplace=True)
#Merging two data
WHO_Not_Wiki_a=df[~df['Country'].isin(dfCon_n['Country'].unique())]
Wiki_Not_WHO_a=dfCon[~dfCon['Country'].isin(df['Country'].unique())]                    
df=df.merge(dfCon_n)
df['Continent'].fillna('Unknown',inplace=True)
#Calculating the CPP,DPP and CFR
df['CPP']=df['Total Cases']/df['Population']*1e5
df['DPP']=df['Total Deaths']/df['Population']*1e5
df['CFR']=df['Total Deaths']/df['Total Cases']*100
# Opening the shapefile with geopandas
shapefile='worldcountries.shp'
gdf = gdp.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
gdf.columns = ['Country', 'Country_code', 'geometry']
gdf_Not_WHO_b=gdf[~gdf['Country'].isin(df['Country'].unique())]
WHO_Not_gdf_b=df[~df['Country'].isin(gdf['Country'].unique())]
gdf['Country'].replace('eSwatini','Eswatini',regex=True,inplace=True)
#Revision to match the WHO reports names with .shp file
""" The revision process only includes general name revision."""
df['Country'].replace('Saint Barthélemy','Saint Barthelemy',regex=True,inplace=True)
df['Country'].replace('Serbia','Republic of Serbia',regex=True,inplace=True)
df['Country'].replace('North Macedonia','Macedonia',regex=True,inplace=True)
df['Country'].replace('São Tomé and Príncipe','São Tomé and Principe',regex=True,inplace=True)
gdf_Not_WHO_a=gdf[~gdf['Country'].isin(df['Country'].unique())]
WHO_Not_gdf_a=df[~df['Country'].isin(gdf['Country'].unique())]
#Merging two dataset WHO with .shpfile
dfgmap=gdf.merge(df)
dfgmap=dfgmap[['Country','geometry', 'Total Cases','Total Deaths','New Cases','New Deaths','CPP',\
               'DPP','CFR','Continent','Area']]
#Extracting and storing the coordination of each country in a new columns with lambda
dfgmap['coords'] = dfgmap['geometry'].apply(lambda x: x.representative_point().coords[:])
#Preparation for plotting
#using font_manger to adjust the properties of figures
font = font_manager.FontProperties(weight='bold', size=10)
fontdict= {'weight': 'bold',
        'size': 10,
        }
#plots:
#Total Cases
ax=dfgmap.plot(column='Total Cases',cmap='Reds',figsize=(100,100),linewidth=0.8,edgecolor='0.8')
for country, case, label in zip(dfgmap['Country'],dfgmap['Total Cases'],dfgmap['coords']):
    if case>4e5:
       if country=='United States of America':
          country='USA'
       plt.annotate(s=country,xy=label[0],horizontalalignment='center',weight='bold', fontsize=8,color='black')
ax.axis('off')
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=df['Total Cases'].min(), vmax=df['Total Cases'].max()))
cbar=plt.colorbar(sm, orientation='vertical', fraction=0.036, pad=0.1, aspect = 30,shrink=0.6)
ticks=list(cbar.get_ticks())
ticks=[int(x) for x in ticks]
cbar.ax.set_yticklabels(ticks,fontdict=fontdict)
ax.set_title('Number of Confirmed Cases',fontdict=fontdict,color='maroon')      
ax.annotate('Source: WHO -https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports',\
		xy=(.25, .2), xycoords='figure fraction', fontsize=10,weight='bold',color='maroon')
plt.show()
#Total Deaths:
ax=dfgmap.plot(column='Total Deaths',cmap='Reds',figsize=(100,100),linewidth=0.8,edgecolor='0.8')
for country, death, label in zip(dfgmap['Country'],dfgmap['Total Deaths'],dfgmap['coords']):
    if death>5e4:
       if country=='United States of America':
          country='USA'
       if country=='United Kingdom':
          country='UK'
       plt.annotate(s=country,xy=label[0],horizontalalignment='center',weight='bold', fontsize=8,color='black')
ax.axis('off')
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=df['Total Deaths'].min(), vmax=df['Total Deaths'].max()))
cbar=plt.colorbar(sm, orientation='vertical', fraction=0.036, pad=0.1, aspect = 30,shrink=0.6)
ticks=list(cbar.get_ticks())
ticks=[int(x) for x in ticks]
cbar.ax.set_yticklabels(ticks,fontdict=fontdict)
ax.set_title('Number of Total Deaths',fontdict=fontdict, color='maroon')      
ax.annotate('Source: WHO -https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports',\
		xy=(.25, .2), xycoords='figure fraction', fontsize=10,weight='bold',color='maroon')
plt.show()
    
#New Cases
ax=dfgmap.plot(column='New Cases',cmap='Reds',figsize=(100,100),linewidth=0.8,edgecolor='0.8')
for country, case, label in zip(dfgmap['Country'],dfgmap['New Cases'],dfgmap['coords']):
    if case>5e3:
       if country=='United States of America':
          country='USA'
       plt.annotate(s=country,xy=label[0],horizontalalignment='center',weight='bold', fontsize=8,color='black')
ax.axis('off')
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=df['New Cases'].min(), vmax=df['New Cases'].max()))
cbar=plt.colorbar(sm, orientation='vertical', fraction=0.036, pad=0.1, aspect = 30,shrink=0.6)
ticks=list(cbar.get_ticks())
ticks=[int(x) for x in ticks]
cbar.ax.set_yticklabels(ticks,fontdict=fontdict)
ax.set_title('Number of Daily New Cases',fontdict=fontdict,color='maroon')      
ax.annotate('Source: WHO -https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports',\
		xy=(.25, .2), xycoords='figure fraction', fontsize=10,weight='bold',color='maroon')
plt.show()

#New Deaths
ax=dfgmap.plot(column='New Deaths',cmap='Reds',figsize=(100,100),linewidth=0.8,edgecolor='0.8')
for country, death, label in zip(dfgmap['Country'],dfgmap['New Deaths'],dfgmap['coords']):
    if death>700:
       if country=='United States of America':
          country='USA'
       plt.annotate(s=country,xy=label[0],horizontalalignment='center',weight='bold', fontsize=8,color='black')
ax.axis('off')
sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=df['New Deaths'].min(), vmax=df['New Deaths'].max()))
cbar=plt.colorbar(sm, orientation='vertical', fraction=0.036, pad=0.1, aspect = 30,shrink=0.6)
ticks=list(cbar.get_ticks())
ticks=[int(x) for x in ticks]
cbar.ax.set_yticklabels(ticks,fontdict=fontdict)
ax.set_title('Number of Daily New Deaths',fontdict=fontdict,color='maroon')      
ax.annotate('Source: WHO -https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports',\
		xy=(.25, .2), xycoords='figure fraction', fontsize=10,weight='bold',color='maroon')
plt.show()
#CPP
ax=dfgmap.plot(column='CPP',cmap='Blues',figsize=(100,100),linewidth=0.8,edgecolor='0.8')
ax.axis('off')
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=df['CPP'].min(), vmax=df['CPP'].max()))
cbar=plt.colorbar(sm, orientation='vertical', fraction=0.036, pad=0.1, aspect = 30,shrink=0.6)
ticks=list(cbar.get_ticks())
ticks=[int(x) for x in ticks]
cbar.ax.set_yticklabels(ticks,fontdict=fontdict)
ax.set_title('Total Cases per Population in 100,000',fontdict=fontdict,color='navy')      
ax.annotate('Source: WHO -https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports',\
		xy=(.25, .2), xycoords='figure fraction', fontsize=10,weight='bold',color='navy')
plt.show()
#DPP
ax=dfgmap.plot(column='DPP',cmap='Blues',figsize=(100,100),linewidth=0.8,edgecolor='0.8')
ax.axis('off')
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=df['DPP'].min(), vmax=df['DPP'].max()))
cbar=plt.colorbar(sm, orientation='vertical', fraction=0.036, pad=0.1, aspect = 30,shrink=0.6)
ticks=list(cbar.get_ticks())
ticks=[int(x) for x in ticks]
cbar.ax.set_yticklabels(ticks,fontdict=fontdict)
ax.set_title('Total Deaths per Population in 100,000',fontdict=fontdict,color='navy')      
ax.annotate('Source: WHO -https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports',\
		xy=(.25, .2), xycoords='figure fraction', fontsize=10,weight='bold',color='navy')
plt.show()
#CFR
ax=dfgmap.plot(column='CFR',cmap='Blues',figsize=(100,100),linewidth=0.8,edgecolor='0.8')
ax.axis('off')
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=df['CFR'].min(), vmax=df['CFR'].max()))
cbar=plt.colorbar(sm, orientation='vertical', fraction=0.036, pad=0.1, aspect = 30,shrink=0.6)
ticks=list(cbar.get_ticks())
ticks=[int(x) for x in ticks]
cbar.ax.set_yticklabels(ticks,fontdict=fontdict)
ax.set_title('Case Fatality Rate%',fontdict=fontdict,color='navy')      
ax.annotate('Source: WHO -https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports',\
		xy=(.25, .2), xycoords='figure fraction', fontsize=10,weight='bold',color='navy')
plt.show()
#Folium & Choropleth map
""" In order to create a choropleth map in folium following stages required:
1- Reading the GeoJson file by using OS library.
2- Creating a map via folium library.
3- Adding choropleth layer to the created map.
4- Saving the map in html format.
key_on is the key element to create the choropleth from a GeoJson. It must be matched
with the field name of .shp file that were used for creation of GeoJson. For instance, if GeoJson file is in
the format of {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"ADMIN": "Indonesia"},"geometry": ...
,the key_on='feature.properties.ADMIN'. Moreover, cosmetic changes to choropleth can be done via changing parameters like
legend_name, threshold_scale for further information following website is recommended:
https://vverde.github.io/blob/interactivechoropleth.html
"""
#Total Cases
wm=folium.Map(location=[40.52,34.34])# centre of the world
geo_world=os.path.join('CountriesAdminCovid.json')#getting the GeoJson file
wm.choropleth(geo_data=geo_world,data=df,columns=['Country','Total Cases'],\
			 key_on='feature.properties.ADMIN',fill_color='YlOrRd',\
			 threshold_scale=[0,50000,500000,1e6,2e6,25e5],fill_opacity=0.7,line_opacity=0.2,\
			 legend_name='COVID-19 Confirmed Cases')
wm.save('June26Cases.html')
#Total Deaths
wm=folium.Map(location=[40.52,34.34])# centre of the world
wm.choropleth(geo_data=geo_world,data=df,columns=['Country','Total Deaths'],\
			 key_on='feature.properties.ADMIN',fill_color='YlOrRd',\
			 threshold_scale=[0,10e3,50e3,100e3,150e3],fill_opacity=0.7,line_opacity=0.2,\
			 legend_name='COVID-19 Total Deaths')
wm.save('June26Deaths.html')
#New Cases
wm=folium.Map(location=[40.52,34.34])# centre of the world
wm.choropleth(geo_data=geo_world,data=df,columns=['Country','New Cases'],\
			 key_on='feature.properties.ADMIN',fill_color='YlOrRd',\
			 fill_opacity=0.7,line_opacity=0.2,\
			 legend_name='COVID-19 New Cases')
wm.save('June26NewCases.html')
#New Deaths
wm=folium.Map(location=[40.52,34.34])# centre of the world
wm.choropleth(geo_data=geo_world,data=df,columns=['Country','New Deaths'],\
			 key_on='feature.properties.ADMIN',fill_color='YlOrRd',\
			 fill_opacity=0.7,line_opacity=0.2,\
			 legend_name='COVID-19 New Deaths')
wm.save('June26NewDeaths.html')
#CPP
wm=folium.Map(location=[40.52,34.34])# centre of the world
wm.choropleth(geo_data=geo_world,data=df,columns=['Country','CPP'],\
			 key_on='feature.properties.ADMIN',fill_color='PuBu',\
			 fill_opacity=0.7,line_opacity=0.2,\
			 legend_name='COVID-19 Total Cases per Population in 100,000')
wm.save('June26CPP.html')
#CPP
wm=folium.Map(location=[40.52,34.34])# centre of the world
wm.choropleth(geo_data=geo_world,data=df,columns=['Country','DPP'],\
			 key_on='feature.properties.ADMIN',fill_color='PuBu',\
			 fill_opacity=0.7,line_opacity=0.2,\
			 legend_name='COVID-19 Total Deaths per Population in 100,000')
wm.save('June26DPP.html')
#CFR
wm=folium.Map(location=[40.52,34.34])# centre of the world
wm.choropleth(geo_data=geo_world,data=df,columns=['Country','CFR'],\
			 key_on='feature.properties.ADMIN',fill_color='PuBu',\
			 fill_opacity=0.7,line_opacity=0.2,\
			 legend_name='COVID-19 Case Fatality Rate')
wm.save('June26CFR.html')
