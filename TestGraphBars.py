""" The purpose of this code is to visualize the evolution of Covid-19 with time by using Pandas
and  Matplotlib.Pyplot libraries. The graphs are mainly in bar format here. The procedure can be
be summarized as follows:
1- Extracting data from WHO reports
2- Combining the extracted data from WHO with datasets obtained with Wikipedia websites
contain general information about countries.
3- Calculating total cases and deaths per population in 100,000 as well as case fatality rate.
4- Finding the last updated dataset for each month and each continet and stored them
in  new dataframes seperately through a loop.
5- Sorting the new dataframe with descendiing order of total number of cases and total deaths.
6- Grouping the dataframes by month and continents and finding most affected and least affected
countries.
7- Creating pivot tables to summarized the findings.
8- Using Matplotlib.Pyplot to plot the pie and bar charts.

A simmilar procedure can be conducted to monitor Coronavirus changes with a constant
time intervals. More information about the first 2 steps was discussed in:
https://github.com/HodaMYoung/Covid19
https://gist.github.com/HodaMYoung

Source:
https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports"""

# importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager 
#Preprocessing
df=pd.read_csv('WHOrJune29.csv')
df.drop(columns=['Unnamed: 0'],inplace=True)
dfCon_n=pd.read_csv('RevisedWikiCountries.csv')
dfCon_n.drop(columns=['Unnamed: 0'],inplace=True)
months=list(df['Month'].unique())
continents=sorted(df['Continent'].unique())
NumCol=['Total Cases','Total Deaths','Area','Population']#Numeric data columns
dfn=df[['Country','Updated','Total Cases','Total Deaths','Month','Continent',\
        'Area','Population','Density']]
#making sure all numeric data are in numeric format
for col in NumCol:
    dfn[col]=[int(x) for x in dfn[col]]
   
dfn['CPP1e5']=dfn['Total Cases']/dfn['Population']*1e5#Case per population in 100,000
dfn['DPP1e5']=dfn['Total Deaths']/dfn['Population']*1e5#Death per population in 100,000
#preprocessing
#T5 and B5 represent 5 most affected and 5 least affected, respectively.
#cpp, and dpp denote cases per population and death per population in 100,000, respectively.
#
dfN=pd.DataFrame(columns=dfn.columns)
dfT5,dfB5,dfNcount=dfN,dfN,dfN
df_cpp_T5,df_cpp_B5,df_dpp_T5,df_dpp_B5=dfN,dfN,dfN,dfN
df_Africa,df_Asia=dfn[dfn['Continent']=='Africa'],dfn[dfn['Continent']=='Asia']
df_Europe,df_Oceania=dfn[dfn['Continent']=='Europe'],dfn[dfn['Continent']=='Oceania']
df_AmericaN=dfn[dfn['Continent']=='North America']
df_AmericaS=dfn[dfn['Continent']=='South America']
AfricaT5,AfricaB5,AfricaCount=dfN,dfN,dfN
Africa_cpp_T5,Africa_cpp_B5,Africa_dpp_T5,Africa_dpp_B5=dfN,dfN,dfN,dfN
AsiaT5,AsiaB5,AsiaCount=dfN,dfN,dfN
Asia_cpp_T5,Asia_cpp_B5,Asia_dpp_T5,Asia_dpp_B5=dfN,dfN,dfN,dfN
EuropeT5,EuropeB5,EuropeCount=dfN,dfN,dfN
Europe_cpp_T5,Europe_cpp_B5,Europe_dpp_T5,Europe_dpp_B5=dfN,dfN,dfN,dfN
NAmericaT5,NAmericaB5,NAmericaCount=dfN,dfN,dfN
NAmerica_cpp_T5,NAmerica_cpp_B5,NAmerica_dpp_T5,NAmerica_dpp_B5=dfN,dfN,dfN,dfN
SAmericaT5,SAmericaB5,SAmericaCount=dfN,dfN,dfN
SAmerica_cpp_T5,SAmerica_cpp_B5,SAmerica_dpp_T5,SAmerica_dpp_B5=dfN,dfN,dfN,dfN
OceaniaT5,OceaniaB5,OceaniaCount=dfN,dfN,dfN
Oceania_cpp_T5,Oceania_cpp_B5,Oceania_dpp_T5,Oceania_dpp_B5=dfN,dfN,dfN,dfN
#monthly analysis vs.fixed time period
for month in months:
    df0=dfn[dfn['Month']==month]
    df1=df0[df0['Updated']==df0['Updated'].max()]
    df1.sort_values(by=['Total Cases','Total Deaths'],ascending=False,inplace=True)
    ndxDP=df1[df1['Country']=='Diamond Princess'].index.to_list()
    if len(ndxDP)>0:
       df_DP_out=df1.drop(index=[ndxDP[0]])
    else:
        df_DP_out=df1
    dfN=pd.concat([dfN,df1],axis=0)
    dfN.reset_index(drop=True,inplace=True)
    dfNcount=pd.concat([dfNcount,df_DP_out],axis=0)
    dfNcount.reset_index(drop=True,inplace=True)
    dfT5=pd.concat([dfT5,df_DP_out.head(5)],axis=0)#most affected number of cases
    dfB5=pd.concat([dfB5,df_DP_out.tail(5)],axis=0)#least affected number of cases
    df_cpp_T5=pd.concat([df_cpp_T5,df_DP_out.sort_values(by=['CPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by CPP
    df_dpp_T5=pd.concat([df_dpp_T5,df_DP_out.sort_values(by=['DPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by DPP
    df_cpp_B5=pd.concat([df_cpp_B5,df_DP_out.sort_values(by=['CPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by CPP
    df_dpp_B5=pd.concat([df_dpp_B5,df_DP_out.sort_values(by=['DPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by DPP    
    #African Countries:
    Africa0=df1[df1['Continent']=='Africa']
    if len(Africa0)>0:
       Africa0.sort_values(by=['Total Cases','Total Deaths'],ascending=False,inplace=True)
    else:
       Africa0=pd.DataFrame({'Country':[np.nan],'Updated':[df0['Updated'].max()],\
                             'Total Cases':[0],'Total Deaths':[0],'Month':[month],\
                             'Continent':['Africa'],'Area':[0],'Population':[0],\
                             'Density':[0],'CPP1e5':[0],'DPP1e5':[0]})                                                       
    AfricaCount=pd.concat([AfricaCount,Africa0],axis=0)
    AfricaT5=pd.concat([AfricaT5,Africa0.head(5)],axis=0)#most affected
    AfricaB5=pd.concat([AfricaB5,Africa0.tail(5)],axis=0)#least affected
    Africa_cpp_T5=pd.concat([Africa_cpp_T5,Africa0.sort_values(by=['CPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by CPP
    Africa_dpp_T5=pd.concat([Africa_dpp_T5,Africa0.sort_values(by=['DPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by DPP
    Africa_cpp_B5=pd.concat([Africa_cpp_B5,Africa0.sort_values(by=['CPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by CPP
    Africa_dpp_B5=pd.concat([Africa_dpp_B5,Africa0.sort_values(by=['DPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by DPP 
 
    #Asian Countries:
    Asia0=df1[df1['Continent']=='Asia']
    if len(Asia0)>0:
       Asia0.sort_values(by=['Total Cases','Total Deaths'],ascending=False,inplace=True)
    else:
       Asia0=pd.DataFrame({'Country':[np.nan],'Updated':[df0['Updated'].max()],\
                             'Total Cases':[0],'Total Deaths':[0],'Month':[month],\
                             'Continent':['Asia'],'Area':[0],'Population':[0],\
                             'Density':[0],'CPP1e5':[0],'DPP1e5':[0]})                                                       
    AsiaCount=pd.concat([AsiaCount,Asia0],axis=0)
    AsiaT5=pd.concat([AsiaT5,Asia0.head(5)],axis=0)#most affected
    AsiaB5=pd.concat([AsiaB5,Asia0.tail(5)],axis=0)#least affected
    Asia_cpp_T5=pd.concat([Asia_cpp_T5,Asia0.sort_values(by=['CPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by CPP   
    Asia_dpp_T5=pd.concat([Asia_dpp_T5,Asia0.sort_values(by=['DPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by DPP
    Asia_cpp_B5=pd.concat([Asia_cpp_B5,Asia0.sort_values(by=['CPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by CPP
    Asia_dpp_B5=pd.concat([Asia_dpp_B5,Asia0.sort_values(by=['DPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by DPP 
    #European Countries:
    Europe0=df1[df1['Continent']=='Europe']
    if len(Europe0)>0:
       Europe0.sort_values(by=['Total Cases','Total Deaths'],ascending=False,inplace=True)
    else:
       Europe0=pd.DataFrame({'Country':[np.nan],'Updated':[df0['Updated'].max()],\
                             'Total Cases':[0],'Total Deaths':[0],'Month':[month],\
                             'Continent':['Europe'],'Area':[0],'Population':[0],\
                             'Density':[0],'CPP1e5':[0],'DPP1e5':[0]})                                                       
    EuropeCount=pd.concat([EuropeCount,Europe0],axis=0)
    EuropeT5=pd.concat([ EuropeT5, Europe0.head(5)],axis=0)#most affected
    EuropeB5=pd.concat([EuropeB5, Europe0.tail(5)],axis=0)#least affected
    Europe_cpp_T5=pd.concat([Europe_cpp_T5,Europe0.sort_values(by=['CPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by CPP    
    Europe_dpp_T5=pd.concat([Europe_dpp_T5,Europe0.sort_values(by=['DPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by DPP
    Europe_cpp_B5=pd.concat([Europe_cpp_B5,Europe0.sort_values(by=['CPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by CPP
    Europe_dpp_B5=pd.concat([Europe_dpp_B5,Europe0.sort_values(by=['DPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by DPP
    #Oceanian Countries:
    Oceania0=df1[df1['Continent']=='Oceania']
    if len(Oceania0)>0:
       Oceania0.sort_values(by=['Total Cases','Total Deaths'],ascending=False,inplace=True)
    else:
       Oceania0=pd.DataFrame({'Country':[np.nan],'Updated':[df0['Updated'].max()],\
                             'Total Cases':[0],'Total Deaths':[0],'Month':[month],\
                             'Continent':['Oceania'],'Area':[0],'Population':[0],\
                             'Density':[0],'CPP1e5':[0],'DPP1e5':[0]})                                                       
    OceaniaCount=pd.concat([OceaniaCount,Oceania0],axis=0)
    OceaniaT5=pd.concat([ OceaniaT5, Oceania0.head(5)],axis=0)#most affected
    OceaniaB5=pd.concat([OceaniaB5, Oceania0.tail(5)],axis=0)#least affected
    Oceania_cpp_T5=pd.concat([Oceania_cpp_T5,Oceania0.sort_values(by=['CPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by CPP
    Oceania_dpp_T5=pd.concat([Oceania_dpp_T5,Oceania0.sort_values(by=['DPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by DPP
    Oceania_cpp_B5=pd.concat([Oceania_cpp_B5,Oceania0.sort_values(by=['CPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by CPP
    Oceania_dpp_B5=pd.concat([Oceania_dpp_B5,Oceania0.sort_values(by=['DPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by DPP    
    #North Americam Countries:
    NAmerica0=df1[df1['Continent']=='North America']
    if len(NAmerica0)>0:
       NAmerica0.sort_values(by=['Total Cases','Total Deaths'],ascending=False,inplace=True)
    else:
       NAmerica0=pd.DataFrame({'Country':[np.nan],'Updated':[df0['Updated'].max()],\
                             'Total Cases':[0],'Total Deaths':[0],'Month':[month],\
                             'Continent':['North America'],'Area':[0],'Population':[0],\
                             'Density':[0],'CPP1e5':[0],'DPP1e5':[0]})                                                       
    NAmericaCount=pd.concat([NAmericaCount,NAmerica0],axis=0)
    NAmericaT5=pd.concat([ NAmericaT5, NAmerica0.head(5)],axis=0)#most affected
    NAmericaB5=pd.concat([NAmericaB5, NAmerica0.tail(5)],axis=0)#least affected
    NAmerica_cpp_T5=pd.concat([NAmerica_cpp_T5,NAmerica0.sort_values(by=['CPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by CPP
    NAmerica_dpp_T5=pd.concat([NAmerica_dpp_T5,NAmerica0.sort_values(by=['DPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by DPP
    NAmerica_cpp_B5=pd.concat([NAmerica_cpp_B5,NAmerica0.sort_values(by=['CPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by CPP
    NAmerica_dpp_B5=pd.concat([NAmerica_dpp_B5,NAmerica0.sort_values(by=['DPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by DPP    
    #South Americam Countries:
    SAmerica0=df1[df1['Continent']=='South America']
    if len(SAmerica0)>0:
       SAmerica0.sort_values(by=['Total Cases','Total Deaths'],ascending=False,inplace=True)
    else:
       SAmerica0=pd.DataFrame({'Country':[np.nan],'Updated':[df0['Updated'].max()],\
                             'Total Cases':[0],'Total Deaths':[0],'Month':[month],\
                             'Continent':['South America'],'Area':[0],'Population':[0],\
                             'Density':[0],'CPP1e5':[0],'DPP1e5':[0]})                                                       
    SAmericaCount=pd.concat([SAmericaCount,SAmerica0],axis=0)
    SAmericaT5=pd.concat([ SAmericaT5, SAmerica0.head(5)],axis=0)#most affected
    SAmericaB5=pd.concat([SAmericaB5, SAmerica0.tail(5)],axis=0)#least affected
    SAmerica_cpp_T5=pd.concat([SAmerica_cpp_T5,SAmerica0.sort_values(by=['CPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by CPP
    SAmerica_dpp_T5=pd.concat([SAmerica_dpp_T5,SAmerica0.sort_values(by=['DPP1e5'],ascending=False).head(5)],\
                        axis=0)#most affected by DPP
    SAmerica_cpp_B5=pd.concat([SAmerica_cpp_B5,SAmerica0.sort_values(by=['CPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by CPP
    SAmerica_dpp_B5=pd.concat([SAmerica_dpp_B5,SAmerica0.sort_values(by=['DPP1e5'],ascending=False).tail(5)],\
                        axis=0)#least affected by DPP  
   
#worldwide, grouping worldwide data by continent 
dfConTotal1=dfCon_n.groupby(by=['Continent'])['Country'].count()
dfConTotal1=dfConTotal1.reset_index()
dfConTotal1.rename(columns={'Country':'Total'},inplace=True)
dfConTotal2=dfCon_n[['Area', 'Population','Continent']].groupby(by=['Continent']).sum()
dfConTotal2=dfConTotal2.reset_index()
dfConTotal2['Density']=dfConTotal2['Population']/dfConTotal2['Area']
dfNT1=dfNcount[['Country','Month']].groupby(by=['Month'],sort=False).count()
dfConTotal=dfConTotal1.merge(dfConTotal2)
dfNT1.reset_index(inplace=True)
dfNT1.rename(columns={'Country':'No. Countries'},inplace=True)
dfNT2=dfN[['Total Cases','Total Deaths','Population','Area','Month']]\
       .groupby(by=['Month'],sort=False).sum()
dfNT2.reset_index(inplace=True)
dfNT3=dfT5[['Country','Month']].groupby(by=['Month'],\
                                         sort=False).agg(lambda x:','.join(x))
dfNT3.reset_index(inplace=True)
dfNT3.rename(columns={'Country':'5 Most Affected'},inplace=True)
dfNT4=dfB5[['Country','Month']].groupby(by=['Month'],\
                                         sort=False).agg(lambda x:','.join(x))
dfNT4.reset_index(inplace=True)
dfNT4.rename(columns={'Country':'5 Least Affected'},inplace=True)
dfNT5=df_cpp_T5[['Country','Month']].groupby(by=['Month'],\
                                         sort=False).agg(lambda x:','.join(x))
dfNT5.reset_index(inplace=True)
dfNT5.rename(columns={'Country':'5 top CPP'},inplace=True)
dfNT6=df_dpp_T5[['Country','Month']].groupby(by=['Month'],\
                                         sort=False).agg(lambda x:','.join(x))
dfNT6.reset_index(inplace=True)
dfNT6.rename(columns={'Country':'5 top DPP'},inplace=True)
dfNT7=df_cpp_B5[['Country','Month']].groupby(by=['Month'],\
                                         sort=False).agg(lambda x:','.join(x))
dfNT7.reset_index(inplace=True)
dfNT7.rename(columns={'Country':'5 bottom CPP'},inplace=True)
dfNT8=df_dpp_B5[['Country','Month']].groupby(by=['Month'],\
                                         sort=False).agg(lambda x:','.join(x))
dfNT8.reset_index(inplace=True)
dfNT8.rename(columns={'Country':'5 bottom DPP'},inplace=True)
dfNT_monthly=pd.concat([dfNT1,dfNT2,dfNT3,dfNT4,dfNT5,dfNT6,dfNT7,dfNT8],axis=1)
dfNT_monthly = dfNT_monthly.loc[:,~dfNT_monthly.columns.duplicated()]#getting rid of duplicated columns
#monthly evolution in each continent
dfNT1_C=dfNcount[['Country','Month','Continent']].\
         groupby(by=['Month','Continent'],sort=False).count()
dfNT1_C.reset_index(inplace=True)
dfNT1_C.rename(columns={'Country':'No. Countries'},inplace=True)
dfNT2_C=dfN[['Total Cases','Total Deaths','Population','Area',\
             'Month','Continent']].groupby(by=['Month','Continent'],sort=False).sum()
dfNT2_C.reset_index(inplace=True)
#pivot tables by month and continet
#Number of Countries as the value columns
dfCount_pivot=dfNT1_C.pivot(index='Month',columns='Continent',values='No. Countries')
dfCount_pivot.fillna(0,inplace=True)
dfCount_PivOrder1=dfCount_pivot.reindex(months)
#Total Cases and Total Deaths as the value columns
dfCD_pivot=dfNT2_C.pivot(index='Month',columns='Continent',values=['Total Cases','Total Deaths','Population','Area'])
dfCD_pivot.fillna(0,inplace=True)
dfCD_PivOrder1=dfCD_pivot.reindex(months)
#Graphics
#Color sets & fonts 
color_set0=['grey','yellow','forestgreen','maroon','skyblue','chocolate']
color_set1=['grey','brown','chocolate','gold','olive','navy']
color_set2=['black','grey']#Used for Africa
color_set3=['chocolate','peachpuff']#Used for Asia
color_set4=['forestgreen','lime']#Used for Europe
color_set5=['maroon','seashell']#Used forNorth America
color_set6=['skyblue','navy']#Used for Oceania
color_set7=['indigo','pink']#Used for South America
font = font_manager.FontProperties(weight='bold', size=10)
fontdict= {'weight': 'bold',
        'size': 10,
        }
#Worldwide
#world population distribution
ax0=dfConTotal.plot(kind='pie',y='Population',labels=continents,colors=color_set0,fontsize=10,\
                    autopct='%1.0f%%',shadow=True,explode=(0,0,0,0.3,.3,.3),\
                    textprops={'fontsize': 10,'weight':'bold'},startangle= 45)
ax0.set_ylabel('')
ax0.get_legend().remove()
ax0.set_title('Population',fontdict={'fontsize':10,'fontweight':'bold',\
                                     'verticalalignment':'top','horizontalalignment':'center'})
plt.show()
#Number of Countries
plt.bar(x='Month',height='No. Countries',data=dfNT_monthly,\
            color='navy')
ax1=plt.gca()
ylabels1 = list(ax1.get_yticks())
xlocs1=list(ax1.get_xticks())
plt.ylabel('Number of Countries/Territories',fontsize='medium',fontweight='bold')
plt.xlabel('')
plt.xticks(dfNT_monthly['Month'],rotation=0,fontsize='medium',fontweight='bold')
plt.yticks(ylabels1,fontsize='medium',fontweight='bold')
#first finding the value and location of higest jump in values:
prc=[]
for n in range(1,len(months)):
    v=(dfNT_monthly.iloc[n,1]-dfNT_monthly.iloc[n-1,1])*\
       100/dfNT_monthly.iloc[n-1,1]
    prc.append(int(round(v)))
txt1=str(max(prc))+'%'+'Increase'
loc0=prc.index(max(prc))#starting point for the arrow
xlocA0,xlocA1=(xlocs1[loc0]+xlocs1[loc0+1])/2,xlocs1[loc0]
ylocA0,ylocA1=dfNT_monthly.iloc[loc0+1,1],dfNT_monthly.iloc[loc0,1]
plt.annotate('',xy=(xlocA0,ylocA0),xytext=(xlocA1,ylocA1),\
             arrowprops=dict(width=1.5,facecolor='black',edgecolor='black'))
plt.annotate(txt1,xy=(xlocA1,(dfNT_monthly.iloc[loc0+1,1]+dfNT_monthly.iloc[loc0,1])/2*.8),\
             fontsize='medium',fontweight='bold',\
             rotation=.9*np.degrees(np.arctan((ylocA0-ylocA1)/(xlocs1[loc0+1]-xlocs1[loc0]))))
plt.show()
#Number of Cases
plt.bar(x='Month',height='Total Cases',data=dfNT_monthly,\
        color='navy')
ax2=plt.gca()
ylabels2=list(ax2.get_yticks())
xlocs2=list(ax2.get_xticks())
plt.ylabel('Number of Cases',fontsize='medium',fontweight='bold')
plt.xlabel('')
plt.xticks(dfNT_monthly['Month'],rotation=0,fontsize='medium',fontweight='bold')
plt.yticks(ylabels2,fontsize='medium',fontweight='bold')
for n in range(len(dfNT_monthly)):
	if dfNT_monthly.loc[n,'Total Cases']>5e5:
	   txt2=f"{int(dfNT_monthly.loc[n,'Total Cases']):,}"
	   plt.annotate(txt2,xy=(-.35+xlocs2[n],dfNT_monthly.loc[n,'Total Cases']+40),\
                        fontsize='medium',fontweight='bold',rotation=0)
plt.show()
#Number of Cases per Population in 100,000
dfNT_monthly['CPP']=dfNT_monthly['Total Cases']/dfNT_monthly['Population']*1e5
plt.bar(x='Month',height='CPP',data=dfNT_monthly,\
            color='navy')
ax3=plt.gca()
ylabels3=list(ax3.get_yticks())
xlocs3=list(ax3.get_xticks())
plt.ylabel('Number of Cases per Population in 100,000',fontsize='medium',fontweight='bold')
plt.xlabel('')
plt.xticks(dfNT_monthly['Month'],rotation=0,fontsize='medium',fontweight='bold')
plt.yticks(ylabels3,fontsize='medium',fontweight='bold')
for n in range(len(dfNT_monthly)):
	if dfNT_monthly.loc[n,'CPP']>10:
	   txt3=str(int(round(dfNT_monthly.loc[n,'CPP'])))
	   plt.annotate(txt3,xy=(-.1+xlocs3[n],dfNT_monthly.loc[n,'CPP']+.2),\
                        fontsize='medium',fontweight='bold',rotation=0)
plt.show()
#Number of Deaths
plt.bar(x='Month',height='Total Deaths',data=dfNT_monthly,\
            color='navy')
ax4=plt.gca()
ylabels4=list(ax4.get_yticks())
xlocs4=list(ax4.get_xticks())
plt.ylabel('Number of Deaths',fontsize='medium',fontweight='bold')
plt.xlabel('')
plt.xticks(dfNT_monthly['Month'],rotation=0,fontsize='medium',fontweight='bold')
plt.yticks(ylabels4,fontsize='medium',fontweight='bold')
for n in range(len(dfNT_monthly)):
	if dfNT_monthly.loc[n,'Total Deaths']>10000:
	   txt4=f"{int(dfNT_monthly.loc[n,'Total Deaths']):,}"
	   plt.annotate(txt4,xy=(-.25+xlocs4[n],dfNT_monthly.loc[n,'Total Deaths']+8),\
                        fontsize='medium',fontweight='bold',rotation=0)
plt.show()
#Number of Deaths per Population in 100,000
dfNT_monthly['DPP']=dfNT_monthly['Total Deaths']/dfNT_monthly['Population']*1e5
plt.bar(x='Month',height='DPP',data=dfNT_monthly,\
            color='navy')
ax5=plt.gca()
ylabels5=list(ax5.get_yticks())
xlocs5=list(ax5.get_xticks())
plt.ylabel('Number of Deaths per Population in 100,000',fontsize='medium',fontweight='bold')
plt.xlabel('')
plt.xticks(dfNT_monthly['Month'],rotation=0,fontsize='medium',fontweight='bold')
plt.yticks(ylabels5,fontsize='medium',fontweight='bold')
for n in range(len(dfNT_monthly)):
	if dfNT_monthly.loc[n,'DPP']>0.5:
	   txt5=str(round(dfNT_monthly.loc[n,'DPP'],2))
	   plt.annotate(txt5,xy=(-.1+xlocs5[n],dfNT_monthly.loc[n,'DPP']+.02),\
                        fontsize='medium',fontweight='bold',rotation=0)
plt.show()
#CFR
dfNT_monthly['CFR']=dfNT_monthly['Total Deaths']/dfNT_monthly['Total Cases']*100
plt.bar(x='Month',height='CFR',data=dfNT_monthly,\
            color='navy')
ax6=plt.gca()
ylabels6=list(ax6.get_yticks())
xlocs6=list(ax6.get_xticks())
plt.ylabel('Case Fatality Rate%',fontsize='medium',fontweight='bold')
plt.xlabel('')
plt.xticks(dfNT_monthly['Month'],rotation=0,fontsize='medium',fontweight='bold')
plt.yticks(ylabels6,fontsize='medium',fontweight='bold')
for n in range(len(dfNT_monthly)):
	if dfNT_monthly.loc[n,'CFR']>0:
	   txt6=str(round(dfNT_monthly.loc[n,'CFR'],2))+'%'
	   plt.annotate(txt6,xy=(-.2+xlocs6[n],dfNT_monthly.loc[n,'CFR']+.07),\
                        fontsize='medium',fontweight='bold',rotation=0)
plt.show()

#In each Continet
#Number of Countries
ax7=dfCount_PivOrder1.plot(kind='bar',figsize=(13,6),ylim=(0,60),color=color_set1,\
                           width=.8,fontsize=10)
ax7.legend(continents,prop=font)
ax7.set_ylabel('Number of Countries/Territories',fontsize='medium',fontweight='bold')
ax7.set_xlabel('')
ax7.set_xticklabels(months,rotation=0,fontdict=fontdict)
ylabels7=list(ax7.get_yticks())
ax7.set_yticklabels(ylabels7,fontdict=fontdict)
hght7=[x.get_height() for x in ax7.patches]
wdth7=[x.get_width() for x in ax7.patches]
for npatch in range(len(dfCount_PivOrder1.columns.values)):
    widths=-.375+.125*npatch
    for NPatch in range(len(dfCount_PivOrder1.index)):
        widtht7=widths+8*.125*NPatch
        hndx=npatch*len(months)+NPatch
        if hght7[hndx]>0:
            txt7=str(int(hght7[hndx]))
            plt.annotate(txt7, xy=(widtht7,hght7[hndx]+.2),\
                        rotation=0, weight='bold', fontsize=8,color='black')

plt.show()
#Total Cases and Total Deaths per populations in 100,00
for continent  in continents:
    for month in months:
        if dfCD_PivOrder1.loc[month,('Total Cases',continent)]>0:
            dfCD_PivOrder1.loc[month,('CPP',continent)]=dfCD_PivOrder1.loc[month,('Total Cases',continent)]/dfCD_PivOrder1.\
                                                         loc[month,('Population',continent)]*1e5
            dfCD_PivOrder1.loc[month,('DPP',continent)]=dfCD_PivOrder1.loc[month,('Total Deaths',continent)]/dfCD_PivOrder1.\
                                                         loc[month,('Population',continent)]*1e5
            dfCD_PivOrder1.loc[month,('CFR',continent)]=dfCD_PivOrder1.loc[month,('Total Deaths',continent)]/dfCD_PivOrder1.\
                                                         loc[month,('Total Cases',continent)]*100
        else:
            dfCD_PivOrder1.loc[month,('CPP',continent)],dfCD_PivOrder1.loc[month,('DPP',continent)]=0,0
            dfCD_PivOrder1.loc[month,('CFR',continent)]=0

dfCD_PivOrder1=dfCD_PivOrder1[['Total Cases','Total Deaths','CPP','DPP','CFR']]
dfCD_PivOrder1.fillna(0,inplace=True)
#Total Cases
ax8=dfCD_PivOrder1.plot(kind='bar',y='Total Cases',figsize=(13,6),color=color_set1,\
                           width=.8,fontsize=10)
ax8.legend(continents,prop=font)
ax8.set_ylabel('Number of Cases',fontsize='medium',fontweight='bold')
ax8.set_xlabel('')
ax8.set_xticklabels(months,rotation=0,fontdict=fontdict)
ylabels8=list(ax8.get_yticks())
ylabels8=[int(x) for x in ylabels8]
ax8.set_yticklabels(ylabels8,fontdict=fontdict)
hght8=[x.get_height() for x in ax8.patches]
wdth8=[x.get_width() for x in ax8.patches]
for npatch in range(len(continents)):
    widths=-.38+.125*npatch
    for NPatch in range(len(dfCD_PivOrder1)):
        value1=dfCD_PivOrder1.loc[months[NPatch],('Total Cases',continents[npatch])]
        value2=dfNT_monthly.loc[NPatch,'Total Cases']
        textvalueR=value1/value2*100
        if textvalueR>98:
           textvalueR=round(textvalueR,1)
        else:
           textvalueR=int(round(textvalueR))
        widtht8=widths+8*.125*NPatch
        hndx=npatch*len(months)+NPatch
        if textvalueR>5:
           txt8=str(textvalueR)+'%'
           plt.annotate(txt8, xy=(widtht8,hght8[hndx]+.2),\
                        rotation=0, weight='bold', fontsize=7.5,color='black')
plt.show()
           
#Total Deaths
ax9=dfCD_PivOrder1.plot(kind='bar',y='Total Deaths',figsize=(13,6),color=color_set1,\
                           width=.8,fontsize=10)
ax9.legend(continents,prop=font)
ax9.set_ylabel('Number of Deaths',fontsize='medium',fontweight='bold')
ax9.set_xlabel('')
ax9.set_xticklabels(months,rotation=0,fontdict=fontdict)
ylabels9=list(ax9.get_yticks())
ylabels9=[int(x) for x in ylabels9]
ax9.set_yticklabels(ylabels9,fontdict=fontdict)
hght9=[x.get_height() for x in ax9.patches]
wdth9=[x.get_width() for x in ax9.patches]
for npatch in range(len(continents)):
    widths=-.375+.125*npatch
    for NPatch in range(len(dfCD_PivOrder1)):
        value1=dfCD_PivOrder1.loc[months[NPatch],('Total Deaths',continents[npatch])]
        value2=dfNT_monthly.loc[NPatch,'Total Deaths']
        textvalueR=value1/value2*100
        if textvalueR>98:
           textvalueR=round(textvalueR,1)
        else:
           textvalueR=int(round(textvalueR))
        widtht9=widths+8*.125*NPatch
        hndx=npatch*len(months)+NPatch
        if textvalueR>5:
           txt9=str(textvalueR)+'%'
           plt.annotate(txt9, xy=(widtht9,hght9[hndx]+.2),\
                        rotation=0, weight='bold', fontsize=7.5,color='black')

#Total CPP
ax10=dfCD_PivOrder1.plot(kind='bar',y='CPP',figsize=(13,6),color=color_set1,\
                           width=.8,fontsize=10)
ax10.legend(continents,prop=font)
ax10.set_ylabel('Number of Cases per Population in 100,000',fontsize='medium',fontweight='bold')
ax10.set_xlabel('')
ax10.set_xticklabels(months,rotation=0,fontdict=fontdict)
ylabels10=list(ax10.get_yticks())
ylabels10=[int(x) for x in ylabels10]
ax10.set_yticklabels(ylabels10,fontdict=fontdict)
hght10=[x.get_height() for x in ax10.patches]
wdth10=[x.get_width() for x in ax10.patches]
for npatch in range(len(continents)):
    widths=-.375+.125*npatch
    for NPatch in range(len(dfCD_PivOrder1)):
        widtht10=widths+8*.125*NPatch
        hndx=npatch*len(months)+NPatch
        if hght10[hndx]>40:
           txt10=str(int(round(hght10[hndx])))
           plt.annotate(txt10, xy=(widtht10,hght10[hndx]+.2),\
                        rotation=0, weight='bold', fontsize=7.5,color='black')
plt.show()

#Total DPP
ax11=dfCD_PivOrder1.plot(kind='bar',y='DPP',figsize=(13,6),color=color_set1,\
                           width=.8,fontsize=10)
ax11.legend(continents,prop=font)
ax11.set_ylabel('Number of Deaths per Death in 100,000',fontsize='medium',fontweight='bold')
ax11.set_xlabel('')
ax11.set_xticklabels(months,rotation=0,fontdict=fontdict)
ylabels11=list(ax11.get_yticks())
ylabels11=[int(x) for x in ylabels11]
ax11.set_yticklabels(ylabels11,fontdict=fontdict)
hght11=[x.get_height() for x in ax11.patches]
wdth11=[x.get_width() for x in ax11.patches]
for npatch in range(len(continents)):
    widths=-.375+.125*npatch
    for NPatch in range(len(dfCD_PivOrder1)):
        widtht11=widths+8*.125*NPatch
        hndx=npatch*len(months)+NPatch
        if hght11[hndx]>ylabels11[1]*.8:
           txt11=str(int(round(hght11[hndx])))
           plt.annotate(txt11, xy=(widtht11,hght11[hndx]+.08),\
                        rotation=0, weight='bold', fontsize=7.5,color='black')
plt.show()

#Total CFR
ax12=dfCD_PivOrder1.plot(kind='bar',y='CFR',figsize=(13,6),color=color_set1,\
                           width=.8,fontsize=10)
ax12.legend(continents,prop=font)
ax12.set_ylabel('Case Fatality Rate',fontsize='medium',fontweight='bold')
ax12.set_xlabel('')
ax12.set_xticklabels(months,rotation=0,fontdict=fontdict)
ylabels12=list(ax12.get_yticks())
ylabels12=[int(x) for x in ylabels12]
ax12.set_yticklabels(ylabels12,fontdict=fontdict)
hght12=[x.get_height() for x in ax12.patches]
wdth12=[x.get_width() for x in ax12.patches]
for npatch in range(len(continents)):
    widths=-.385+.125*npatch
    for NPatch in range(len(dfCD_PivOrder1)):
        widtht12=widths+8*.125*NPatch
        hndx=npatch*len(months)+NPatch
        if hght12[hndx]>2:
           txt12=str(int(round(hght12[hndx])))+'%'
           plt.annotate(txt12, xy=(widtht12,hght12[hndx]+.08),\
                        rotation=0, weight='bold', fontsize=7.5,color='black')
plt.show()
#Continents details
#Africa
AfricaNo=AfricaCount[['Country','Month']].groupby(by=['Month'],sort=False).count()
AfricaNo.reset_index(inplace=True)
AfricaNo.rename(columns={'Country':'Affected'},inplace=True)
AfricaNo['Total']=dfConTotal[(dfConTotal['Continent']=='Africa')].loc[:,'Total']
AfricaNo['Total'].fillna(AfricaNo['Total'].max(),inplace=True)
AfricaNo['Not Affected']=AfricaNo['Total']-AfricaNo['Affected']
AfricaNumCol=AfricaCount[NumCol+['Month']].groupby(by=['Month'],sort=False).sum()
AfricaNumCol.reset_index(inplace=True)
AfricaNumCol['Density']=AfricaNumCol['Population']/AfricaNumCol['Area']
AfricaNumCol['Density'].fillna(0,inplace=True)
AfricaNumCol.rename(columns={'Area':'Aff_Area','Population':'Aff_Population',\
                             'Density':'Aff_Density'},inplace=True)
AfricaNumCol['T_Area']=dfConTotal[(dfConTotal['Continent']=='Africa')].loc[:,'Area']
AfricaNumCol['T_Area'].fillna(AfricaNumCol['T_Area'].max(),inplace=True)
AfricaNumCol['T_Population']=dfConTotal[(dfConTotal['Continent']=='Africa')]\
                              .loc[:,'Population']
AfricaNumCol['T_Population'].fillna(AfricaNumCol['T_Population'].max(),inplace=True)
AfricaNumCol['N_Population']=AfricaNumCol['T_Population']-AfricaNumCol['Aff_Population']
Africa=AfricaNo.merge(AfricaNumCol)
#grapgh_No_Country_Africa
ax13=Africa[['Month','Affected','Not Affected']].plot(kind='bar',color=color_set2,ylim=(0,80)\
                           ,width=.8,fontsize=10,stacked=True)
ax13.legend(('Affected','Not Affected'),prop=font)
ax13.set_ylabel('Number of African Countries/Territories',fontsize='medium',fontweight='bold')
ax13.set_xlabel('')
ax13.set_xticklabels(Africa['Month'],rotation=0,fontdict=fontdict)
ylabels13=list(ax13.get_yticks())
ax13.set_yticklabels(ylabels13,fontdict=fontdict)
hght13=[x.get_height() for x in ax13.patches]
wdth13=[x.get_width() for x in ax13.patches]
for npatch in range(len(Africa)):
    value1=Africa.iloc[npatch,1]
    value2=Africa.iloc[npatch,2]
    textvalueR=value1/value2*100
    textvalue=round(textvalueR)
    textvalue=int(textvalueR)
    textvalue=str(textvalue)+'%' 
    widths=-.175+1*npatch
    if textvalueR>0:
       txt13=textvalue
       plt.annotate(txt13, xy=(widths,hght13[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Total Cases 
ax14=Africa[['Month','Total Cases']].plot(kind='bar',color='grey',\
                                         width=.8,fontsize=10,stacked=True)
ax14.get_legend().remove()
ax14.set_ylabel('Number of Cases in Africa',fontsize='medium',fontweight='bold')
ax14.set_xlabel('')
ax14.set_xticklabels(Africa['Month'],rotation=0,fontdict=fontdict)
ylabels14=list(ax14.get_yticks())
ax14.set_yticklabels(ylabels14,fontdict=fontdict)
hght14=[x.get_height() for x in ax14.patches]
wdth14=[x.get_width() for x in ax14.patches]
for npatch in range(len(Africa)):
    textvalueR=Africa.iloc[npatch,4]
    widths=-.175+1*npatch
    if textvalueR>3:
       plt.annotate(f"{int(Africa.loc[npatch,'Total Cases']):,}", \
                    xy=(widths,hght14[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Total Deaths 
ax15=Africa[['Month','Total Deaths']].plot(kind='bar',color='grey',\
                                         width=.8,fontsize=10,stacked=True)
ax15.get_legend().remove()
ax15.set_ylabel('Number of Deaths in Africa',fontsize='medium',fontweight='bold')
ax15.set_xlabel('')
ax15.set_xticklabels(Africa['Month'],rotation=0,fontdict=fontdict)
ylabels15=list(ax15.get_yticks())
ax15.set_yticklabels(ylabels15,fontdict=fontdict)
hght15=[x.get_height() for x in ax15.patches]
wdth15=[x.get_width() for x in ax15.patches]
for npatch in range(len(Africa)):
    textvalueR=Africa.iloc[npatch,5]
    widths=-.175+1*npatch
    if textvalueR>ylabels15[0]*.8:
       plt.annotate(f"{int(Africa.loc[npatch,'Total Deaths']):,}", \
                    xy=(widths,hght15[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Fatality Rate
Africa['CFR']=Africa['Total Deaths']/Africa['Total Cases']*100
Africa['CFR'].fillna(0,inplace=True)
ax16=Africa[['Month','CFR']].plot(kind='bar',color='grey',\
                                         width=.8,fontsize=10,stacked=True)
ax16.get_legend().remove()
ax16.set_ylabel('Case Fatality Rate in Africa %',fontsize='medium',fontweight='bold')
ax16.set_xlabel('')
ax16.set_xticklabels(Africa['Month'],rotation=0,fontdict=fontdict)
ylabels16=list(ax16.get_yticks())
ax16.set_yticklabels(ylabels16,fontdict=fontdict)
hght16=[x.get_height() for x in ax16.patches]
wdth16=[x.get_width() for x in ax16.patches]
for npatch in range(len(Africa)):
    textvalueR=round(Africa.loc[npatch,'CFR'],2)
    widths=-.175+1*npatch
    if textvalueR>ylabels16[0]*.8:
       txt16=str(textvalueR)+'%'
       plt.annotate(txt16, xy=(widths,hght16[npatch]+.02),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()

#graph Total Cases per Affected population in 100,000

Africa['CPP']=Africa['Total Cases']/Africa['Aff_Population']*1e5
ax17=Africa[['Month','CPP']].plot(kind='bar',color='grey',width=.8,fontsize=10)
ax17.get_legend().remove()
ax17.set_ylabel('Cases per population in 100,000 in Africa',fontsize='medium',fontweight='bold')
ax17.set_xlabel('')
ax17.set_xticklabels(Africa['Month'],rotation=0,fontdict=fontdict)
ylabels17=list(ax17.get_yticks())
ax17.set_yticklabels(ylabels17,fontdict=fontdict)
hght17=[x.get_height() for x in ax17.patches]
wdth17=[x.get_width() for x in ax17.patches]
for npatch in range(len(Africa)):
    textvalueR=round(Africa.loc[npatch,'CPP'],2)
    widths=-.175+1*npatch
    if textvalueR>ylabels17[0]*.8:
       txt17=str(textvalueR)
       plt.annotate(txt17, xy=(widths,hght17[npatch]+.002),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()

#graph Death per Capita
Africa['DPP']=Africa['Total Deaths']/Africa['Aff_Population']*1e5
ax18=Africa[['Month','DPP']].plot(kind='bar',color='grey',width=.8,fontsize=10)
ax18.get_legend().remove()
ax18.set_ylabel('Deaths per 100,000 population in Africa',fontsize='medium',fontweight='bold')
ax18.set_xlabel('')
ax18.set_xticklabels(Africa['Month'],rotation=0,fontdict=fontdict)
ylabels18=list(ax18.get_yticks())
ylabels18=[round(x,2) for x in ylabels18]
ax18.set_yticklabels(ylabels18,fontdict=fontdict)
hght18=[x.get_height() for x in ax18.patches]
wdth18=[x.get_width() for x in ax18.patches]
for npatch in range(len(Africa)):
    textvalueR=round(Africa.loc[npatch,'DPP'],2)
    widths=-.175+1*npatch
    if textvalueR>ylabels18[0]*.8:
       txt18=str(textvalueR)
       plt.annotate(txt18, xy=(widths,hght18[npatch]+.002),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
# Affected population
ax19=Africa[['Month','Aff_Population','N_Population']].plot(kind='bar',color=[color_set2[1],color_set2[0]]\
                           ,width=.8,ylim=(0,1.4e9),fontsize=10,stacked=True)
ax19.legend(('Affected','Not Affected'),prop=font)
ax19.set_ylabel('African Population',fontsize='medium',fontweight='bold')
ax19.set_xlabel('')
ax19.set_xticklabels(Africa['Month'],rotation=0,fontdict=fontdict)
ylabels19=list(ax19.get_yticks())
ax19.set_yticklabels(ylabels19,fontdict=fontdict)
hght19=[x.get_height() for x in ax19.patches]
wdth19=[x.get_width() for x in ax19.patches]
for npatch in range(len(Africa)):
    value1=Africa.iloc[npatch,7]
    value2=Africa.iloc[npatch,10]
    textvalueR=value1/value2*100
    textvalue=round(textvalueR)
    textvalue=int(textvalueR)
    textvalue=str(textvalue)+'%' 
    widths=-.175+1*npatch
    if textvalueR>0:
       txt19=textvalue
       plt.annotate(txt19, xy=(widths,hght19[npatch]-4),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()

#Asia
AsiaNo=AsiaCount[['Country','Month']].groupby(by=['Month'],sort=False).count()
AsiaNo.reset_index(inplace=True)
AsiaNo.rename(columns={'Country':'Affected'},inplace=True)
AsiaNo['Total']=dfConTotal[(dfConTotal['Continent']=='Asia')].loc[:,'Total']
AsiaNo['Total'].fillna(AsiaNo['Total'].max(),inplace=True)
AsiaNo['Not Affected']=AsiaNo['Total']-AsiaNo['Affected']
AsiaNumCol=AsiaCount[NumCol+['Month']].groupby(by=['Month'],sort=False).sum()
AsiaNumCol.reset_index(inplace=True)
AsiaNumCol['Density']=AsiaNumCol['Population']/AsiaNumCol['Area']
AsiaNumCol['Density'].fillna(0,inplace=True)
AsiaNumCol.rename(columns={'Area':'Aff_Area','Population':'Aff_Population',\
                             'Density':'Aff_Density'},inplace=True)
AsiaNumCol['T_Area']=dfConTotal[(dfConTotal['Continent']=='Asia')].loc[:,'Area']
AsiaNumCol['T_Area'].fillna(AsiaNumCol['T_Area'].max(),inplace=True)
AsiaNumCol['T_Population']=dfConTotal[(dfConTotal['Continent']=='Asia')]\
                              .loc[:,'Population']
AsiaNumCol['T_Population'].fillna(AsiaNumCol['T_Population'].max(),inplace=True)
AsiaNumCol['N_Population']=AsiaNumCol['T_Population']-AsiaNumCol['Aff_Population']
Asia=AsiaNo.merge(AsiaNumCol)
#grapgh_No_Country_Asia
ax20=Asia[['Month','Affected','Not Affected']].plot(kind='bar',color=color_set3,ylim=(0,50)\
                           ,width=.8,fontsize=10,stacked=True)
ax20.legend(('Affected','Not Affected'),prop=font)
ax20.set_ylabel('Number of Asian Countries/Territories',fontsize='medium',fontweight='bold')
ax20.set_xlabel('')
ax20.set_xticklabels(Asia['Month'],rotation=0,fontdict=fontdict)
ylabels20=list(ax20.get_yticks())
ax20.set_yticklabels(ylabels20,fontdict=fontdict)
hght20=[x.get_height() for x in ax20.patches]
wdth20=[x.get_width() for x in ax20.patches]
for npatch in range(len(Asia)):
    value1=Asia.iloc[npatch,1]
    value2=Asia.iloc[npatch,2]
    textvalueR=value1/value2*100
    textvalue=round(textvalueR)
    textvalue=int(textvalueR)
    textvalue=str(textvalue)+'%' 
    widths=-.175+1*npatch
    if textvalueR>0:
       txt20=textvalue
       plt.annotate(txt20, xy=(widths,hght20[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Total Cases 
ax21=Asia[['Month','Total Cases']].plot(kind='bar',color='chocolate',\
                                         width=.8,fontsize=10,stacked=True)
ax21.get_legend().remove()
ax21.set_ylabel('Number of Cases in Asia',fontsize='medium',fontweight='bold')
ax21.set_xlabel('')
ax21.set_xticklabels(Asia['Month'],rotation=0,fontdict=fontdict)
ylabels21=list(ax21.get_yticks())
xlabels21=list(ax21.get_xticks())
ylabels21=[int(x) for x in ylabels21]
ax21.set_yticklabels(ylabels21,fontdict=fontdict)
hght21=[x.get_height() for x in ax21.patches]
wdth21=[x.get_width() for x in ax21.patches]
for npatch in range(len(Asia)):
    textvalueR=Asia.iloc[npatch,4]
    widths=-.1+1*npatch
    if textvalueR>ylabels21[1]*.9:
       plt.annotate(f"{int(Asia.loc[npatch,'Total Cases']):,}", \
                    xy=(xlabels21[npatch]+-.25,hght21[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Total Deaths 
ax22=Asia[['Month','Total Deaths']].plot(kind='bar',color='chocolate',\
                                         width=.8,fontsize=10,stacked=True)
ax22.get_legend().remove()
ax22.set_ylabel('Number of Deaths in Asia',fontsize='medium',fontweight='bold')
ax22.set_xlabel('')
ax22.set_xticklabels(Asia['Month'],rotation=0,fontdict=fontdict)
ylabels22=list(ax22.get_yticks())
ylabels22=[int(x) for x in ylabels22]
ax22.set_yticklabels(ylabels22,fontdict=fontdict)
hght22=[x.get_height() for x in ax22.patches]
wdth22=[x.get_width() for x in ax22.patches]
for npatch in range(len(Asia)):
    textvalueR=Asia.iloc[npatch,5]
    widths=-.25+1*npatch
    if textvalueR>ylabels22[1]*.8:
       plt.annotate(f"{int(Asia.loc[npatch,'Total Deaths']):,}", \
                    xy=(widths,hght22[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Fatality Rate
Asia['CFR']=Asia['Total Deaths']/Asia['Total Cases']*100
Asia['CFR'].fillna(0,inplace=True)
ax23=Asia[['Month','CFR']].plot(kind='bar',color='chocolate',\
                                         width=.8,fontsize=10,stacked=True)
ax23.get_legend().remove()
ax23.set_ylabel('Case Fatality Rate in Asia %',fontsize='medium',fontweight='bold')
ax23.set_xlabel('')
ax23.set_xticklabels(Asia['Month'],rotation=0,fontdict=fontdict)
ylabels23=list(ax23.get_yticks())
ax23.set_yticklabels(ylabels23,fontdict=fontdict)
hght23=[x.get_height() for x in ax23.patches]
wdth23=[x.get_width() for x in ax23.patches]
for npatch in range(len(Asia)):
    textvalueR=round(Asia.loc[npatch,'CFR'],2)
    widths=-.175+1*npatch
    if textvalueR>ylabels23[1]*.8:
       txt23=str(textvalueR)+'%'
       plt.annotate(txt23, xy=(widths,hght23[npatch]+.02),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph Total Cases per Affected population in 100,000
Asia['CPP']=Asia['Total Cases']/Asia['Aff_Population']*1e5
ax24=Asia[['Month','CPP']].plot(kind='bar',color='chocolate',width=.8,fontsize=10)
ax24.get_legend().remove()
ax24.set_ylabel('Cases per population in 100,000 in Asia',fontsize='medium',fontweight='bold')
ax24.set_xlabel('')
ax24.set_xticklabels(Asia['Month'],rotation=0,fontdict=fontdict)
ylabels24=list(ax24.get_yticks())
ax24.set_yticklabels(ylabels24,fontdict=fontdict)
hght24=[x.get_height() for x in ax24.patches]
wdth24=[x.get_width() for x in ax24.patches]
for npatch in range(len(Asia)):
    textvalueR=round(Asia.loc[npatch,'CPP'],2)
    widths=-.175+1*npatch
    if textvalueR>ylabels24[1]*.7:
       txt24=str(textvalueR)
       plt.annotate(txt24, xy=(widths,hght24[npatch]+.002),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph Total Deaths per Affected population in 100,000
Asia['DPP']=Asia['Total Deaths']/Asia['Aff_Population']*1e5
ax25=Asia[['Month','DPP']].plot(kind='bar',color='chocolate',width=.8,fontsize=10)
ax25.get_legend().remove()
ax25.set_ylabel('Deaths per population in 100,000 in Asia',fontsize='medium',fontweight='bold')
ax25.set_xlabel('')
ax25.set_xticklabels(Asia['Month'],rotation=0,fontdict=fontdict)
ylabels25=list(ax25.get_yticks())
ylabels25=[round(x,2) for x in ylabels25]
ax25.set_yticklabels(ylabels25,fontdict=fontdict)
hght25=[x.get_height() for x in ax25.patches]
wdth25=[x.get_width() for x in ax25.patches]
for npatch in range(len(Asia)):
    textvalueR=round(Asia.loc[npatch,'DPP'],2)
    widths=-.175+1*npatch
    if textvalueR>0:
       txt25=str(textvalueR)
       plt.annotate(txt25, xy=(widths,hght25[npatch]+.002),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
# Affected population
ax26=Asia[['Month','Aff_Population','N_Population']].plot(kind='bar',color=color_set3\
                           ,width=.8,ylim=(0,6e9),fontsize=10,stacked=True)
ax26.legend(('Affected','Not Affected'),prop=font)
ax26.set_ylabel('Asian Population',fontsize='medium',fontweight='bold')
ax26.set_xlabel('')
ax26.set_xticklabels(Asia['Month'],rotation=0,fontdict=fontdict)
ylabels26=list(ax26.get_yticks())
ylabels26=[int(x) for x in ylabels26]
ax26.set_yticklabels(ylabels26,fontdict=fontdict)
hght26=[x.get_height() for x in ax26.patches]
wdth26=[x.get_width() for x in ax26.patches]
for npatch in range(len(Asia)):
    value1=Asia.iloc[npatch,7]
    value2=Asia.iloc[npatch,10]
    textvalueR=value1/value2*100
    textvalue=round(textvalueR)
    textvalue=int(textvalueR)
    textvalue=str(textvalue)+'%' 
    widths=-.175+1*npatch
    if textvalueR>0:
       txt26=textvalue
       plt.annotate(txt26, xy=(widths,hght26[npatch]*.9),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()

#Europe
EuropeNo=EuropeCount[['Country','Month']].groupby(by=['Month'],sort=False).count()
EuropeNo.reset_index(inplace=True)
EuropeNo.rename(columns={'Country':'Affected'},inplace=True)
EuropeNo['Total']=dfConTotal[(dfConTotal['Continent']=='Europe')].loc[:,'Total']
EuropeNo['Total'].fillna(EuropeNo['Total'].max(),inplace=True)
EuropeNo['Not Affected']=EuropeNo['Total']-EuropeNo['Affected']
EuropeNumCol=EuropeCount[NumCol+['Month']].groupby(by=['Month'],sort=False).sum()
EuropeNumCol.reset_index(inplace=True)
EuropeNumCol['Density']=EuropeNumCol['Population']/EuropeNumCol['Area']
EuropeNumCol['Density'].fillna(0,inplace=True)
EuropeNumCol.rename(columns={'Area':'Aff_Area','Population':'Aff_Population',\
                             'Density':'Aff_Density'},inplace=True)
EuropeNumCol['T_Area']=dfConTotal[(dfConTotal['Continent']=='Europe')].loc[:,'Area']
EuropeNumCol['T_Area'].fillna(EuropeNumCol['T_Area'].max(),inplace=True)
EuropeNumCol['T_Population']=dfConTotal[(dfConTotal['Continent']=='Europe')]\
                              .loc[:,'Population']
EuropeNumCol['T_Population'].fillna(EuropeNumCol['T_Population'].max(),inplace=True)
EuropeNumCol['N_Population']=EuropeNumCol['T_Population']-EuropeNumCol['Aff_Population']
Europe=EuropeNo.merge(EuropeNumCol)
#grapgh_No_Country_Europe
ax27=Europe[['Month','Affected','Not Affected']].plot(kind='bar',color=color_set4,\
                                                      width=.8,fontsize=10,stacked=True,ylim=(0,80))
ax27.legend(('Affected','Not Affected'),prop=font)
ax27.set_ylabel('Number of Europen Countries/Territories',fontsize='medium',fontweight='bold')
ax27.set_xlabel('')
ax27.set_xticklabels(Europe['Month'],rotation=0,fontdict=fontdict)
ylabels27=list(ax27.get_yticks())
ax27.set_yticklabels(ylabels27,fontdict=fontdict)
hght27=[x.get_height() for x in ax27.patches]
wdth27=[x.get_width() for x in ax27.patches]
for npatch in range(len(Europe)):
    value1=Europe.iloc[npatch,1]
    value2=Europe.iloc[npatch,2]
    textvalueR=value1/value2*100
    textvalue=round(textvalueR)
    textvalue=int(textvalueR)
    textvalue=str(textvalue)+'%' 
    widths=-.175+1*npatch
    if textvalueR>0:
       txt27=textvalue
       plt.annotate(txt27, xy=(widths,hght27[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Total Cases 
ax28=Europe[['Month','Total Cases']].plot(kind='bar',color='forestgreen',\
                                         width=.8,fontsize=10,stacked=True)
ax28.get_legend().remove()
ax28.set_ylabel('Number of Cases in Europe',fontsize='medium',fontweight='bold')
ax28.set_xlabel('')
ax28.set_xticklabels(Europe['Month'],rotation=0,fontdict=fontdict)
ylabels28=list(ax28.get_yticks())
xlabels28=list(ax28.get_xticks())
ylabels28=[int(x) for x in ylabels28]
ax28.set_yticklabels(ylabels28,fontdict=fontdict)
hght28=[x.get_height() for x in ax28.patches]
wdth28=[x.get_width() for x in ax28.patches]
for npatch in range(len(Europe)):
    textvalueR=Europe.iloc[npatch,4]
    widths=-.1+1*npatch
    if textvalueR>ylabels28[1]*.5:
       plt.annotate(f"{int(Europe.loc[npatch,'Total Cases']):,}", \
                    xy=(xlabels28[npatch]+-.25,hght28[npatch]+.2),rotation=0, weight='bold',\
                    fontsize=8,color='black')
plt.show()
#graph_Case Total Deaths 
ax29=Europe[['Month','Total Deaths']].plot(kind='bar',color='forestgreen',\
                                         width=.8,fontsize=10,stacked=True)
ax29.get_legend().remove()
ax29.set_ylabel('Number of Deaths in Europe',fontsize='medium',fontweight='bold')
ax29.set_xlabel('')
ax29.set_xticklabels(Europe['Month'],rotation=0,fontdict=fontdict)
ylabels29=list(ax29.get_yticks())
ylabels29=[int(x) for x in ylabels29]
ax29.set_yticklabels(ylabels29,fontdict=fontdict)
hght29=[x.get_height() for x in ax29.patches]
wdth29=[x.get_width() for x in ax29.patches]
for npatch in range(len(Europe)):
    textvalueR=Europe.iloc[npatch,5]
    widths=-.25+1*npatch
    if textvalueR>ylabels29[1]*.8:
       plt.annotate(f"{int(Europe.loc[npatch,'Total Deaths']):,}", \
                    xy=(widths,hght29[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Fatality Rate
Europe['CFR']=Europe['Total Deaths']/Europe['Total Cases']*100
Europe['CFR'].fillna(0,inplace=True)
ax30=Europe[['Month','CFR']].plot(kind='bar',color='forestgreen',\
                                         width=.8,fontsize=10,stacked=True)
ax30.get_legend().remove()
ax30.set_ylabel('Case Fatality Rate in Europe %',fontsize='medium',fontweight='bold')
ax30.set_xlabel('')
ax30.set_xticklabels(Europe['Month'],rotation=0,fontdict=fontdict)
ylabels30=list(ax30.get_yticks())
ax30.set_yticklabels(ylabels30,fontdict=fontdict)
hght30=[x.get_height() for x in ax30.patches]
wdth30=[x.get_width() for x in ax30.patches]
for npatch in range(len(Europe)):
    textvalueR=round(Europe.loc[npatch,'CFR'],2)
    widths=-.175+1*npatch
    if textvalueR>ylabels30[1]*.8:
       txt30=str(textvalueR)+'%'
       plt.annotate(txt30, xy=(widths,hght30[npatch]+.02),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph Total Cases per Affected population in 100,000
Europe['CPP']=Europe['Total Cases']/Europe['Aff_Population']*1e5
ax31=Europe[['Month','CPP']].plot(kind='bar',color='forestgreen',width=.8,fontsize=10)
ax31.get_legend().remove()
ax31.set_ylabel('Cases per population in 100,000 in Europe',fontsize='medium',fontweight='bold')
ax31.set_xlabel('')
ax31.set_xticklabels(Europe['Month'],rotation=0,fontdict=fontdict)
ylabels31=list(ax31.get_yticks())
ax31.set_yticklabels(ylabels31,fontdict=fontdict)
hght31=[x.get_height() for x in ax31.patches]
wdth31=[x.get_width() for x in ax31.patches]
for npatch in range(len(Europe)):
    textvalueR=round(Europe.loc[npatch,'CPP'],2)
    widths=-.175+1*npatch
    if textvalueR>0:
       txt31=str(textvalueR)
       plt.annotate(txt31, xy=(widths,hght31[npatch]+.002),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph Total Deaths per Affected population in 100,000
Europe['DPP']=Europe['Total Deaths']/Europe['Aff_Population']*1e5
ax32=Europe[['Month','DPP']].plot(kind='bar',color='forestgreen',width=.8,fontsize=10)
ax32.get_legend().remove()
ax32.set_ylabel('Cases per population in 100,000 in Europe',fontsize='medium',fontweight='bold')
ax32.set_xlabel('')
ax32.set_xticklabels(Europe['Month'],rotation=0,fontdict=fontdict)
ylabels32=list(ax32.get_yticks())
ax32.set_yticklabels(ylabels32,fontdict=fontdict)
hght32=[x.get_height() for x in ax32.patches]
wdth32=[x.get_width() for x in ax32.patches]
for npatch in range(len(Europe)):
    textvalueR=round(Europe.loc[npatch,'DPP'],2)
    widths=-.175+1*npatch
    if textvalueR>0:
       txt32=str(textvalueR)
       plt.annotate(txt32, xy=(widths,hght32[npatch]+.002),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
# Affected population
ax33=Europe[['Month','Aff_Population','N_Population']].plot(kind='bar',color=color_set4\
                           ,width=.8,ylim=(0,1.2e9),fontsize=10,stacked=True)
ax33.legend(('Affected','Not Affected'),prop=font)
ax33.set_ylabel('Europen Population',fontsize='medium',fontweight='bold')
ax33.set_xlabel('')
ax33.set_xticklabels(Europe['Month'],rotation=0,fontdict=fontdict)
ylabels33=list(ax33.get_yticks())
ylabels33=[int(x) for x in ylabels33]
ax33.set_yticklabels(ylabels33,fontdict=fontdict)
hght33=[x.get_height() for x in ax33.patches]
wdth33=[x.get_width() for x in ax33.patches]
for npatch in range(len(Europe)):
    value1=Europe.iloc[npatch,7]
    value2=Europe.iloc[npatch,10]
    textvalueR=value1/value2*100
    textvalue=round(textvalueR)
    textvalue=int(textvalueR)
    textvalue=str(textvalue)+'%' 
    widths=-.175+1*npatch
    if textvalueR>0:
       txt33=textvalue
       plt.annotate(txt33, xy=(widths,hght33[npatch]*.85),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#NAmerica
NAmericaNo=NAmericaCount[['Country','Month']].groupby(by=['Month'],sort=False).count()
NAmericaNo.reset_index(inplace=True)
NAmericaNo.rename(columns={'Country':'Affected'},inplace=True)
NAmericaNo['Total']=dfConTotal[(dfConTotal['Continent']=='North America')].loc[:,'Total']
NAmericaNo['Total'].fillna(NAmericaNo['Total'].max(),inplace=True)
NAmericaNo['Not Affected']=NAmericaNo['Total']-NAmericaNo['Affected']
NAmericaNumCol=NAmericaCount[NumCol+['Month']].groupby(by=['Month'],sort=False).sum()
NAmericaNumCol.reset_index(inplace=True)
NAmericaNumCol['Density']=NAmericaNumCol['Population']/NAmericaNumCol['Area']
NAmericaNumCol['Density'].fillna(0,inplace=True)
NAmericaNumCol.rename(columns={'Area':'Aff_Area','Population':'Aff_Population',\
                             'Density':'Aff_Density'},inplace=True)
NAmericaNumCol['T_Area']=dfConTotal[(dfConTotal['Continent']=='North America')].loc[:,'Area']
NAmericaNumCol['T_Area'].fillna(NAmericaNumCol['T_Area'].max(),inplace=True)
NAmericaNumCol['T_Population']=dfConTotal[(dfConTotal['Continent']=='North America')]\
                              .loc[:,'Population']
NAmericaNumCol['T_Population'].fillna(NAmericaNumCol['T_Population'].max(),inplace=True)
NAmericaNumCol['N_Population']=NAmericaNumCol['T_Population']-NAmericaNumCol['Aff_Population']
NAmerica=NAmericaNo.merge(NAmericaNumCol)
#grapgh_No_Country_North America
ax34=NAmerica[['Month','Affected','Not Affected']].plot(kind='bar',color=color_set5,\
                                                      width=.8,fontsize=10,stacked=True,ylim=(0,60))
ax34.legend(('Affected','Not Affected'),prop=font)
ax34.set_ylabel('Number of North American Countries/Territories',fontsize='medium',fontweight='bold')
ax34.set_xlabel('')
ax34.set_xticklabels(NAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels34=list(ax34.get_yticks())
ax34.set_yticklabels(ylabels34,fontdict=fontdict)
hght34=[x.get_height() for x in ax34.patches]
wdth34=[x.get_width() for x in ax34.patches]
for npatch in range(len(NAmerica)):
    value1=NAmerica.iloc[npatch,1]
    value2=NAmerica.iloc[npatch,2]
    textvalueR=value1/value2*100
    textvalue=round(textvalueR)
    textvalue=int(textvalueR)
    textvalue=str(textvalue)+'%' 
    widths=-.175+1*npatch
    if textvalueR>0:
       txt34=textvalue
       plt.annotate(txt34, xy=(widths,hght34[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Total Cases 
ax35=NAmerica[['Month','Total Cases']].plot(kind='bar',color='maroon',\
                                         width=.8,fontsize=10,stacked=True)
ax35.get_legend().remove()
ax35.set_ylabel('Number of Cases in North America',fontsize='medium',fontweight='bold')
ax35.set_xlabel('')
ax35.set_xticklabels(NAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels35=list(ax35.get_yticks())
xlabels35=list(ax35.get_xticks())
ylabels35=[int(x) for x in ylabels35]
ax35.set_yticklabels(ylabels35,fontdict=fontdict)
hght35=[x.get_height() for x in ax35.patches]
wdth35=[x.get_width() for x in ax35.patches]
for npatch in range(len(NAmerica)):
    textvalueR=NAmerica.iloc[npatch,4]
    widths=-.1+1*npatch
    if textvalueR>ylabels35[1]*.3:
       plt.annotate(f"{int(NAmerica.loc[npatch,'Total Cases']):,}", \
                    xy=(xlabels35[npatch]+-.25,hght35[npatch]+.2),rotation=0, weight='bold',\
                    fontsize=8,color='black')
plt.show()
#graph_Case Total Deaths 
ax36=NAmerica[['Month','Total Deaths']].plot(kind='bar',color='maroon',\
                                         width=.8,fontsize=10,stacked=True)
ax36.get_legend().remove()
ax36.set_ylabel('Number of Deaths in North America',fontsize='medium',fontweight='bold')
ax36.set_xlabel('')
ax36.set_xticklabels(NAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels36=list(ax36.get_yticks())
ylabels36=[int(x) for x in ylabels36]
ax36.set_yticklabels(ylabels36,fontdict=fontdict)
hght36=[x.get_height() for x in ax36.patches]
wdth36=[x.get_width() for x in ax36.patches]
for npatch in range(len(NAmerica)):
    textvalueR=NAmerica.iloc[npatch,5]
    widths=-.25+1*npatch
    if textvalueR>0:
       plt.annotate(f"{int(NAmerica.loc[npatch,'Total Deaths']):,}", \
                    xy=(widths,hght36[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Fatality Rate
NAmerica['CFR']=NAmerica['Total Deaths']/NAmerica['Total Cases']*100
NAmerica['CFR'].fillna(0,inplace=True)
ax37=NAmerica[['Month','CFR']].plot(kind='bar',color='maroon',\
                                         width=.8,fontsize=10,stacked=True)
ax37.get_legend().remove()
ax37.set_ylabel('Case Fatality Rate in North America %',fontsize='medium',fontweight='bold')
ax37.set_xlabel('')
ax37.set_xticklabels(NAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels37=list(ax37.get_yticks())
ax37.set_yticklabels(ylabels37,fontdict=fontdict)
hght37=[x.get_height() for x in ax37.patches]
wdth37=[x.get_width() for x in ax37.patches]
for npatch in range(len(NAmerica)):
    textvalueR=round(NAmerica.loc[npatch,'CFR'],2)
    widths=-.175+1*npatch
    if textvalueR>0:
       txt37=str(textvalueR)+'%'
       plt.annotate(txt37, xy=(widths,hght37[npatch]+.02),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph Total Cases per Affected population in 100,000
NAmerica['CPP']=NAmerica['Total Cases']/NAmerica['Aff_Population']*1e5
ax38=NAmerica[['Month','CPP']].plot(kind='bar',color='maroon',width=.8,fontsize=10)
ax38.get_legend().remove()
ax38.set_ylabel('Cases per population in 100,000 in North America',fontsize='medium',fontweight='bold')
ax38.set_xlabel('')
ax38.set_xticklabels(NAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels38=list(ax38.get_yticks())
ax38.set_yticklabels(ylabels38,fontdict=fontdict)
hght38=[x.get_height() for x in ax38.patches]
wdth38=[x.get_width() for x in ax38.patches]
for npatch in range(len(NAmerica)):
    textvalueR=round(NAmerica.loc[npatch,'CPP'],2)
    widths=-.175+1*npatch
    if textvalueR>ylabels38[1]*.5:
       txt38=str(textvalueR)
       plt.annotate(txt38, xy=(widths,hght38[npatch]+.002),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph Total Deaths per Affected population in 100,000
NAmerica['DPP']=NAmerica['Total Deaths']/NAmerica['Aff_Population']*1e5
ax39=NAmerica[['Month','DPP']].plot(kind='bar',color='maroon',width=.8,fontsize=10)
ax39.get_legend().remove()
ax39.set_ylabel('Cases per population in 100,000 in North America',fontsize='medium',fontweight='bold')
ax39.set_xlabel('')
ax39.set_xticklabels(NAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels39=list(ax39.get_yticks())
ax39.set_yticklabels(ylabels39,fontdict=fontdict)
hght39=[x.get_height() for x in ax39.patches]
wdth39=[x.get_width() for x in ax39.patches]
for npatch in range(len(NAmerica)):
    textvalueR=round(NAmerica.loc[npatch,'DPP'],2)
    widths=-.175+1*npatch
    if textvalueR>0:
       txt39=str(textvalueR)
       plt.annotate(txt39, xy=(widths,hght39[npatch]+.002),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
# Affected population
ax40=NAmerica[['Month','Aff_Population','N_Population']].plot(kind='bar',color=color_set5\
                           ,width=.8,ylim=(0,8e8),fontsize=10,stacked=True)
ax40.legend(('Affected','Not Affected'),prop=font)
ax40.set_ylabel('North American Population',fontsize='medium',fontweight='bold')
ax40.set_xlabel('')
ax40.set_xticklabels(NAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels40=list(ax40.get_yticks())
ylabels40=[int(x) for x in ylabels40]
ax40.set_yticklabels(ylabels40,fontdict=fontdict)
hght40=[x.get_height() for x in ax40.patches]
wdth40=[x.get_width() for x in ax40.patches]
for npatch in range(len(NAmerica)):
    value1=NAmerica.iloc[npatch,7]
    value2=NAmerica.iloc[npatch,10]
    textvalueR=value1/value2*100
    textvalue=round(textvalueR)
    textvalue=int(textvalueR)
    textvalue=str(textvalue)+'%' 
    widths=-.175+1*npatch
    if textvalueR>0:
       txt40=textvalue
       plt.annotate(txt40, xy=(widths,hght40[npatch]*.9),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#Oceania
OceaniaNo=OceaniaCount[['Country','Month']].groupby(by=['Month'],sort=False).count()
OceaniaNo.reset_index(inplace=True)
OceaniaNo.rename(columns={'Country':'Affected'},inplace=True)
OceaniaNo['Total']=dfConTotal[(dfConTotal['Continent']=='Oceania')].loc[:,'Total']
OceaniaNo['Total'].fillna(OceaniaNo['Total'].max(),inplace=True)
OceaniaNo['Not Affected']=OceaniaNo['Total']-OceaniaNo['Affected']
OceaniaNumCol=OceaniaCount[NumCol+['Month']].groupby(by=['Month'],sort=False).sum()
OceaniaNumCol.reset_index(inplace=True)
OceaniaNumCol['Density']=OceaniaNumCol['Population']/OceaniaNumCol['Area']
OceaniaNumCol['Density'].fillna(0,inplace=True)
OceaniaNumCol.rename(columns={'Area':'Aff_Area','Population':'Aff_Population',\
                             'Density':'Aff_Density'},inplace=True)
OceaniaNumCol['T_Area']=dfConTotal[(dfConTotal['Continent']=='Oceania')].loc[:,'Area']
OceaniaNumCol['T_Area'].fillna(OceaniaNumCol['T_Area'].max(),inplace=True)
OceaniaNumCol['T_Population']=dfConTotal[(dfConTotal['Continent']=='Oceania')]\
                              .loc[:,'Population']
OceaniaNumCol['T_Population'].fillna(OceaniaNumCol['T_Population'].max(),inplace=True)
OceaniaNumCol['N_Population']=OceaniaNumCol['T_Population']-OceaniaNumCol['Aff_Population']
Oceania=OceaniaNo.merge(OceaniaNumCol)
#grapgh_No_Country_Oceania
ax41=Oceania[['Month','Affected','Not Affected']].plot(kind='bar',color=[color_set6[1],color_set6[0]],\
                                                      width=.8,fontsize=10,stacked=True,ylim=(0,36))
ax41.legend(('Affected','Not Affected'),prop=font)
ax41.set_ylabel('Number of Oceanian Countries/Territories',fontsize='medium',fontweight='bold')
ax41.set_xlabel('')
ax41.set_xticklabels(Oceania['Month'],rotation=0,fontdict=fontdict)
ylabels41=list(ax41.get_yticks())
ax41.set_yticklabels(ylabels41,fontdict=fontdict)
hght41=[x.get_height() for x in ax41.patches]
wdth41=[x.get_width() for x in ax41.patches]
for npatch in range(len(Oceania)):
    value1=Oceania.iloc[npatch,1]
    value2=Oceania.iloc[npatch,2]
    textvalueR=value1/value2*100
    textvalue=round(textvalueR)
    textvalue=int(textvalueR)
    textvalue=str(textvalue)+'%' 
    widths=-.175+1*npatch
    if textvalueR>0:
       txt41=textvalue
       plt.annotate(txt41, xy=(widths,hght41[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Total Cases 
ax42=Oceania[['Month','Total Cases']].plot(kind='bar',color='skyblue',\
                                         width=.8,fontsize=10,stacked=True)
ax42.get_legend().remove()
ax42.set_ylabel('Number of Cases in Oceania',fontsize='medium',fontweight='bold')
ax42.set_xlabel('')
ax42.set_xticklabels(Oceania['Month'],rotation=0,fontdict=fontdict)
ylabels42=list(ax42.get_yticks())
xlabels42=list(ax42.get_xticks())
ylabels42=[int(x) for x in ylabels42]
ax42.set_yticklabels(ylabels42,fontdict=fontdict)
hght42=[x.get_height() for x in ax42.patches]
wdth42=[x.get_width() for x in ax42.patches]
for npatch in range(len(Oceania)):
    textvalueR=Oceania.iloc[npatch,4]
    widths=-.1+1*npatch
    if textvalueR>ylabels42[1]*.3:
       plt.annotate(f"{int(Oceania.loc[npatch,'Total Cases']):,}", \
                    xy=(xlabels42[npatch]+-.25,hght42[npatch]+20),rotation=0, weight='bold',\
                    fontsize=8,color='black')
plt.show()
#graph_Case Total Deaths 
ax43=Oceania[['Month','Total Deaths']].plot(kind='bar',color='skyblue',\
                                         width=.8,fontsize=10,stacked=True)
ax43.get_legend().remove()
ax43.set_ylabel('Number of Deaths in Oceania',fontsize='medium',fontweight='bold')
ax43.set_xlabel('')
ax43.set_xticklabels(Oceania['Month'],rotation=0,fontdict=fontdict)
ylabels43=list(ax43.get_yticks())
ylabels43=[int(x) for x in ylabels43]
ax43.set_yticklabels(ylabels43,fontdict=fontdict)
hght43=[x.get_height() for x in ax43.patches]
wdth43=[x.get_width() for x in ax43.patches]
for npatch in range(len(Oceania)):
    textvalueR=Oceania.iloc[npatch,5]
    widths=-.25+1*npatch
    if textvalueR>0:
       plt.annotate(f"{int(Oceania.loc[npatch,'Total Deaths']):,}", \
                    xy=(widths,hght43[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Fatality Rate
Oceania['CFR']=Oceania['Total Deaths']/Oceania['Total Cases']*100
Oceania['CFR'].fillna(0,inplace=True)
ax44=Oceania[['Month','CFR']].plot(kind='bar',color='skyblue',\
                                         width=.8,fontsize=10,stacked=True)
ax44.get_legend().remove()
ax44.set_ylabel('Case Fatality Rate in Oceania %',fontsize='medium',fontweight='bold')
ax44.set_xlabel('')
ax44.set_xticklabels(Oceania['Month'],rotation=0,fontdict=fontdict)
ylabels44=list(ax44.get_yticks())
ylabels44=[round(x,2) for x in ylabels44]
ax44.set_yticklabels(ylabels44,fontdict=fontdict)
hght44=[x.get_height() for x in ax44.patches]
wdth44=[x.get_width() for x in ax44.patches]
for npatch in range(len(Oceania)):
    textvalueR=round(Oceania.loc[npatch,'CFR'],2)
    widths=-.175+1*npatch
    if textvalueR>0:
       txt44=str(textvalueR)+'%'
       plt.annotate(txt44, xy=(widths,hght44[npatch]+.01),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph Total Cases per Affected population in 100,000
Oceania['CPP']=Oceania['Total Cases']/Oceania['Aff_Population']*1e5
ax45=Oceania[['Month','CPP']].plot(kind='bar',color='skyblue',width=.8,fontsize=10)
ax45.get_legend().remove()
ax45.set_ylabel('Cases per population in 100,000 in Oceania',fontsize='medium',fontweight='bold')
ax45.set_xlabel('')
ax45.set_xticklabels(Oceania['Month'],rotation=0,fontdict=fontdict)
ylabels45=list(ax45.get_yticks())
ax45.set_yticklabels(ylabels45,fontdict=fontdict)
hght45=[x.get_height() for x in ax45.patches]
wdth45=[x.get_width() for x in ax45.patches]
for npatch in range(len(Oceania)):
    textvalueR=round(Oceania.loc[npatch,'CPP'],2)
    widths=-.175+1*npatch
    if textvalueR>ylabels45[1]*.5:
       txt45=str(textvalueR)
       plt.annotate(txt45, xy=(widths,hght45[npatch]+.002),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph Total Deaths per Affected population in 100,000
Oceania['DPP']=Oceania['Total Deaths']/Oceania['Aff_Population']*1e5
ax46=Oceania[['Month','DPP']].plot(kind='bar',color='skyblue',width=.8,fontsize=10)
ax46.get_legend().remove()
ax46.set_ylabel('Cases per population in 100,000 in Oceania',fontsize='medium',fontweight='bold')
ax46.set_xlabel('')
ax46.set_xticklabels(Oceania['Month'],rotation=0,fontdict=fontdict)
ylabels46=list(ax46.get_yticks())
ylabels46=[round(x,2) for x in ylabels46]
ax46.set_yticklabels(ylabels46,fontdict=fontdict)
hght46=[x.get_height() for x in ax46.patches]
wdth46=[x.get_width() for x in ax46.patches]
for npatch in range(len(Oceania)):
    textvalueR=round(Oceania.loc[npatch,'DPP'],2)
    widths=-.175+1*npatch
    if textvalueR>0:
       txt46=str(textvalueR)
       plt.annotate(txt46, xy=(widths,hght46[npatch]+.002),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
# Affected population
ax47=Oceania[['Month','Aff_Population','N_Population']].plot(kind='bar',color=[color_set6[1],color_set6[0]]\
                           ,width=.8,ylim=(0,5.6e7),fontsize=10,stacked=True)
ax47.legend(('Affected','Not Affected'),prop=font)
ax47.set_ylabel('Oceanian Population',fontsize='medium',fontweight='bold')
ax47.set_xlabel('')
ax47.set_xticklabels(Oceania['Month'],rotation=0,fontdict=fontdict)
ylabels47=list(ax47.get_yticks())
ylabels47=[int(x) for x in ylabels47]
ax47.set_yticklabels(ylabels47,fontdict=fontdict)
hght47=[x.get_height() for x in ax47.patches]
wdth47=[x.get_width() for x in ax47.patches]
for npatch in range(len(Oceania)):
    value1=Oceania.iloc[npatch,7]
    value2=Oceania.iloc[npatch,10]
    textvalueR=value1/value2*100
    textvalue=round(textvalueR)
    textvalue=int(textvalueR)
    textvalue=str(textvalue)+'%' 
    widths=-.175+1*npatch
    if textvalueR>0:
       txt47=textvalue
       plt.annotate(txt47, xy=(widths,hght47[npatch]+2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#SAmerica
SAmericaNo=SAmericaCount[['Country','Month']].groupby(by=['Month'],sort=False).count()
SAmericaNo.reset_index(inplace=True)
SAmericaNo.rename(columns={'Country':'Affected'},inplace=True)
SAmericaNo['Total']=dfConTotal[(dfConTotal['Continent']=='South America')].loc[:,'Total']
SAmericaNo['Total'].fillna(SAmericaNo['Total'].max(),inplace=True)
SAmericaNo['Not Affected']=SAmericaNo['Total']-SAmericaNo['Affected']
SAmericaNumCol=SAmericaCount[NumCol+['Month']].groupby(by=['Month'],sort=False).sum()
SAmericaNumCol.reset_index(inplace=True)
SAmericaNumCol['Density']=SAmericaNumCol['Population']/SAmericaNumCol['Area']
SAmericaNumCol['Density'].fillna(0,inplace=True)
SAmericaNumCol.rename(columns={'Area':'Aff_Area','Population':'Aff_Population',\
                             'Density':'Aff_Density'},inplace=True)
SAmericaNumCol['T_Area']=dfConTotal[(dfConTotal['Continent']=='South America')].loc[:,'Area']
SAmericaNumCol['T_Area'].fillna(SAmericaNumCol['T_Area'].max(),inplace=True)
SAmericaNumCol['T_Population']=dfConTotal[(dfConTotal['Continent']=='South America')]\
                              .loc[:,'Population']
SAmericaNumCol['T_Population'].fillna(SAmericaNumCol['T_Population'].max(),inplace=True)
SAmericaNumCol['N_Population']=SAmericaNumCol['T_Population']-SAmericaNumCol['Aff_Population']
SAmerica=SAmericaNo.merge(SAmericaNumCol)
#grapgh_No_Country_South America
ax48=SAmerica[['Month','Affected','Not Affected']].plot(kind='bar',color=color_set7,\
                                                      width=.8,fontsize=10,stacked=True,ylim=(0,20))
ax48.legend(('Affected','Not Affected'),prop=font)
ax48.set_ylabel('Number of South American Countries/Territories',fontsize='medium',fontweight='bold')
ax48.set_xlabel('')
ax48.set_xticklabels(SAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels48=list(ax48.get_yticks())
ax48.set_yticklabels(ylabels48,fontdict=fontdict)
hght48=[x.get_height() for x in ax48.patches]
wdth48=[x.get_width() for x in ax48.patches]
for npatch in range(len(SAmerica)):
    value1=SAmerica.iloc[npatch,1]
    value2=SAmerica.iloc[npatch,2]
    textvalueR=value1/value2*100
    textvalue=round(textvalueR)
    textvalue=int(textvalueR)
    textvalue=str(textvalue)+'%' 
    widths=-.175+1*npatch
    if textvalueR>0:
       txt48=textvalue
       plt.annotate(txt48, xy=(widths,hght48[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Total Cases 
ax49=SAmerica[['Month','Total Cases']].plot(kind='bar',color='indigo',\
                                         width=.8,fontsize=10,stacked=True)
ax49.get_legend().remove()
ax49.set_ylabel('Number of Cases in South America',fontsize='medium',fontweight='bold')
ax49.set_xlabel('')
ax49.set_xticklabels(SAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels49=list(ax49.get_yticks())
xlabels49=list(ax49.get_xticks())
ylabels49=[int(x) for x in ylabels49]
ax49.set_yticklabels(ylabels49,fontdict=fontdict)
hght49=[x.get_height() for x in ax49.patches]
wdth49=[x.get_width() for x in ax49.patches]
for npatch in range(len(SAmerica)):
    textvalueR=SAmerica.iloc[npatch,4]
    widths=-.1+1*npatch
    if textvalueR>ylabels49[1]*.3:
       plt.annotate(f"{int(SAmerica.loc[npatch,'Total Cases']):,}", \
                    xy=(xlabels49[npatch]+-.25,hght49[npatch]+.2),rotation=0, weight='bold',\
                    fontsize=8,color='black')
plt.show()
#graph_Case Total Deaths 
ax50=SAmerica[['Month','Total Deaths']].plot(kind='bar',color='indigo',\
                                         width=.8,fontsize=10,stacked=True)
ax50.get_legend().remove()
ax50.set_ylabel('Number of Deaths in South America',fontsize='medium',fontweight='bold')
ax50.set_xlabel('')
ax50.set_xticklabels(SAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels50=list(ax50.get_yticks())
ylabels50=[int(x) for x in ylabels50]
ax50.set_yticklabels(ylabels50,fontdict=fontdict)
hght50=[x.get_height() for x in ax50.patches]
wdth50=[x.get_width() for x in ax50.patches]
for npatch in range(len(SAmerica)):
    textvalueR=SAmerica.iloc[npatch,5]
    widths=-.25+1*npatch
    if textvalueR>0:
       plt.annotate(f"{int(SAmerica.loc[npatch,'Total Deaths']):,}", \
                    xy=(widths,hght50[npatch]+.2),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph_Case Fatality Rate
SAmerica['CFR']=SAmerica['Total Deaths']/SAmerica['Total Cases']*100
SAmerica['CFR'].fillna(0,inplace=True)
ax51=SAmerica[['Month','CFR']].plot(kind='bar',color='indigo',\
                                         width=.8,fontsize=10,stacked=True)
ax51.get_legend().remove()
ax51.set_ylabel('Case Fatality Rate in South America %',fontsize='medium',fontweight='bold')
ax51.set_xlabel('')
ax51.set_xticklabels(SAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels51=list(ax51.get_yticks())
ax51.set_yticklabels(ylabels51,fontdict=fontdict)
hght51=[x.get_height() for x in ax51.patches]
wdth51=[x.get_width() for x in ax51.patches]
for npatch in range(len(SAmerica)):
    textvalueR=round(SAmerica.loc[npatch,'CFR'],2)
    widths=-.175+1*npatch
    if textvalueR>0:
       txt51=str(textvalueR)+'%'
       plt.annotate(txt51, xy=(widths,hght51[npatch]+.02),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph Total Cases per Affected population in 100,000
SAmerica['CPP']=SAmerica['Total Cases']/SAmerica['Aff_Population']*1e5
ax52=SAmerica[['Month','CPP']].plot(kind='bar',color='indigo',width=.8,fontsize=10)
ax52.get_legend().remove()
ax52.set_ylabel('Cases per population in 100,000 in South America',fontsize='medium',fontweight='bold')
ax52.set_xlabel('')
ax52.set_xticklabels(SAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels52=list(ax48.get_yticks())
ax52.set_yticklabels(ylabels52,fontdict=fontdict)
hght52=[x.get_height() for x in ax52.patches]
wdth52=[x.get_width() for x in ax52.patches]
for npatch in range(len(SAmerica)):
    textvalueR=round(SAmerica.loc[npatch,'CPP'],2)
    widths=-.175+1*npatch
    if textvalueR>ylabels48[1]*.5:
       txt52=str(textvalueR)
       plt.annotate(txt52, xy=(widths,hght52[npatch]+.002),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
#graph Total Deaths per Affected population in 100,000
SAmerica['DPP']=SAmerica['Total Deaths']/SAmerica['Aff_Population']*1e5
ax53=SAmerica[['Month','DPP']].plot(kind='bar',color='indigo',width=.8,fontsize=10)
ax53.get_legend().remove()
ax53.set_ylabel('Cases per population in 100,000 in South America',fontsize='medium',fontweight='bold')
ax53.set_xlabel('')
ax53.set_xticklabels(SAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels53=list(ax53.get_yticks())
ax53.set_yticklabels(ylabels53,fontdict=fontdict)
hght53=[x.get_height() for x in ax53.patches]
wdth53=[x.get_width() for x in ax53.patches]
for npatch in range(len(SAmerica)):
    textvalueR=round(SAmerica.loc[npatch,'DPP'],2)
    widths=-.175+1*npatch
    if textvalueR>0:
       txt53=str(textvalueR)
       plt.annotate(txt53, xy=(widths,hght53[npatch]+.002),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()
# Affected population
ax54=SAmerica[['Month','Aff_Population','N_Population']].plot(kind='bar',color=color_set7\
                           ,width=.8,ylim=(0,5e8),fontsize=10,stacked=True)
ax54.legend(('Affected','Not Affected'),prop=font)
ax54.set_ylabel('South American Population',fontsize='medium',fontweight='bold')
ax54.set_xlabel('')
ax54.set_xticklabels(SAmerica['Month'],rotation=0,fontdict=fontdict)
ylabels54=list(ax54.get_yticks())
ylabels54=[int(x) for x in ylabels54]
ax54.set_yticklabels(ylabels54,fontdict=fontdict)
hght54=[x.get_height() for x in ax54.patches]
wdth54=[x.get_width() for x in ax54.patches]
for npatch in range(len(SAmerica)):
    value1=SAmerica.iloc[npatch,7]
    value2=SAmerica.iloc[npatch,10]
    textvalueR=value1/value2*100
    textvalue=round(textvalueR)
    textvalue=int(textvalueR)
    textvalue=str(textvalue)+'%' 
    widths=-.175+1*npatch
    if textvalueR>0:
       txt54=textvalue
       plt.annotate(txt54, xy=(widths,hght54[npatch]*.9),rotation=0, weight='bold', fontsize=8,color='black')
plt.show()

#plt.annotate('',xy=(1.5,202),xytext=(1,56),arrowprops={'arrowstyle':'->'})
#plt.annotate('272% increase',xy=(1,125),fontsize='medium',fontweight='bold')
        


