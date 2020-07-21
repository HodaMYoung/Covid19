""" The purpose of this code is to compare th Coronavirus outbreak situation among G7 countries (Canada,
France,Germany, Italy, Japan, the UK and the USA) by visualizing and mathematical modelling. Additionally,
the impact of imposed quarantine in each country is investigated in this snippet.The procedure can
be summarized as follows:
1- Extracting data from WHO reports.
2- Calculating basic epidemiological variables such as total cases and deaths per population in 100,000
as well as case fatality rate.
3- Using Matplotlib.Pyplot to plot the pie and line charts for visualization.
4- Mathematical modelings are divided into two major stages:
   i) Modeling before imposed quarantine
   ii) Modeling after imposed quarantine
5- Total Cases are modeled as a function of time using Numpy polyfit function with Sklearn
train_test_split function for stage ii, respectively.
6- Total Deaths are modeled as a function of time and total cases using SkLearn Pipeline function with
PolynomialFeature model for stage i), and GridSearchCV function for stage ii).
7- Model validation using coefficient of determination and mean squared error

More information about the first 2 steps was discussed in:
https://github.com/HodaMYoung/Covid19

Source:
https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports"""

# importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager 
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,normalize
from sklearn.metrics import  mean_squared_error,r2_score
from sklearn.pipeline import  Pipeline
from matplotlib.patches import Ellipse
from sklearn.model_selection import train_test_split, GridSearchCV
dfCon_n=pd.read_csv('RevisedWikiCountries.csv')
dfCon_n.drop(columns=['Unnamed: 0'],inplace=True)
df=pd.read_csv('WHOrJuly19.csv')
df.drop(columns=['Unnamed: 0'],inplace=True)
g7=['Canada','France','Germany','Italy','Japan','United Kingdom',\
    'United States of America']
df0=df[df['Updated']==df['Updated'].max()]
dfg7=df0[df0['Country'].isin(g7)]
Numcol=['Total Cases','Total Deaths','New Cases','New Deaths','Population']
df1=pd.DataFrame(dfg7[Numcol].sum()).transpose()
dfNg7=df0[~df0['Country'].isin(g7)]
df2=pd.DataFrame(dfNg7[Numcol].sum()).transpose()
df1['Country']='G7 Countries'
df2['Country']='Rest of the World'
dfn=pd.concat([df1,df2],axis=0)
dfn.reset_index(drop=True,inplace=True)
dfn['CPP']=dfn['Total Cases']/dfn['Population']*1e5
dfn['DPP']=dfn['Total Deaths']/dfn['Population']*1e5
dfn['CFR']=dfn['Total Deaths']/dfn['Total Cases']*100
# Color set & fonts
font = font_manager.FontProperties(weight='bold', size=10)
fontdict= {'weight': 'bold',
        'size': 10,
        }
font1 = font_manager.FontProperties(weight='bold', size=8)
fontdict1= {'weight': 'bold',
        'size': 8,
        }
#G7 Countries vs. rest of the world
#population
ax0=dfn.plot(kind='pie',y='Population',labels=dfn['Country'].unique(),colors=['red','blue'],\
               fontsize=10,autopct='%1.0f%%',shadow=False,textprops=\
               {'fontsize': 10,'weight':'bold'},startangle= 45)
ax0.set_ylabel('')
ax0.get_legend().remove()
ax0.set_title('Population',fontdict={'fontsize':10,'fontweight':'bold',\
                                     'verticalalignment':'top','horizontalalignment':'center'})
plt.show()
#Total Cases
ax1=dfn.plot(kind='pie',y='Total Cases',labels=dfn['Country'].unique(),colors=['red','blue'],\
               fontsize=10,autopct='%1.0f%%',shadow=False,textprops=\
               {'fontsize': 10,'weight':'bold'},startangle= 45)
ax1.set_ylabel('')
ax1.get_legend().remove()
ax1.set_title('Total Cases',fontdict={'fontsize':10,'fontweight':'bold',\
                                     'verticalalignment':'top','horizontalalignment':'center'})
plt.show()

#Total Deaths

ax2=dfn.plot(kind='pie',y='Total Deaths',labels=dfn['Country'].unique(),colors=['red','blue'],\
               fontsize=10,autopct='%1.0f%%',shadow=False,textprops=\
               {'fontsize': 10,'weight':'bold'},startangle= 45)
ax2.set_ylabel('')
ax2.get_legend().remove()
ax2.set_title('Total Deaths',fontdict={'fontsize':10,'fontweight':'bold',\
                                     'verticalalignment':'top','horizontalalignment':'center'})
plt.show()

#New Cases
ax3=dfn.plot(kind='pie',y='New Cases',labels=dfn['Country'].unique(),colors=['red','blue'],\
               fontsize=10,autopct='%1.0f%%',shadow=False,textprops=\
               {'fontsize': 10,'weight':'bold'},startangle= 45)
ax3.set_ylabel('')
ax3.get_legend().remove()
ax3.set_title('New Cases',fontdict={'fontsize':10,'fontweight':'bold',\
                                     'verticalalignment':'top','horizontalalignment':'center'})
plt.show()

#New Deaths
ax4=dfn.plot(kind='pie',y='New Deaths',labels=dfn['Country'].unique(),colors=['red','blue'],\
               fontsize=10,autopct='%1.0f%%',shadow=False,textprops=\
               {'fontsize': 10,'weight':'bold'},startangle= 45)
ax4.set_ylabel('')
ax4.get_legend().remove()
ax4.set_title('New Deaths',fontdict={'fontsize':10,'fontweight':'bold',\
                                     'verticalalignment':'top','horizontalalignment':'center'})
plt.show()
#CPP
ax5=dfn.plot(kind='bar',x='Country',y='CPP',color=['red','blue'])
ax5.get_legend().remove()
ax5.set_xlabel('')
ax5.set_ylabel('Number of Cases per Population in 100,000',\
               fontsize='medium',fontweight='bold')
ax5.set_xticklabels(['G7 Countries','Rest of the World'],rotation=0,fontdict=fontdict)
ylabels5=list(ax5.get_yticks())
ylabels5=[int(x) for x in ylabels5]
xlabels5=list(ax5.get_xticks())
ax5.set_yticklabels(ylabels5,fontdict=fontdict)
hght5=[x.get_height() for x in ax5.patches]
for npatch in range(0,2):
    txt5=str(round(dfn.loc[npatch,'CPP'],2))
    plt.annotate(txt5,xy=(xlabels5[npatch]-.125,hght5[npatch]+2),rotation=0,\
                          weight='bold',fontsize=10,color='black')

plt.show()

#DPP
ax6=dfn.plot(kind='bar',x='Country',y='DPP',color=['red','blue'])
ax6.get_legend().remove()
ax6.set_xlabel('')
ax6.set_ylabel('Number of Deaths per Population in 100,000',\
               fontsize='medium',fontweight='bold')
ax6.set_xticklabels(['G7 Countries','Rest of the World'],rotation=0,fontdict=fontdict)
ylabels6=list(ax6.get_yticks())
ylabels6=[int(x) for x in ylabels6]
xlabels6=list(ax6.get_xticks())
ax6.set_yticklabels(ylabels6,fontdict=fontdict)
hght6=[x.get_height() for x in ax6.patches]
for npatch in range(0,2):
    txt6=str(round(dfn.loc[npatch,'DPP'],2))
    plt.annotate(txt6,xy=(xlabels6[npatch]-.075,hght6[npatch]+.2),rotation=0,\
                          weight='bold',fontsize=10,color='black')

plt.show()

#CFR
ax7=dfn.plot(kind='bar',x='Country',y='CFR',color=['red','blue'])
ax7.get_legend().remove()
ax7.set_xlabel('')
ax7.set_ylabel('Case Fatality Rate %',fontsize='medium',fontweight='bold')
ax7.set_xticklabels(['G7 Countries','Rest of the World'],rotation=0,\
                    fontdict=fontdict)
ylabels7=list(ax7.get_yticks())
ylabels7=[int(x) for x in ylabels7]
xlabels7=list(ax7.get_xticks())
ax7.set_yticklabels(ylabels7,fontdict=fontdict)
hght7=[x.get_height() for x in ax7.patches]
for npatch in range(0,2):
    txt7=str(round(dfn.loc[npatch,'CFR'],2))
    plt.annotate(txt7,xy=(xlabels7[npatch]-.075,hght7[npatch]+.02),rotation=0,\
                          weight='bold',fontsize=10,color='black')
plt.show()

#g7
Canada=df[df['Country']=='Canada']
Canada.reset_index(drop=True,inplace=True)
France=df[df['Country']=='France']
France.reset_index(drop=True,inplace=True)
Germany=df[df['Country']=='Germany']
Germany.reset_index(drop=True,inplace=True)
Italy=df[df['Country']=='Italy']
Italy.reset_index(drop=True,inplace=True)
Japan=df[df['Country']=='Japan']
Japan.reset_index(drop=True,inplace=True)
UK=df[df['Country']=='United Kingdom']
UK.reset_index(drop=True,inplace=True)
USA=df[df['Country']=='United States of America']
USA.reset_index(drop=True,inplace=True)
Canada['CPP']=Canada['Total Cases']/Canada['Population']*1e5
Canada['DPP']=Canada['Total Deaths']/Canada['Population']*1e5
Canada['CFR']=Canada['Total Deaths']/Canada['Total Cases']*100
France['CPP']=France['Total Cases']/France['Population']*1e5
France['DPP']=France['Total Deaths']/France['Population']*1e5
France['CFR']=France['Total Deaths']/France['Total Cases']*100
Germany['CPP']=Germany['Total Cases']/Germany['Population']*1e5
Germany['DPP']=Germany['Total Deaths']/Germany['Population']*1e5
Germany['CFR']=Germany['Total Deaths']/Germany['Total Cases']*100
Italy['CPP']=Italy['Total Cases']/Italy['Population']*1e5
Italy['DPP']=Italy['Total Deaths']/Italy['Population']*1e5
Italy['CFR']=Italy['Total Deaths']/Italy['Total Cases']*100
Japan['CPP']=Japan['Total Cases']/Japan['Population']*1e5
Japan['DPP']=Japan['Total Deaths']/Japan['Population']*1e5
Japan['CFR']=Japan['Total Deaths']/Japan['Total Cases']*100
UK['CPP']=UK['Total Cases']/UK['Population']*1e5
UK['DPP']=UK['Total Deaths']/UK['Population']*1e5
UK['CFR']=UK['Total Deaths']/UK['Total Cases']*100
USA['CPP']=USA['Total Cases']/USA['Population']*1e5
USA['DPP']=USA['Total Deaths']/USA['Population']*1e5
USA['CFR']=USA['Total Deaths']/USA['Total Cases']*100

#Total Cases
fig = plt.figure()
fig.subplots_adjust(wspace=.3)
ax8 = fig.add_subplot(121)
Canada.plot(kind='line',x='Updated',y='Total Cases', color='red',ax=ax8)
France.plot(kind='line',x='Updated',y='Total Cases',ax=ax8,color='blue')
Germany.plot(kind='line',x='Updated',y='Total Cases',ax=ax8,color='yellow')
Italy.plot(kind='line',x='Updated',y='Total Cases', ax=ax8,color='green')
Japan.plot(kind='line',x='Updated',y='Total Cases', ax=ax8,color='maroon',\
           linestyle=':')
UK.plot(kind='line',x='Updated',y='Total Cases',ax=ax8,color='black',\
           linestyle='-.')
USA.plot(kind='line',x='Updated',y='Total Cases',ax=ax8,color='navy',\
           linestyle='--')
ax8.legend(('Canada','France','Germany','Italy','Japan','UK','USA'),prop=font1)             
ax8.set_ylabel('Total Cases',fontsize='small',fontweight='bold')
ax8.set_xlabel('Days',fontsize='small',fontweight='bold')
ylabels8=list(ax8.get_yticks())
ylabels8=[int(x) for x in ylabels8]
xlabels8=list(ax8.get_xticks())
xlabels8=[int(x) for x in xlabels8]
ax8.set_xticklabels(xlabels8,fontdict=fontdict1)
ax8.set_yticklabels(ylabels8,rotation=0,fontdict=fontdict1)
#Total Deaths
ax9 = fig.add_subplot(122)
Canada.plot(kind='line',x='Updated',y='Total Deaths', color='red',ax=ax9)
France.plot(kind='line',x='Updated',y='Total Deaths',ax=ax9,color='blue')
Germany.plot(kind='line',x='Updated',y='Total Deaths', ax=ax9,color='yellow')
Italy.plot(kind='line',x='Updated',y='Total Deaths', ax=ax9,color='green')
Japan.plot(kind='line',x='Updated',y='Total Deaths', ax=ax9,color='maroon',\
           linestyle=':')
UK.plot(kind='line',x='Updated',y='Total Deaths',ax=ax9,color='black',\
           linestyle='-.')
USA.plot(kind='line',x='Updated',y='Total Deaths',ax=ax9,color='navy',\
           linestyle='--')
ax9.legend(('Canada','France','Germany','Italy','Japan','UK','USA'),prop=font1)             
ax9.set_ylabel('Total Deaths',fontsize='small',fontweight='bold')
ax9.set_xlabel('Days',fontsize='small',fontweight='bold')
ylabels9=list(ax9.get_yticks())
ylabels9=[int(x) for x in ylabels9]
xlabels9=list(ax9.get_xticks())
xlabels9=[int(x) for x in xlabels9]
ax9.set_xticklabels(xlabels9,fontdict=fontdict1)
ax9.set_yticklabels(ylabels9,rotation=0,fontdict=fontdict1) 
plt.show()
#Exclude US, G7
#Total Cases
fig = plt.figure()
fig.subplots_adjust(wspace=.3)
ax10 = fig.add_subplot(121)
Canada.plot(kind='line',x='Updated',y='Total Cases', color='red',ax=ax10)
France.plot(kind='line',x='Updated',y='Total Cases',ax=ax10,color='blue')
Germany.plot(kind='line',x='Updated',y='Total Cases',ax=ax10,color='yellow')
Italy.plot(kind='line',x='Updated',y='Total Cases', ax=ax10,color='green')
Japan.plot(kind='line',x='Updated',y='Total Cases', ax=ax10,color='maroon',\
           linestyle=':')
UK.plot(kind='line',x='Updated',y='Total Cases',ax=ax10,color='black',\
           linestyle='-.')
ax10.legend(('Canada','France','Germany','Italy','Japan','UK'),prop=font1)             
ax10.set_ylabel('Total Cases',fontsize='small',fontweight='bold')
ax10.set_xlabel('Days',fontsize='small',fontweight='bold')
ylabels10=list(ax10.get_yticks())
ylabels10=[int(x) for x in ylabels10]
xlabels10=list(ax8.get_xticks())
xlabels10=[int(x) for x in xlabels10]
ax10.set_xticklabels(xlabels10,fontdict=fontdict1)
ax10.set_yticklabels(ylabels10,rotation=0,fontdict=fontdict1)
#Total Deaths
ax11 = fig.add_subplot(122)
Canada.plot(kind='line',x='Updated',y='Total Deaths', color='red',ax=ax11)
France.plot(kind='line',x='Updated',y='Total Deaths',ax=ax11,color='blue')
Germany.plot(kind='line',x='Updated',y='Total Deaths', ax=ax11,color='yellow')
Italy.plot(kind='line',x='Updated',y='Total Deaths', ax=ax11,color='green')
Japan.plot(kind='line',x='Updated',y='Total Deaths', ax=ax11,color='maroon',\
           linestyle=':')
UK.plot(kind='line',x='Updated',y='Total Deaths',ax=ax11,color='black',\
           linestyle='-.')
ax11.legend(('Canada','France','Germany','Italy','Japan','UK'),prop=font1)             
ax11.set_ylabel('Total Deaths',fontsize='small',fontweight='bold')
ax11.set_xlabel('Days',fontsize='small',fontweight='bold')
ylabels11=list(ax11.get_yticks())
ylabels11=[int(x) for x in ylabels11]
xlabels11=list(ax11.get_xticks())
xlabels11=[int(x) for x in xlabels11]
ax11.set_xticklabels(xlabels11,fontdict=fontdict1)
ax11.set_yticklabels(ylabels11,rotation=0,fontdict=fontdict1) 
plt.show()
#CPP
fig = plt.figure()
fig.subplots_adjust(wspace=.3)
ax12 = fig.add_subplot(121)
Canada.plot(kind='line',x='Updated',y='CPP', color='red',ax=ax12)
France.plot(kind='line',x='Updated',y='CPP',ax=ax12,color='blue')
Germany.plot(kind='line',x='Updated',y='CPP',ax=ax12,color='yellow')
Italy.plot(kind='line',x='Updated',y='CPP', ax=ax12,color='green')
Japan.plot(kind='line',x='Updated',y='CPP', ax=ax12,color='maroon',\
           linestyle=':')
UK.plot(kind='line',x='Updated',y='CPP',ax=ax12,color='black',\
           linestyle='-.')
USA.plot(kind='line',x='Updated',y='CPP',ax=ax12,color='navy',\
           linestyle='--')
ax12.legend(('Canada','France','Germany','Italy','Japan','UK','USA'),prop=font1)             
ax12.set_ylabel('Cases per Population in 100,000',fontsize='small',fontweight='bold')
ax12.set_xlabel('Days',fontsize='small',fontweight='bold')
ylabels12=list(ax12.get_yticks())
ylabels12=[int(x) for x in ylabels12]
xlabels12=list(ax12.get_xticks())
xlabels12=[int(x) for x in xlabels12]
ax12.set_xticklabels(xlabels12,fontdict=fontdict1)
ax12.set_yticklabels(ylabels12,fontdict=fontdict1)
#DPP
ax13 = fig.add_subplot(122)
Canada.plot(kind='line',x='Updated',y='DPP', color='red',ax=ax13)
France.plot(kind='line',x='Updated',y='DPP',ax=ax13,color='blue')
Germany.plot(kind='line',x='Updated',y='DPP', ax=ax13,color='yellow')
Italy.plot(kind='line',x='Updated',y='DPP', ax=ax13,color='green')
Japan.plot(kind='line',x='Updated',y='DPP', ax=ax13,color='maroon',\
           linestyle=':')
UK.plot(kind='line',x='Updated',y='DPP',ax=ax13,color='black',\
           linestyle='-.')
USA.plot(kind='line',x='Updated',y='DPP',ax=ax13,color='navy',\
           linestyle='--')
ax13.legend(('Canada','France','Germany','Italy','Japan','UK','USA'),prop=font1)             
ax13.set_ylabel('Deaths per Population in 100,000',fontsize='small',fontweight='bold')
ax13.set_xlabel('Days',fontsize='small',fontweight='bold')
ylabels13=list(ax13.get_yticks())
ylabels13=[int(x) for x in ylabels13]
xlabels13=list(ax13.get_xticks())
xlabels13=[int(x) for x in xlabels13]
ax13.set_xticklabels(xlabels13,fontdict=fontdict1)
ax13.set_yticklabels(ylabels13,fontdict=fontdict1) 
plt.show()
#CFR
ax14=Canada.plot(kind='line',x='Updated',y='CFR', color='red')
France.plot(kind='line',x='Updated',y='CFR',ax=ax14,color='blue')
Germany.plot(kind='line',x='Updated',y='CFR', ax=ax14,color='yellow')
Italy.plot(kind='line',x='Updated',y='CFR', ax=ax14,color='green')
Japan.plot(kind='line',x='Updated',y='CFR', ax=ax14,color='maroon',\
           linestyle=':')
UK.plot(kind='line',x='Updated',y='CFR',ax=ax14,color='black',\
           linestyle='-.')
USA.plot(kind='line',x='Updated',y='CFR',ax=ax14,color='navy',\
           linestyle='--')
ax14.legend(('Canada','France','Germany','Italy','Japan','UK','USA'),prop=font)             
ax14.set_ylabel('Case Fatality Rate%',fontsize='medium',fontweight='bold')
ax14.set_xlabel('Days',fontsize='medium',fontweight='bold')
ylabels14=list(ax14.get_yticks())
ylabels14=[int(x) for x in ylabels14]
xlabels14=list(ax14.get_xticks())
xlabels14=[int(x) for x in xlabels14]
ax14.set_xticklabels(xlabels14,fontdict=fontdict)
ax14.set_yticklabels(ylabels14,fontdict=fontdict) 
#Pie G7
dfg7['Country'].replace(['United States of America','United Kingdom'],['USA','UK'],\
                        regex=True,inplace=True)
color_set1=['pink','navy','blue','red','yellow','green','grey']
ax0=dfg7.plot(kind='pie',y='Population',labels=dfg7['Country'].unique(),colors=color_set1,\
               fontsize=10,autopct='%1.0f%%',shadow=False,textprops=\
               {'fontsize': 10,'weight':'bold'},startangle= 45)
ax0.set_ylabel('')
ax0.get_legend().remove()
ax0.set_title('Population',fontdict={'fontsize':10,'fontweight':'bold',\
                                     'verticalalignment':'top','horizontalalignment':'center'})
plt.show()
#Total Cases
ax1=dfg7.plot(kind='pie',y='Total Cases',labels=dfg7['Country'].unique(),colors=color_set1,\
              fontsize=10,autopct='%1.0f%%',shadow=True,explode=(0.5,0,0.3,0.3,0.3,0.3,0),\
              textprops={'fontsize': 8,'weight':'bold'},startangle= 45)
ax1.set_ylabel('')
ax1.get_legend().remove()
ax1.set_title('Total Cases',fontdict={'fontsize':8,'fontweight':'bold',\
                                     'verticalalignment':'top','horizontalalignment':'center'})
plt.show()

#Total Deaths

ax2=dfg7.plot(kind='pie',y='Total Deaths',labels=dfg7['Country'].unique(),colors=color_set1,\
               fontsize=10,autopct='%1.1f%%',shadow=True,textprops=\
               {'fontsize': 8,'weight':'bold'},explode=(0.5,0,0,0.6,0.6,0,0),startangle= 0)
ax2.set_ylabel('')
ax2.get_legend().remove()
ax2.set_title('Total Deaths',fontdict={'fontsize':8,'fontweight':'bold',\
                                     'verticalalignment':'top','horizontalalignment':'center'})
plt.show()


####
lockdowns=pd.DataFrame({'Country':['Canada','France','Germany','Italy','Japan','UK','USA'],\
			    'Start':['30-03-2020','17-03-2020','23-03-2020','09-03-2020',\
				     '16-04-2020','23-03-2020','22-03-2020'],\
			    'End':['19-05-2020','11-05-2020','10-05-2020','04-05-2020',\
				   '25-05-2020','01-06-2020','11-05-2020']})

QS_Canada=Canada[Canada['Report Date']==\
                 lockdowns[lockdowns['Country']=='Canada'].iloc[0,1]].index.to_list()[0]
QE_Canada=Canada[Canada['Report Date']==\
                 lockdowns[lockdowns['Country']=='Canada'].iloc[0,2]].index.to_list()[0]
QS_France=France[France['Report Date']==\
                 lockdowns[lockdowns['Country']=='France'].iloc[0,1]].index.to_list()[0]
QE_France=France[France['Report Date']==\
                 lockdowns[lockdowns['Country']=='France'].iloc[0,2]].index.to_list()[0]
QS_Germany=Germany[Germany['Report Date']==\
                 lockdowns[lockdowns['Country']=='Germany'].iloc[0,1]].index.to_list()[0]
QE_Germany=Germany[Germany['Report Date']==\
                 lockdowns[lockdowns['Country']=='Germany'].iloc[0,2]].index.to_list()[0]
QS_Italy=Italy[Italy['Report Date']==\
                 lockdowns[lockdowns['Country']=='Italy'].iloc[0,1]].index.to_list()[0]
QE_Italy=Italy[Italy['Report Date']==\
                 lockdowns[lockdowns['Country']=='Italy'].iloc[0,2]].index.to_list()[0]
QS_Japan=Japan[Japan['Report Date']==\
                 lockdowns[lockdowns['Country']=='Japan'].iloc[0,1]].index.to_list()[0]
QE_Japan=Japan[Japan['Report Date']==\
                 lockdowns[lockdowns['Country']=='Japan'].iloc[0,2]].index.to_list()[0]
QS_UK=UK[UK['Report Date']==\
                 lockdowns[lockdowns['Country']=='UK'].iloc[0,1]].index.to_list()[0]
QE_UK=UK[UK['Report Date']==\
                 lockdowns[lockdowns['Country']=='UK'].iloc[0,2]].index.to_list()[0]
QS_USA=USA[USA['Report Date']==\
                 lockdowns[lockdowns['Country']=='USA'].iloc[0,1]].index.to_list()[0]
QE_USA=USA[USA['Report Date']==\
                 lockdowns[lockdowns['Country']=='USA'].iloc[0,2]].index.to_list()[0]
#Canada
#Total Cases
CanadaI=Canada.iloc[:QS_Canada,:]
x,y=CanadaI['Updated'],CanadaI['Total Cases']
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
        f=np.polyfit(x,y,n)
        p=np.poly1d(f)
        yhat=p(x)
        r2.append(r2_score(y, yhat))
        mse.append(mean_squared_error(y,yhat))
        degree.append(n)
        if n>1:
           pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
           pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
        else:
            pr2.append(0)
            pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
axC_f1=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=axC_f1,color='black',marker='*')
axC_f1.legend(('R2','MSE'),prop=font)
axC_f1.set_xlabel('Days',fontsize='medium',weight='bold')
axC_f1.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelCf1=list(axC_f1.get_xticks())
xlabelCf1=[int(x) for x in xlabelCf1]
axC_f1.set_xticklabels(xlabelCf1,fontdict=fontdict)
ylabelCf1=list(axC_f1.get_yticks())
ylabelCf1=[round(x,2) for x in ylabelCf1]
axC_f1.set_yticklabels(ylabelCf1,fontdict=fontdict)
ellipse=Ellipse(xy=(4.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(4.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
axC_f1.add_patch(ellipse)
plt.show()
#n=5, order polynomial
f=np.polyfit(x,y,5)
p=np.poly1d(f)
xp=Canada['Updated']
yp=p(xp)
#xp=np.array(xp)
combined = np.vstack((xp, yp)).T
CanadaP=pd.DataFrame(combined,columns=['Updated','Total Cases'])
###
axC1=Canada.plot(kind='line',x='Updated',y='Total Cases',color='red',ylim=(0,Canada['Total Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axC1.fill_between(Canada.iloc[QS_Canada:QE_Canada,17],Canada.iloc[QS_Canada:QE_Canada,10],facecolor='pink',\
                  edgecolor='red')
axC1.set_xlabel('Days',fontsize='medium',weight='bold')
axC1.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelaxC1=list(axC1.get_xticks())
xlabelaxC1=[int(x) for x in xlabelaxC1]
axC1.set_xticklabels(xlabelaxC1,fontdict=fontdict)
ylabelaxC1=list(axC1.get_yticks())
ylabelaxC1=[int(x) for x in ylabelaxC1]
axC1.set_yticklabels(ylabelaxC1,fontdict=fontdict)
CanadaP.plot(kind='line',x='Updated',y='Total Cases',color='black',fontsize=10,ax=axC1,marker='*')
axC1.legend(('Actual','Predicted'),prop=font, loc='upper left')
axC1.fill_between(Canada.iloc[QS_Canada:,17],Canada.iloc[QS_Canada:,10],CanadaP.iloc[QS_Canada:,1],\
                  hatch='x',facecolor='white',edgecolor='black')
xyt=(Canada.iloc[QS_Canada,17]+16,Canada.iloc[QE_Canada,10]/4)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='red',fontsize='large',\
             weight='bold')
plt.show()
#Total Deaths
Z,Y=CanadaI[['Updated','Total Cases']],CanadaI[['Total Deaths']]
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
    Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=n)),('model',LinearRegression())]
    pipe=Pipeline(Input)
    pipe.fit(Z,Y)
    ypipe=pipe.predict(Z)
    r2.append(r2_score(ypipe,Y))
    mse.append(mean_squared_error(ypipe,Y))
    degree.append(n)
    if n>1:
       pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
       pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
    else:
        pr2.append(0)
        pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
AxC_f2=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=AxC_f2,color='black',marker='*')
AxC_f2.legend(('R2','MSE'),prop=font)
AxC_f2.set_xlabel('Days',fontsize='medium',weight='bold')
AxC_f2.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelFf2=list(AxC_f2.get_xticks())
xlabelFf2=[int(x) for x in xlabelFf2]
AxC_f2.set_xticklabels(xlabelFf2,fontdict=fontdict)
ylabelFf2=list(AxC_f2.get_yticks())
ylabelFf2=[round(x,2) for x in ylabelFf2]
AxC_f2.set_yticklabels(ylabelFf2,fontdict=fontdict)
ellipse=Ellipse(xy=(3.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(3.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
AxC_f2.add_patch(ellipse)
plt.show()
#n=3, order polynomial
ZP=Canada[['Updated','Total Cases']]
Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=3)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,Y)
YPn=pipe.predict(ZP)
CanadaP=pd.DataFrame(YPn)
CanadaP.columns=['Total Deaths']
CanadaP['Updated']=Canada['Updated']
AxC2=Canada.plot(kind='line',x='Updated',y='Total Deaths',color='red',ylim=(0,Canada['Total Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
AxC2.fill_between(Canada.iloc[QS_Canada:QE_Canada,17],Canada.iloc[QS_Canada:QE_Canada,11],facecolor='pink',\
		  edgecolor='red')
AxC2.set_xlabel('Days',fontsize='medium',weight='bold')
AxC2.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxC2=list(AxC2.get_xticks())
xlabelAxC2=[int(x) for x in xlabelAxC2]
AxC2.set_xticklabels(xlabelAxC2,fontdict=fontdict)
ylabelAxC2=list(AxC2.get_yticks())
ylabelAxC2=[int(x) for x in ylabelAxC2]
AxC2.set_yticklabels(ylabelAxC2,fontdict=fontdict)
CanadaP.plot(kind='line',x='Updated',y='Total Deaths',color='black',fontsize=10,ax=AxC2,marker='*')
AxC2.legend(('Actual','Predicted'),prop=font,loc='upper left')

AxC2.fill_between(Canada.iloc[QS_Canada:,17],Canada.iloc[QS_Canada:,11],CanadaP.iloc[QS_Canada:,0],\
                  facecolor='white',edgecolor='black', hatch='x')
xyt=(Canada.iloc[QS_Canada,17]+20,Canada.iloc[QE_Canada,11]/4.5)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='red',fontsize='large',\
             weight='bold')
plt.show()
#New Cases
axC3=Canada.plot(kind='line',x='Updated',y='New Cases',color='red',ylim=(0,Canada['New Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axC3.fill_between(Canada.iloc[QS_Canada:QE_Canada,17],Canada.iloc[QS_Canada:QE_Canada,3],facecolor='pink',\
		     hatch='x',edgecolor='red')
axC3.get_legend().remove()
axC3.set_xlabel('Days',fontsize='medium',weight='bold')
axC3.set_ylabel('New Cases',fontsize='medium',weight='bold')
xlabelaxC3=list(axC3.get_xticks())
xlabelaxC3=[int(x) for x in xlabelaxC3]
axC3.set_xticklabels(xlabelaxC3,fontdict=fontdict)
ylabelaxC3=list(axC3.get_yticks())
ylabelaxC3=[int(x) for x in ylabelaxC3]
axC3.set_yticklabels(ylabelaxC3,fontdict=fontdict)
plt.show()
#New Deaths
axC4=Canada.plot(kind='line',x='Updated',y='New Deaths',color='red',ylim=(0,Canada['New Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axC4.fill_between(Canada.iloc[QS_Canada:QE_Canada,17],Canada.iloc[QS_Canada:QE_Canada,4],facecolor='pink',\
		     hatch='x',edgecolor='red')
axC4.get_legend().remove()
axC4.set_xlabel('Days',fontsize='medium',weight='bold')
axC4.set_ylabel('New Deaths',fontsize='medium',weight='bold')
xlabelaxC4=list(axC4.get_xticks())
xlabelaxC4=[int(x) for x in xlabelaxC4]
axC4.set_xticklabels(xlabelaxC4,fontdict=fontdict)
ylabelaxC4=list(axC4.get_yticks())
ylabelaxC4=[int(x) for x in ylabelaxC4]
axC4.set_yticklabels(ylabelaxC4,fontdict=fontdict)
plt.show()
#France
#Total Cases
FranceI=France.iloc[:QS_France,:]
x,y=FranceI['Updated'],FranceI['Total Cases']
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
        f=np.polyfit(x,y,n)
        p=np.poly1d(f)
        yhat=p(x)
        r2.append(r2_score(y, yhat))
        mse.append(mean_squared_error(y,yhat))
        degree.append(n)
        if n>1:
           pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
           pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
        else:
            pr2.append(0)
            pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
axF_f1=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=axF_f1,color='black',marker='*')
axF_f1.legend(('R2','MSE'),prop=font)
axF_f1.set_xlabel('Days',fontsize='medium',weight='bold')
axF_f1.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelFf1=list(axF_f1.get_xticks())
xlabelFf1=[int(x) for x in xlabelFf1]
axF_f1.set_xticklabels(xlabelFf1,fontdict=fontdict)
ylabelFf1=list(axF_f1.get_yticks())
ylabelFf1=[round(x,2) for x in ylabelFf1]
axF_f1.set_yticklabels(ylabelFf1,fontdict=fontdict)
ellipse=Ellipse(xy=(4.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(4.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
axF_f1.add_patch(ellipse)
plt.show()
#n=5, order polynomial
f=np.polyfit(x,y,5)
p=np.poly1d(f)
xp=France['Updated']
yp=p(xp)
combined = np.vstack((xp, yp)).T
FranceP=pd.DataFrame(combined,columns=['Updated','Total Cases'])
###
axF1=France.plot(kind='line',x='Updated',y='Total Cases',color='navy',ylim=(0,France['Total Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axF1.fill_between(France.iloc[QS_France:QE_France,17],France.iloc[QS_France:QE_France,10],facecolor='azure',\
		  edgecolor='navy')
axF1.set_xlabel('Days',fontsize='medium',weight='bold')
axF1.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelaxF1=list(axF1.get_xticks())
xlabelaxF1=[int(x) for x in xlabelaxF1]
axF1.set_xticklabels(xlabelaxF1,fontdict=fontdict)
ylabelaxF1=list(axF1.get_yticks())
ylabelaxF1=[int(x) for x in ylabelaxF1]
axF1.set_yticklabels(ylabelaxF1,fontdict=fontdict)
FranceP.plot(kind='line',x='Updated',y='Total Cases',color='black',fontsize=10,ax=axF1,marker='*')
axF1.legend(('Actual','Predicted'),prop=font,loc='upper left')
axF1.fill_between(France.iloc[QS_France:,17],France.iloc[QS_France:,10],FranceP.iloc[QS_France:,1],\
                  facecolor='white',edgecolor='black', hatch='x')
xyt=(France.iloc[QS_France,17]+16,France.iloc[QE_France,10]/4)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='navy',fontsize='large',\
             weight='bold')
plt.show()
#Total Deaths
Z,Y=FranceI[['Updated','Total Cases']],FranceI[['Total Deaths']]
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
    Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=n)),('model',LinearRegression())]
    pipe=Pipeline(Input)
    pipe.fit(Z,Y)
    ypipe=pipe.predict(Z)
    r2.append(r2_score(ypipe,Y))
    mse.append(mean_squared_error(ypipe,Y))
    degree.append(n)
    if n>1:
       pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
       pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
    else:
        pr2.append(0)
        pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
AxF_f2=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=AxF_f2,color='black',marker='*')
AxF_f2.legend(('R2','MSE'),prop=font)
AxF_f2.set_xlabel('Days',fontsize='medium',weight='bold')
AxF_f2.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelFf2=list(AxF_f2.get_xticks())
xlabelFf2=[int(x) for x in xlabelFf2]
AxF_f2.set_xticklabels(xlabelFf2,fontdict=fontdict)
ylabelFf2=list(AxF_f2.get_yticks())
ylabelFf2=[round(x,2) for x in ylabelFf2]
AxF_f2.set_yticklabels(ylabelFf2,fontdict=fontdict)
ellipse=Ellipse(xy=(3.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(3.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
AxF_f2.add_patch(ellipse)
plt.show()
#n=3, order polynomial
ZP=France[['Updated','Total Cases']]
Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=3)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,Y)
YPn=pipe.predict(ZP)
FranceP=pd.DataFrame(YPn)
FranceP.columns=['Total Deaths']
FranceP['Updated']=France['Updated']
AxF2=France.plot(kind='line',x='Updated',y='Total Deaths',color='navy',ylim=(0,France['Total Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
AxF2.fill_between(France.iloc[QS_France:QE_France,17],France.iloc[QS_France:QE_France,11],facecolor='aqua',\
		  edgecolor='navy')
AxF2.set_xlabel('Days',fontsize='medium',weight='bold')
AxF2.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxF2=list(AxF2.get_xticks())
xlabelAxF2=[int(x) for x in xlabelAxF2]
AxF2.set_xticklabels(xlabelAxF2,fontdict=fontdict)
ylabelAxF2=list(AxF2.get_yticks())
ylabelAxF2=[int(x) for x in ylabelAxF2]
AxF2.set_yticklabels(ylabelAxF2,fontdict=fontdict)
FranceP.plot(kind='line',x='Updated',y='Total Deaths',color='black',fontsize=10,ax=AxF2,marker='*')
AxF2.legend(('Actual','Predicted'),prop=font,loc='upper left')

AxF2.fill_between(France.iloc[QS_France:,17],France.iloc[QS_France:,11],FranceP.iloc[QS_France:,0],\
                  facecolor='white',edgecolor='black', hatch='x')
xyt=(France.iloc[QS_France,17]+20,France.iloc[QE_France,11]/4)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='navy',fontsize='large',\
             weight='bold')
plt.show()
#New Cases
axF3=France.plot(kind='line',x='Updated',y='New Cases',color='navy',ylim=(0,France['New Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axF3.fill_between(France.iloc[QS_France:QE_France,17],France.iloc[QS_France:QE_France,3],facecolor='azure',\
		     hatch='x',edgecolor='navy')
axF3.get_legend().remove()
axF3.set_xlabel('Days',fontsize='medium',weight='bold')
axF3.set_ylabel('New Cases',fontsize='medium',weight='bold')
xlabelaxF3=list(axF3.get_xticks())
xlabelaxF3=[int(x) for x in xlabelaxF3]
axF3.set_xticklabels(xlabelaxF3,fontdict=fontdict)
ylabelaxF3=list(axF3.get_yticks())
ylabelaxF3=[int(x) for x in ylabelaxF3]
axF3.set_yticklabels(ylabelaxF3,fontdict=fontdict)
plt.show()
#New Deaths
axF4=France.plot(kind='line',x='Updated',y='New Deaths',color='navy',ylim=(0,France['New Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axF4.fill_between(France.iloc[QS_France:QE_France,17],France.iloc[QS_France:QE_France,4],facecolor='azure',\
		     hatch='x',edgecolor='navy')
axF4.get_legend().remove()
axF4.set_xlabel('Days',fontsize='medium',weight='bold')
axF4.set_ylabel('New Deaths',fontsize='medium',weight='bold')
xlabelaxF4=list(axF4.get_xticks())
xlabelaxF4=[int(x) for x in xlabelaxF4]
axF4.set_xticklabels(xlabelaxF4,fontdict=fontdict)
ylabelaxF4=list(axF4.get_yticks())
ylabelaxF4=[int(x) for x in ylabelaxF4]
axF4.set_yticklabels(ylabelaxF4,fontdict=fontdict)
plt.show()
#Germany
#Total Cases
GermanyI=Germany.iloc[:QS_Germany,:]
x,y,z=GermanyI['Updated'],GermanyI['Total Cases'],GermanyI['Total Deaths']
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
        f=np.polyfit(x,y,n)
        p=np.poly1d(f)
        yhat=p(x)
        r2.append(r2_score(y, yhat))
        mse.append(mean_squared_error(y,yhat))
        degree.append(n)
        if n>1:
           pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
           pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
        else:
            pr2.append(0)
            pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
AxG_f1=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=AxG_f1,color='black',marker='*')
AxG_f1.legend(('R2','MSE'),prop=font)
AxG_f1.set_xlabel('Days',fontsize='medium',weight='bold')
AxG_f1.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelFf1=list(AxG_f1.get_xticks())
xlabelFf1=[int(x) for x in xlabelFf1]
AxG_f1.set_xticklabels(xlabelFf1,fontdict=fontdict)
ylabelFf1=list(AxG_f1.get_yticks())
ylabelFf1=[round(x,2) for x in ylabelFf1]
AxG_f1.set_yticklabels(ylabelFf1,fontdict=fontdict)
ellipse=Ellipse(xy=(4.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(4.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
AxG_f1.add_patch(ellipse)
plt.show()
#n=5, order polynomial
f=np.polyfit(x,y,5)
p=np.poly1d(f)
xp=Germany['Updated']
yp=p(xp)
#xp=np.array(xp)
combined = np.vstack((xp, yp)).T
GermanyP=pd.DataFrame(combined,columns=['Updated','Total Cases'])
###
AxG1=Germany.plot(kind='line',x='Updated',y='Total Cases',color='gold',ylim=(0,Germany['Total Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
AxG1.fill_between(Germany.iloc[QS_Germany:QE_Germany,17],Germany.iloc[QS_Germany:QE_Germany,10],facecolor='yellow',\
		  edgecolor='gold')
AxG1.set_xlabel('Days',fontsize='medium',weight='bold')
AxG1.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelAxG1=list(AxG1.get_xticks())
xlabelAxG1=[int(x) for x in xlabelAxG1]
AxG1.set_xticklabels(xlabelAxG1,fontdict=fontdict)
ylabelAxG1=list(AxG1.get_yticks())
ylabelAxG1=[int(x) for x in ylabelAxG1]
AxG1.set_yticklabels(ylabelAxG1,fontdict=fontdict)
GermanyP.plot(kind='line',x='Updated',y='Total Cases',color='black',fontsize=10,ax=AxG1,marker='*')
AxG1.legend(('Actual','Predicted'),prop=font,loc='upper left')
AxG1.fill_between(Germany.iloc[QS_Germany:,17],Germany.iloc[QS_Germany:,10],GermanyP.iloc[QS_Germany:,1],\
                  facecolor='white',edgecolor='black', hatch='x')
xyt=(Germany.iloc[QS_Germany,17]+12,Germany.iloc[QE_Germany,10]/4)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='chocolate',fontsize='large',\
             weight='bold')
plt.show()
#Total Deaths
#Total Deaths
Z,Y=GermanyI[['Updated','Total Cases']],GermanyI[['Total Deaths']]
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
    Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=n)),('model',LinearRegression())]
    pipe=Pipeline(Input)
    pipe.fit(Z,Y)
    ypipe=pipe.predict(Z)
    r2.append(r2_score(ypipe,Y))
    mse.append(mean_squared_error(ypipe,Y))
    degree.append(n)
    if n>1:
       pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
       pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
    else:
        pr2.append(0)
        pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
AxG_f2=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=AxG_f2,color='black',marker='*')
AxG_f2.legend(('R2','MSE'),prop=font)
AxG_f2.set_xlabel('Days',fontsize='medium',weight='bold')
AxG_f2.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelFf2=list(AxG_f2.get_xticks())
xlabelFf2=[int(x) for x in xlabelFf2]
AxG_f2.set_xticklabels(xlabelFf2,fontdict=fontdict)
ylabelFf2=list(AxG_f2.get_yticks())
ylabelFf2=[round(x,2) for x in ylabelFf2]
AxG_f2.set_yticklabels(ylabelFf2,fontdict=fontdict)
ellipse=Ellipse(xy=(3.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(3.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
AxG_f2.add_patch(ellipse)
plt.show()
#n=3, order polynomial
ZP=Germany[['Updated','Total Cases']]
Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=3)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,Y)
YPn=pipe.predict(ZP)
GermanyP=pd.DataFrame(YPn)
GermanyP.columns=['Total Deaths']
GermanyP['Updated']=Germany['Updated']
AxG2=Germany.plot(kind='line',x='Updated',y='Total Deaths',color='gold',ylim=(0,Germany['Total Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
AxG2.fill_between(Germany.iloc[QS_Germany:QE_Germany,17],Germany.iloc[QS_Germany:QE_Germany,11],facecolor='yellow',\
		  edgecolor='gold')
AxG2.set_xlabel('Days',fontsize='medium',weight='bold')
AxG2.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxG2=list(AxG2.get_xticks())
xlabelAxG2=[int(x) for x in xlabelAxG2]
AxG2.set_xticklabels(xlabelAxG2,fontdict=fontdict)
ylabelAxG2=list(AxG2.get_yticks())
ylabelAxG2=[int(x) for x in ylabelAxG2]
AxG2.set_yticklabels(ylabelAxG2,fontdict=fontdict)
GermanyP.plot(kind='line',x='Updated',y='Total Deaths',color='black',fontsize=10,ax=AxG2,marker='*')
AxG2.legend(('Actual','Predicted'),prop=font,loc='upper left')

AxG2.fill_between(Germany.iloc[QS_Germany:,17],Germany.iloc[QS_Germany:,11],GermanyP.iloc[QS_Germany:,0],\
                  facecolor='white',edgecolor='black', hatch='x')
xyt=(Germany.iloc[QS_Germany,17]+20,Germany.iloc[QE_Germany,11]/4)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='chocolate',fontsize='large',\
             weight='bold')
plt.show()
#New Cases
axG3=Germany.plot(kind='line',x='Updated',y='New Cases',color='gold',ylim=(0,Germany['New Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axG3.fill_between(Germany.iloc[QS_Germany:QE_Germany,17],Germany.iloc[QS_Germany:QE_Germany,3],facecolor='yellow',\
		     hatch='x',edgecolor='gold')
axG3.get_legend().remove()
axG3.set_xlabel('Days',fontsize='medium',weight='bold')
axG3.set_ylabel('New Cases',fontsize='medium',weight='bold')
xlabelaxG3=list(axG3.get_xticks())
xlabelaxG3=[int(x) for x in xlabelaxG3]
axG3.set_xticklabels(xlabelaxG3,fontdict=fontdict)
ylabelaxG3=list(axG3.get_yticks())
ylabelaxG3=[int(x) for x in ylabelaxG3]
axG3.set_yticklabels(ylabelaxG3,fontdict=fontdict)
plt.show()
#New Deaths
axG4=Germany.plot(kind='line',x='Updated',y='New Deaths',color='gold',ylim=(0,Germany['New Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axG4.fill_between(Germany.iloc[QS_Germany:QE_Germany,17],Germany.iloc[QS_Germany:QE_Germany,4],facecolor='yellow',\
		     hatch='x',edgecolor='gold')
axG4.get_legend().remove()
axG4.set_xlabel('Days',fontsize='medium',weight='bold')
axG4.set_ylabel('New Deaths',fontsize='medium',weight='bold')
xlabelaxG4=list(axG4.get_xticks())
xlabelaxG4=[int(x) for x in xlabelaxG4]
axG4.set_xticklabels(xlabelaxG4,fontdict=fontdict)
ylabelaxG4=list(axG4.get_yticks())
ylabelaxG4=[int(x) for x in ylabelaxG4]
axG4.set_yticklabels(ylabelaxG4,fontdict=fontdict)
plt.show()
#Italy
#Total Cases
#Italy
#Total Cases
ItalyI=Italy.iloc[:QS_Italy,:]
x,y=ItalyI['Updated'],ItalyI['Total Cases']
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
        f=np.polyfit(x,y,n)
        p=np.poly1d(f)
        yhat=p(x)
        r2.append(r2_score(y, yhat))
        mse.append(mean_squared_error(y,yhat))
        degree.append(n)
        if n>1:
           pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
           pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
        else:
            pr2.append(0)
            pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
AxI_f1=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=AxI_f1,color='black',marker='*')
AxI_f1.legend(('R2','MSE'),prop=font)
AxI_f1.set_xlabel('Days',fontsize='medium',weight='bold')
AxI_f1.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelFf1=list(AxI_f1.get_xticks())
xlabelFf1=[int(x) for x in xlabelFf1]
AxI_f1.set_xticklabels(xlabelFf1,fontdict=fontdict)
ylabelFf1=list(AxI_f1.get_yticks())
ylabelFf1=[round(x,2) for x in ylabelFf1]
AxI_f1.set_yticklabels(ylabelFf1,fontdict=fontdict)
ellipse=Ellipse(xy=(4.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(4.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
AxI_f1.add_patch(ellipse)
plt.show()
#n=5, order polynomial
f=np.polyfit(x,y,5)
p=np.poly1d(f)
xp=Italy['Updated']
yp=p(xp)
#xp=np.array(xp)
combined = np.vstack((xp, yp)).T
ItalyP=pd.DataFrame(combined,columns=['Updated','Total Cases'])
###
AxI1=Italy.plot(kind='line',x='Updated',y='Total Cases',color='darkgreen',ylim=(0,Italy['Total Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
AxI1.fill_between(Italy.iloc[QS_Italy:QE_Italy,17],Italy.iloc[QS_Italy:QE_Italy,10],facecolor='azure',\
		  edgecolor='lime')
AxI1.set_xlabel('Days',fontsize='medium',weight='bold')
AxI1.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelAxI1=list(AxI1.get_xticks())
xlabelAxI1=[int(x) for x in xlabelAxI1]
AxI1.set_xticklabels(xlabelAxI1,fontdict=fontdict)
ylabelAxI1=list(AxI1.get_yticks())
ylabelAxI1=[int(x) for x in ylabelAxI1]
AxI1.set_yticklabels(ylabelAxI1,fontdict=fontdict)
ItalyP.plot(kind='line',x='Updated',y='Total Cases',color='black',fontsize=10,ax=AxI1,marker='*')
AxI1.legend(('Actual','Predicted'),prop=font,loc='upper left')

AxI1.fill_between(Italy.iloc[QS_Italy:,17],Italy.iloc[QS_Italy:,10],ItalyP.iloc[QS_Italy:,1],\
                  facecolor='white',edgecolor='black', hatch='x')
xyt=(Italy.iloc[QS_Italy,17]+24,Italy.iloc[QE_Italy,10]/4)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='darkgreen',fontsize='large',\
             weight='bold')
plt.show()
#Total Deaths

#Total Deaths
Z,Y=ItalyI[['Updated','Total Cases']],ItalyI[['Total Deaths']]
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
    Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=n)),('model',LinearRegression())]
    pipe=Pipeline(Input)
    pipe.fit(Z,Y)
    ypipe=pipe.predict(Z)
    r2.append(r2_score(ypipe,Y))
    mse.append(mean_squared_error(ypipe,Y))
    degree.append(n)
    if n>1:
       pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
       pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
    else:
        pr2.append(0)
        pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
AxI_f2=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=AxI_f2,color='black',marker='*')
AxI_f2.legend(('R2','MSE'),prop=font)
AxI_f2.set_xlabel('Days',fontsize='medium',weight='bold')
AxI_f2.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelFf2=list(AxI_f2.get_xticks())
xlabelFf2=[int(x) for x in xlabelFf2]
AxI_f2.set_xticklabels(xlabelFf2,fontdict=fontdict)
ylabelFf2=list(AxI_f2.get_yticks())
ylabelFf2=[round(x,2) for x in ylabelFf2]
AxI_f2.set_yticklabels(ylabelFf2,fontdict=fontdict)
ellipse=Ellipse(xy=(3.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(3.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
AxI_f2.add_patch(ellipse)
plt.show()
#n=3, order polynomial
ZP=Italy[['Updated','Total Cases']]
Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=1)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,Y)
YPn=pipe.predict(ZP)
ItalyP=pd.DataFrame(YPn)
ItalyP.columns=['Total Deaths']
ItalyP['Updated']=Italy['Updated']
AxI2=Italy.plot(kind='line',x='Updated',y='Total Deaths',color='navy',ylim=(0,Italy['Total Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
AxI2.fill_between(Italy.iloc[QS_Italy:QE_Italy,17],Italy.iloc[QS_Italy:QE_Italy,11],facecolor='lime',\
		  edgecolor='darkgreen')
AxI2.set_xlabel('Days',fontsize='medium',weight='bold')
AxI2.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxI2=list(AxI2.get_xticks())
xlabelAxI2=[int(x) for x in xlabelAxI2]
AxI2.set_xticklabels(xlabelAxI2,fontdict=fontdict)
ylabelAxI2=list(AxI2.get_yticks())
ylabelAxI2=[int(x) for x in ylabelAxI2]
AxI2.set_yticklabels(ylabelAxI2,fontdict=fontdict)
ItalyP.plot(kind='line',x='Updated',y='Total Deaths',color='black',fontsize=10,ax=AxI2,marker='*')
AxI2.legend(('Actual','Predicted'),prop=font,loc='upper left')

AxI2.fill_between(Italy.iloc[QS_Italy:,17],Italy.iloc[QS_Italy:,11],ItalyP.iloc[QS_Italy:,0],\
                  facecolor='white',edgecolor='black', hatch='x')
xyt=(Italy.iloc[QS_Italy,17]+20,Italy.iloc[QE_Italy,11]/4)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='darkgreen',fontsize='large',\
             weight='bold')
plt.show()
#New Cases
axI3=Italy.plot(kind='line',x='Updated',y='New Cases',color='darkgreen',ylim=(0,Italy['New Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axI3.fill_between(Italy.iloc[QS_Italy:QE_Italy,17],Italy.iloc[QS_Italy:QE_Italy,3],facecolor='lime',\
		     hatch='x',edgecolor='darkgreen')
axI3.get_legend().remove()
axI3.set_xlabel('Days',fontsize='medium',weight='bold')
axI3.set_ylabel('New Cases',fontsize='medium',weight='bold')
xlabelaxI3=list(axI3.get_xticks())
xlabelaxI3=[int(x) for x in xlabelaxI3]
axI3.set_xticklabels(xlabelaxI3,fontdict=fontdict)
ylabelaxI3=list(axI3.get_yticks())
ylabelaxI3=[int(x) for x in ylabelaxI3]
axI3.set_yticklabels(ylabelaxI3,fontdict=fontdict)
plt.show()
#New Deaths
axI4=Italy.plot(kind='line',x='Updated',y='New Deaths',color='darkgreen',ylim=(0,Italy['New Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axI4.fill_between(Italy.iloc[QS_Italy:QE_Italy,17],Italy.iloc[QS_Italy:QE_Italy,4],facecolor='lime',\
		     hatch='x',edgecolor='darkgreen')
axI4.get_legend().remove()
axI4.set_xlabel('Days',fontsize='medium',weight='bold')
axI4.set_ylabel('New Deaths',fontsize='medium',weight='bold')
xlabelaxI4=list(axI4.get_xticks())
xlabelaxI4=[int(x) for x in xlabelaxI4]
axI4.set_xticklabels(xlabelaxI4,fontdict=fontdict)
ylabelaxI4=list(axI4.get_yticks())
ylabelaxI4=[int(x) for x in ylabelaxI4]
axI4.set_yticklabels(ylabelaxI4,fontdict=fontdict)
plt.show()
#Japan
#Total Cases
#Japan
#Total Cases
JapanI=Japan.iloc[:QS_Japan,:]
x,y,z=JapanI['Updated'],JapanI['Total Cases'],JapanI['Total Deaths']
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
        f=np.polyfit(x,y,n)
        p=np.poly1d(f)
        yhat=p(x)
        r2.append(r2_score(y, yhat))
        mse.append(mean_squared_error(y,yhat))
        degree.append(n)
        if n>1:
           pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
           pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
        else:
            pr2.append(0)
            pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
AxJ_f1=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=AxJ_f1,color='black',marker='*')
AxJ_f1.legend(('R2','MSE'),prop=font)
AxJ_f1.set_xlabel('Days',fontsize='medium',weight='bold')
AxJ_f1.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelFf1=list(AxJ_f1.get_xticks())
xlabelFf1=[int(x) for x in xlabelFf1]
AxJ_f1.set_xticklabels(xlabelFf1,fontdict=fontdict)
ylabelFf1=list(AxJ_f1.get_yticks())
ylabelFf1=[round(x,2) for x in ylabelFf1]
AxJ_f1.set_yticklabels(ylabelFf1,fontdict=fontdict)
ellipse=Ellipse(xy=(4.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(4.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
AxJ_f1.add_patch(ellipse)
plt.show()
#n=5, order polynomial
f=np.polyfit(x,y,5)
p=np.poly1d(f)
xp=Japan['Updated']
yp=p(xp)
#xp=np.array(xp)
combined = np.vstack((xp, yp)).T
JapanP=pd.DataFrame(combined,columns=['Updated','Total Cases'])
###
AxJ1=Japan.plot(kind='line',x='Updated',y='Total Cases',color='rebeccapurple',ylim=(0,Japan['Total Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
AxJ1.fill_between(Japan.iloc[QS_Japan:QE_Japan,17],Japan.iloc[QS_Japan:QE_Japan,10],facecolor='magenta',\
		  edgecolor='rebeccapurple')
AxJ1.set_xlabel('Days',fontsize='medium',weight='bold')
AxJ1.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelAxJ1=list(AxJ1.get_xticks())
xlabelAxJ1=[int(x) for x in xlabelAxJ1]
AxJ1.set_xticklabels(xlabelAxJ1,fontdict=fontdict)
ylabelAxJ1=list(AxJ1.get_yticks())
ylabelAxJ1=[int(x) for x in ylabelAxJ1]
AxJ1.set_yticklabels(ylabelAxJ1,fontdict=fontdict)
JapanP.plot(kind='line',x='Updated',y='Total Cases',color='black',fontsize=10,ax=AxJ1,marker='*')
AxJ1.legend(('Actual','Predicted'),prop=font,loc='upper left')

AxJ1.fill_between(Japan.iloc[QS_Japan:,17],Japan.iloc[QS_Japan:,10],JapanP.iloc[QS_Japan:,1],\
                  facecolor='white',edgecolor='black', hatch='x')
xyt=(Japan.iloc[QS_Japan,17]+4,Japan.iloc[QE_Japan,10]/4)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='rebeccapurple',fontsize='large',\
             weight='bold')
plt.show()
#Total Deaths
Z,Y=JapanI[['Updated','Total Cases']],JapanI[['Total Deaths']]
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
    Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=n)),('model',LinearRegression())]
    pipe=Pipeline(Input)
    pipe.fit(Z,Y)
    ypipe=pipe.predict(Z)
    r2.append(r2_score(ypipe,Y))
    mse.append(mean_squared_error(ypipe,Y))
    degree.append(n)
    if n>1:
       pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
       pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
    else:
        pr2.append(0)
        pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
AxJ_f2=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=AxJ_f2,color='black',marker='*')
AxJ_f2.legend(('R2','MSE'),prop=font)
AxJ_f2.set_xlabel('Days',fontsize='medium',weight='bold')
AxJ_f2.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelFf2=list(AxJ_f2.get_xticks())
xlabelFf2=[int(x) for x in xlabelFf2]
AxJ_f2.set_xticklabels(xlabelFf2,fontdict=fontdict)
ylabelFf2=list(AxJ_f2.get_yticks())
ylabelFf2=[round(x,2) for x in ylabelFf2]
AxJ_f2.set_yticklabels(ylabelFf2,fontdict=fontdict)
ellipse=Ellipse(xy=(3.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(3.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
AxJ_f2.add_patch(ellipse)
plt.show()
#n=3, order polynomial
ZP=Japan[['Updated','Total Cases']]
Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=3)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,Y)
YPn=pipe.predict(ZP)
JapanP=pd.DataFrame(YPn)
JapanP.columns=['Total Deaths']
JapanP['Updated']=Japan['Updated']
AxJ2=Japan.plot(kind='line',x='Updated',y='Total Deaths',color='rebeccapurple',ylim=(0,Japan['Total Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
AxJ2.fill_between(Japan.iloc[QS_Japan:QE_Japan,17],Japan.iloc[QS_Japan:QE_Japan,11],facecolor='magenta',\
		  edgecolor='rebeccapurple')
AxJ2.set_xlabel('Days',fontsize='medium',weight='bold')
AxJ2.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxJ2=list(AxJ2.get_xticks())
xlabelAxJ2=[int(x) for x in xlabelAxJ2]
AxJ2.set_xticklabels(xlabelAxJ2,fontdict=fontdict)
ylabelAxJ2=list(AxJ2.get_yticks())
ylabelAxJ2=[int(x) for x in ylabelAxJ2]
AxJ2.set_yticklabels(ylabelAxJ2,fontdict=fontdict)
JapanP.plot(kind='line',x='Updated',y='Total Deaths',color='black',fontsize=10,ax=AxJ2,marker='*')
AxJ2.legend(('Actual','Predicted'),prop=font,loc='upper left')

AxJ2.fill_between(Japan.iloc[QS_Japan:,17],Japan.iloc[QS_Japan:,11],JapanP.iloc[QS_Japan:,0],\
                  facecolor='white',edgecolor='black', hatch='x')
xyt=(Japan.iloc[QS_Japan,17]+8,Japan.iloc[QE_Japan,11]/4)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='rebeccapurple',fontsize='large',\
             weight='bold')
plt.show()
#New Cases
axJ3=Japan.plot(kind='line',x='Updated',y='New Cases',color='purple',ylim=(0,Japan['New Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axJ3.fill_between(Japan.iloc[QS_Japan:QE_Japan,17],Japan.iloc[QS_Japan:QE_Japan,3],facecolor='magenta',\
		     hatch='x',edgecolor='rebeccapurple')
axJ3.get_legend().remove()
axJ3.set_xlabel('Days',fontsize='medium',weight='bold')
axJ3.set_ylabel('New Cases',fontsize='medium',weight='bold')
xlabelaxJ3=list(axJ3.get_xticks())
xlabelaxJ3=[int(x) for x in xlabelaxJ3]
axJ3.set_xticklabels(xlabelaxJ3,fontdict=fontdict)
ylabelaxJ3=list(axJ3.get_yticks())
ylabelaxJ3=[int(x) for x in ylabelaxJ3]
axJ3.set_yticklabels(ylabelaxJ3,fontdict=fontdict)
plt.show()
#New Deaths
axJ4=Japan.plot(kind='line',x='Updated',y='New Deaths',color='rebeccapurple',ylim=(0,Japan['New Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axJ4.fill_between(Japan.iloc[QS_Japan:QE_Japan,17],Japan.iloc[QS_Japan:QE_Japan,4],facecolor='magenta',\
		     hatch='x',edgecolor='rebeccapurple')
axJ4.get_legend().remove()
axJ4.set_xlabel('Days',fontsize='medium',weight='bold')
axJ4.set_ylabel('New Deaths',fontsize='medium',weight='bold')
xlabelaxJ4=list(axJ4.get_xticks())
xlabelaxJ4=[int(x) for x in xlabelaxJ4]
axJ4.set_xticklabels(xlabelaxJ4,fontdict=fontdict)
ylabelaxJ4=list(axJ4.get_yticks())
ylabelaxJ4=[int(x) for x in ylabelaxJ4]
axJ4.set_yticklabels(ylabelaxJ4,fontdict=fontdict)
plt.show()
#UK
#Total Cases
UKI=UK.iloc[:QS_UK,:]
x,y,z=UKI['Updated'],UKI['Total Cases'],UKI['Total Deaths']
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
        f=np.polyfit(x,y,n)
        p=np.poly1d(f)
        yhat=p(x)
        r2.append(r2_score(y, yhat))
        mse.append(mean_squared_error(y,yhat))
        degree.append(n)
        if n>1:
           pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
           pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
        else:
            pr2.append(0)
            pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
AxK_f1=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=AxK_f1,color='black',marker='*')
AxK_f1.legend(('R2','MSE'),prop=font)
AxK_f1.set_xlabel('Days',fontsize='medium',weight='bold')
AxK_f1.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelFf1=list(AxK_f1.get_xticks())
xlabelFf1=[int(x) for x in xlabelFf1]
AxK_f1.set_xticklabels(xlabelFf1,fontdict=fontdict)
ylabelFf1=list(AxK_f1.get_yticks())
ylabelFf1=[round(x,2) for x in ylabelFf1]
AxK_f1.set_yticklabels(ylabelFf1,fontdict=fontdict)
ellipse=Ellipse(xy=(4.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(4.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
AxK_f1.add_patch(ellipse)
plt.show()
#n=5, order polynomial
f=np.polyfit(x,y,5)
p=np.poly1d(f)
xp=UK['Updated']
yp=p(xp)
combined = np.vstack((xp, yp)).T
UKP=pd.DataFrame(combined,columns=['Updated','Total Cases'])
###
AxK1=UK.plot(kind='line',x='Updated',y='Total Cases',color='red',ylim=(0,UK['Total Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
AxK1.fill_between(UK.iloc[QS_UK:QE_UK,17],UK.iloc[QS_UK:QE_UK,10],facecolor='pink',\
		  edgecolor='red')
AxK1.set_xlabel('Days',fontsize='medium',weight='bold')
AxK1.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelAxK1=list(AxK1.get_xticks())
xlabelAxK1=[int(x) for x in xlabelAxK1]
AxK1.set_xticklabels(xlabelAxK1,fontdict=fontdict)
ylabelAxK1=list(AxK1.get_yticks())
ylabelAxK1=[int(x) for x in ylabelAxK1]
AxK1.set_yticklabels(ylabelAxK1,fontdict=fontdict)
UKP.plot(kind='line',x='Updated',y='Total Cases',color='black',fontsize=10,ax=AxK1,marker='*')
AxK1.legend(('Actual','Predicted'),prop=font,loc='upper left')

AxK1.fill_between(UK.iloc[QS_UK:,17],UK.iloc[QS_UK:,10],UKP.iloc[QS_UK:,1],\
                  facecolor='white',edgecolor='black', hatch='x')
xyt=(UK.iloc[QS_UK,17]+32,UK.iloc[QE_UK,10]/4)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='red',fontsize='large',\
             weight='bold')
plt.show()
#Total Deaths
Z,Y=UKI[['Updated','Total Cases']],UKI[['Total Deaths']]
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
    Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=n)),('model',LinearRegression())]
    pipe=Pipeline(Input)
    pipe.fit(Z,Y)
    ypipe=pipe.predict(Z)
    r2.append(r2_score(ypipe,Y))
    mse.append(mean_squared_error(ypipe,Y))
    degree.append(n)
    if n>1:
       pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
       pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
    else:
        pr2.append(0)
        pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
AxK_f2=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=AxK_f2,color='black',marker='*')
AxK_f2.legend(('R2','MSE'),prop=font)
AxK_f2.set_xlabel('Days',fontsize='medium',weight='bold')
AxK_f2.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelFf2=list(AxK_f2.get_xticks())
xlabelFf2=[int(x) for x in xlabelFf2]
AxK_f2.set_xticklabels(xlabelFf2,fontdict=fontdict)
ylabelFf2=list(AxK_f2.get_yticks())
ylabelFf2=[round(x,2) for x in ylabelFf2]
AxK_f2.set_yticklabels(ylabelFf2,fontdict=fontdict)
ellipse=Ellipse(xy=(3.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(3.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
AxK_f2.add_patch(ellipse)
plt.show()
#n=3, order polynomial
ZP=UK[['Updated','Total Cases']]
Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=5)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,Y)
YPn=pipe.predict(ZP)
UKP=pd.DataFrame(YPn)
UKP.columns=['Total Deaths']
UKP['Updated']=UK['Updated']
AxK2=UK.plot(kind='line',x='Updated',y='Total Deaths',color='red',ylim=(0,UK['Total Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
AxK2.fill_between(UK.iloc[QS_UK:QE_UK,17],UK.iloc[QS_UK:QE_UK,11],facecolor='pink',\
		  edgecolor='red')
AxK2.set_xlabel('Days',fontsize='medium',weight='bold')
AxK2.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxK2=list(AxK2.get_xticks())
xlabelAxK2=[int(x) for x in xlabelAxK2]
AxK2.set_xticklabels(xlabelAxK2,fontdict=fontdict)
ylabelAxK2=list(AxK2.get_yticks())
ylabelAxK2=[int(x) for x in ylabelAxK2]
AxK2.set_yticklabels(ylabelAxK2,fontdict=fontdict)
UKP.plot(kind='line',x='Updated',y='Total Deaths',color='black',fontsize=10,ax=AxK2,marker='*')
AxK2.legend(('Actual','Predicted'),prop=font,loc='upper left')

AxK2.fill_between(UK.iloc[QS_UK:,17],UK.iloc[QS_UK:,11],UKP.iloc[QS_UK:,0],\
                  facecolor='white',edgecolor='black', hatch='x')
xyt=(UK.iloc[QS_UK,17]+8,UK.iloc[QE_UK,11]/4)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='red',fontsize='large',\
             weight='bold')
plt.show()
#New Cases
axK3=UK.plot(kind='line',x='Updated',y='New Cases',color='black',ylim=(0,UK['New Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axK3.fill_between(UK.iloc[QS_UK:QE_UK,17],UK.iloc[QS_UK:QE_UK,3],facecolor='silver',\
		     hatch='x',edgecolor='black')
axK3.get_legend().remove()
axK3.set_xlabel('Days',fontsize='medium',weight='bold')
axK3.set_ylabel('New Cases',fontsize='medium',weight='bold')
xlabelaxK3=list(axK3.get_xticks())
xlabelaxK3=[int(x) for x in xlabelaxK3]
axK3.set_xticklabels(xlabelaxK3,fontdict=fontdict)
ylabelaxK3=list(axK3.get_yticks())
ylabelaxK3=[int(x) for x in ylabelaxK3]
axK3.set_yticklabels(ylabelaxK3,fontdict=fontdict)
plt.show()
#New Deaths
axK4=UK.plot(kind='line',x='Updated',y='New Deaths',color='black',ylim=(0,UK['New Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axK4.fill_between(UK.iloc[QS_UK:QE_UK,17],UK.iloc[QS_UK:QE_UK,4],facecolor='silver',\
		     hatch='x',edgecolor='black')
axK4.get_legend().remove()
axK4.set_xlabel('Days',fontsize='medium',weight='bold')
axK4.set_ylabel('New Deaths',fontsize='medium',weight='bold')
xlabelaxK4=list(axK4.get_xticks())
xlabelaxK4=[int(x) for x in xlabelaxK4]
axK4.set_xticklabels(xlabelaxK4,fontdict=fontdict)
ylabelaxK4=list(axK4.get_yticks())
ylabelaxK4=[int(x) for x in ylabelaxK4]
axK4.set_yticklabels(ylabelaxK4,fontdict=fontdict)
plt.show()
#USA
#Total Cases
#USA
#Total Cases
USAI=USA.iloc[:QS_USA,:]
x,y,z=USAI['Updated'],USAI['Total Cases'],USAI['Total Deaths']
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
        f=np.polyfit(x,y,n)
        p=np.poly1d(f)
        yhat=p(x)
        r2.append(r2_score(y, yhat))
        mse.append(mean_squared_error(y,yhat))
        degree.append(n)
        if n>1:
           pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
           pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
        else:
            pr2.append(0)
            pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
AxA_f1=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=AxA_f1,color='black',marker='*')
AxA_f1.legend(('R2','MSE'),prop=font)
AxA_f1.set_xlabel('Days',fontsize='medium',weight='bold')
AxA_f1.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelFf1=list(AxA_f1.get_xticks())
xlabelFf1=[int(x) for x in xlabelFf1]
AxA_f1.set_xticklabels(xlabelFf1,fontdict=fontdict)
ylabelFf1=list(AxA_f1.get_yticks())
ylabelFf1=[round(x,2) for x in ylabelFf1]
AxA_f1.set_yticklabels(ylabelFf1,fontdict=fontdict)
ellipse=Ellipse(xy=(5.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(5.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
AxA_f1.add_patch(ellipse)
plt.show()
#n=6, order polynomial
f=np.polyfit(x,y,6)
p=np.poly1d(f)
xp=USA['Updated']
yp=p(xp)
combined = np.vstack((xp, yp)).T
USAP=pd.DataFrame(combined,columns=['Updated','Total Cases'])
###
AxA1=USA.plot(kind='line',x='Updated',y='Total Cases',color='blue',ylim=(0,USA['Total Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
AxA1.fill_between(USA.iloc[QS_USA:QE_USA,17],USA.iloc[QS_USA:QE_USA,10],facecolor='aqua',\
		  edgecolor='blue')
AxA1.set_xlabel('Days',fontsize='medium',weight='bold')
AxA1.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelAxA1=list(AxA1.get_xticks())
xlabelAxA1=[int(x) for x in xlabelAxA1]
AxA1.set_xticklabels(xlabelAxA1,fontdict=fontdict)
ylabelAxA1=list(AxA1.get_yticks())
ylabelAxA1=[int(x) for x in ylabelAxA1]
AxA1.set_yticklabels(ylabelAxA1,fontdict=fontdict)
USAP.plot(kind='line',x='Updated',y='Total Cases',color='black',fontsize=10,ax=AxA1,marker='*')
AxA1.legend(('Actual','Predicted'),prop=font,loc='upper left')

AxA1.fill_between(USA.iloc[QS_USA:,17],USA.iloc[QS_USA:,10],USAP.iloc[QS_USA:,1],\
                  facecolor='white',edgecolor='black', hatch='x')
xyt=(USA.iloc[QS_USA,17]+16,USA.iloc[QE_USA,10]/6)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='blue',fontsize='large',\
             weight='bold')
plt.show()
#Total Deaths
Z,Y=USAI[['Updated','Total Cases']],USAI[['Total Deaths']]
r2,mse,degree=[],[],[]
pr2,pmse=[],[]
#finding the best polynomial fit before the quarantine 
for n in range(1,11):
    Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=n)),('model',LinearRegression())]
    pipe=Pipeline(Input)
    pipe.fit(Z,Y)
    ypipe=pipe.predict(Z)
    r2.append(r2_score(ypipe,Y))
    mse.append(mean_squared_error(ypipe,Y))
    degree.append(n)
    if n>1:
       pr2.append(abs(r2[n-1]-r2[n-2])/r2[n-2]*100)
       pmse.append(abs(mse[n-1]-mse[n-2])/mse[n-2]*100)
    else:
        pr2.append(0)
        pmse.append(0)
bestfit=pd.DataFrame([degree,r2,mse,pr2,pmse]).transpose()
bestfit.columns=['Degree','R2','MSE','slope_R2%','slope_MSE%']
#Normalizing
MSE = bestfit.MSE
MSE=np.array(MSE)
MSEn=normalize([MSE])
MSEn=pd.DataFrame(MSEn).transpose() # coverting the normalized values to dataframe and merging
MSEn.columns=['MSEN']
bestfit['MSEN']=MSEn['MSEN']
#Graph
AxA_f2=bestfit.plot(x='Degree',y='R2',xlim=(1,10), ylim=(0,1.2),color='navy',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='MSEN',ax=AxA_f2,color='black',marker='*')
AxA_f2.legend(('R2','MSE'),prop=font)
AxA_f2.set_xlabel('Days',fontsize='medium',weight='bold')
AxA_f2.set_ylabel(' ',fontsize='medium',weight='bold')
xlabelFf2=list(AxA_f2.get_xticks())
xlabelFf2=[int(x) for x in xlabelFf2]
AxA_f2.set_xticklabels(xlabelFf2,fontdict=fontdict)
ylabelFf2=list(AxA_f2.get_yticks())
ylabelFf2=[round(x,2) for x in ylabelFf2]
AxA_f2.set_yticklabels(ylabelFf2,fontdict=fontdict)
ellipse=Ellipse(xy=(3.5,.5), width=2, height=.98, facecolor='None',edgecolor='black')
plt.annotate('Best Fitting Area',xy=(3.5,.3),rotation=90,color='black',fontsize='medium',\
             weight='bold')
AxA_f2.add_patch(ellipse)
plt.show()
#n=3, order polynomial
ZP=USA[['Updated','Total Cases']]
Input=[('Scale',StandardScaler()),('Polynomial',PolynomialFeatures(degree=3)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,Y)
YPn=pipe.predict(ZP)
USAP=pd.DataFrame(YPn)
USAP.columns=['Total Deaths']
USAP['Updated']=USA['Updated']
AxA2=USA.plot(kind='line',x='Updated',y='Total Deaths',color='blue',ylim=(0,USA['Total Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
AxA2.fill_between(USA.iloc[QS_USA:QE_USA,17],USA.iloc[QS_USA:QE_USA,11],facecolor='aqua',\
		  edgecolor='blue')
AxA2.set_xlabel('Days',fontsize='medium',weight='bold')
AxA2.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxA2=list(AxA2.get_xticks())
xlabelAxA2=[int(x) for x in xlabelAxA2]
AxA2.set_xticklabels(xlabelAxA2,fontdict=fontdict)
ylabelAxA2=list(AxA2.get_yticks())
ylabelAxA2=[int(x) for x in ylabelAxA2]
AxA2.set_yticklabels(ylabelAxA2,fontdict=fontdict)
USAP.plot(kind='line',x='Updated',y='Total Deaths',color='black',fontsize=10,ax=AxA2,marker='*')
AxA2.legend(('Actual','Pnavyicted'),prop=font,loc='upper left')

AxA2.fill_between(USA.iloc[QS_USA:,17],USA.iloc[QS_USA:,11],USAP.iloc[QS_USA:,0],\
                  facecolor='white',edgecolor='black', hatch='x')
xyt=(USA.iloc[QS_USA,17]+8,USA.iloc[QE_USA,11]/4)
plt.annotate('Quarantine',xy=xyt,rotation=45,color='blue',fontsize='large',\
             weight='bold')
plt.show()
#New Cases
axA3=USA.plot(kind='line',x='Updated',y='New Cases',color='blue',ylim=(0,USA['New Cases'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axA3.fill_between(USA.iloc[QS_USA:QE_USA,17],USA.iloc[QS_USA:QE_USA,3],facecolor='aqua',\
		     hatch='x',edgecolor='blue')
axA3.get_legend().remove()
axA3.set_xlabel('Days',fontsize='medium',weight='bold')
axA3.set_ylabel('New Cases',fontsize='medium',weight='bold')
xlabelaxA3=list(axA3.get_xticks())
xlabelaxA3=[int(x) for x in xlabelaxA3]
axA3.set_xticklabels(xlabelaxA3,fontdict=fontdict)
ylabelaxA3=list(axA3.get_yticks())
ylabelaxA3=[int(x) for x in ylabelaxA3]
axA3.set_yticklabels(ylabelaxA3,fontdict=fontdict)
plt.show()
#New Deaths
axA4=USA.plot(kind='line',x='Updated',y='New Deaths',color='blue',ylim=(0,USA['New Deaths'].max()*1.25),\
		    xlim=(0,df['Updated'].max()),fontsize=10)
axA4.fill_between(USA.iloc[QS_USA:QE_USA,17],USA.iloc[QS_USA:QE_USA,4],facecolor='aqua',\
		     hatch='x',edgecolor='blue')
axA4.get_legend().remove()
axA4.set_xlabel('Days',fontsize='medium',weight='bold')
axA4.set_ylabel('New Deaths',fontsize='medium',weight='bold')
xlabelaxA4=list(axA4.get_xticks())
xlabelaxA4=[int(x) for x in xlabelaxA4]
axA4.set_xticklabels(xlabelaxA4,fontdict=fontdict)
ylabelaxA4=list(axA4.get_yticks())
ylabelaxA4=[int(x) for x in ylabelaxA4]
axA4.set_yticklabels(ylabelaxA4,fontdict=fontdict)
plt.show()
parameters1=[{'alpha':[0.001,.1,1,10,100,1000,10000,100000]}]#alpha parameter in Ridge model 
###Canada
CanadaII=Canada[Canada['Updated']>QS_Canada]
X,Y=CanadaII[['Updated']],CanadaII[['Total Cases']]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.3)
xtr,ytr=x_train['Updated'],y_train['Total Cases']
xtst,ytst=x_test['Updated'],y_test['Total Cases']
r2_tr,mse_tr,r2_tst,mse_tst,degree=[],[],[],[],[]
xp,yp=CanadaII['Updated'],CanadaII['Total Cases']
for n in range(1,11):
    f=np.polyfit(xtr,ytr,n)
    p=np.poly1d(f)
    yhat_tr,yhat_tst=p(xtr),p(xtst)
    r2_tr.append(r2_score(ytr, yhat_tr))
    mse_tr.append(mean_squared_error(ytr, yhat_tr))
    r2_tst.append(r2_score(ytst, yhat_tst))
    mse_tst.append(mean_squared_error(ytst, yhat_tst))
    degree.append(n)

bestfit=pd.DataFrame([degree,r2_tr,mse_tr,r2_tst,mse_tst]).transpose()
bestfit.columns=['Degree','R2_tr','MSE_tr','R2_tst','MSE_tst']
columns=['MSE_tr','MSE_tst']
bestfitN=pd.DataFrame(columns=columns)
for cl in columns:
	NM = bestfit[cl]
	NM=np.array(NM)
	NMn=normalize([NM])
	NMn=pd.DataFrame(NMn).transpose()
	NMn.columns=[cl]
	bestfitN[cl]=NMn[cl]
bestfitN['Degree']=bestfit['Degree']
#graph
AxC_f3=bestfit.plot(x='Degree',y='R2_tr',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='R2_tst',xlim=(1,10), ylim=(0,1.2),color='red',marker='o',fontsize=10,ax=AxC_f3)
bestfitN.plot(x='Degree',y='MSE_tr',xlim=(1,10), ylim=(0,1.2),color='black',marker='*',fontsize=10,ax=AxC_f3)
bestfitN.plot(x='Degree',y='MSE_tst',xlim=(1,10), ylim=(0,1.2),color='black',marker='o',fontsize=10,ax=AxC_f3)
AxC_f3.legend(('R2_tr','R2_tst','MSE_tr','MSE_tst'),prop=font)
AxC_f3.set_xlabel('Degree',fontsize='medium',weight='bold')
xlabelAxC_f3=list(AxC_f3.get_xticks())
xlabelAxC_f3=[int(x) for x in xlabelAxC_f3]
AxC_f3.set_xticklabels(xlabelAxC_f3,fontdict=fontdict)
ylabelAxC_f3=list(AxC_f3.get_yticks())
ylabelAxC_f3=[int(x) for x in ylabelAxC_f3]
AxC_f3.set_yticklabels(ylabelAxC_f3,fontdict=fontdict)
plt.show()
#prdiction Total Cases
f=np.polyfit(xp,yp,4)
p=np.poly1d(f)
yhat_II=p(xp)
d=np.array([xp,yhat_II]).T
data=pd.DataFrame(d,columns=['Updated','Total Cases'])
AxC3=CanadaII.plot(x='Updated',y='Total Cases',color='red',fontsize=10)
AxC3.set_xlabel('Days',fontsize='medium',weight='bold')
AxC3.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelAxC3=list(AxC3.get_xticks())
xlabelAxC3=[int(x) for x in xlabelAxC3]
AxC3.set_xticklabels(xlabelAxC3,fontdict=fontdict)
ylabelAxC3=list(AxC3.get_yticks())
ylabelAxC3=[int(x) for x in ylabelAxC3]
AxC3.set_yticklabels(ylabelAxC3,fontdict=fontdict)
data.plot(kind='scatter',x='Updated',y='Total Cases',marker='*',\
          color='black',ax=AxC3,fontsize=10)
AxC3.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()
#prediction Total Deaths
Z,Y=CanadaII[['Updated','Total Cases']],CanadaII[['Total Deaths']]
X=np.array(CanadaII[['Updated']])
PR=Ridge()
Grid1=GridSearchCV(PR,parameters1,cv=4)
Grid1.fit(Z,Y)
yhatD=Grid1.predict(Z)
combined = np.hstack((X, yhatD))
CanadaDP=pd.DataFrame(combined,columns=['Updated','Total Deaths'])
AxC4=CanadaII.plot(x='Updated',y='Total Deaths',color='red',fontsize=10)
CanadaDP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxC4,fontsize=10)
AxC4.set_xlabel('Days',fontsize='medium',weight='bold')
AxC4.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxC4=list(AxC4.get_xticks())
xlabelAxC4=[int(x) for x in xlabelAxC4]
AxC4.set_xticklabels(xlabelAxC4,fontdict=fontdict)
ylabelAxC4=list(AxC4.get_yticks())
ylabelAxC4=[int(x) for x in ylabelAxC4]
AxC4.set_yticklabels(ylabelAxC4,fontdict=fontdict)
CanadaDP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxC4,fontsize=10)
AxC4.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()
####France
FranceII=France[France['Updated']>QS_France]
X,Y=FranceII[['Updated']],FranceII[['Total Cases']]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.3)
xtr,ytr=x_train['Updated'],y_train['Total Cases']
xtst,ytst=x_test['Updated'],y_test['Total Cases']
r2_tr,mse_tr,r2_tst,mse_tst,degree=[],[],[],[],[]
xp,yp=FranceII['Updated'],FranceII['Total Cases']
for n in range(1,11):
    f=np.polyfit(xtr,ytr,n)
    p=np.poly1d(f)
    yhat_tr,yhat_tst=p(xtr),p(xtst)
    r2_tr.append(r2_score(ytr, yhat_tr))
    mse_tr.append(mean_squared_error(ytr, yhat_tr))
    r2_tst.append(r2_score(ytst, yhat_tst))
    mse_tst.append(mean_squared_error(ytst, yhat_tst))
    degree.append(n)

bestfit=pd.DataFrame([degree,r2_tr,mse_tr,r2_tst,mse_tst]).transpose()
bestfit.columns=['Degree','R2_tr','MSE_tr','R2_tst','MSE_tst']
columns=['MSE_tr','MSE_tst']
bestfitN=pd.DataFrame(columns=columns)
for cl in columns:
	NM = bestfit[cl]
	NM=np.array(NM)
	NMn=normalize([NM])
	NMn=pd.DataFrame(NMn).transpose()
	NMn.columns=[cl]
	bestfitN[cl]=NMn[cl]
bestfitN['Degree']=bestfit['Degree']
#graph
AxF_f3=bestfit.plot(x='Degree',y='R2_tr',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='R2_tst',xlim=(1,10), ylim=(0,1.2),color='red',marker='o',fontsize=10,ax=AxF_f3)
bestfitN.plot(x='Degree',y='MSE_tr',xlim=(1,10), ylim=(0,1.2),color='black',marker='*',fontsize=10,ax=AxF_f3)
bestfitN.plot(x='Degree',y='MSE_tst',xlim=(1,10), ylim=(0,1.2),color='black',marker='o',fontsize=10,ax=AxF_f3)
AxF_f3.legend(('R2_tr','R2_tst','MSE_tr','MSE_tst'),prop=font)
xlabelAxF_f3=list(AxF_f3.get_xticks())
xlabelAxF_f3=[int(x) for x in xlabelAxF_f3]
AxF_f3.set_xticklabels(xlabelAxF_f3,fontdict=fontdict)
ylabelAxF_f3=list(AxF_f3.get_yticks())
ylabelAxF_f3=[int(x) for x in ylabelAxF_f3]
AxF_f3.set_yticklabels(ylabelAxF_f3,fontdict=fontdict)
plt.show()
#prdiction Total Cases
f=np.polyfit(xp,yp,6)
p=np.poly1d(f)
yhat_II=p(xp)
d=np.array([xp,yhat_II]).T
data=pd.DataFrame(d,columns=['Updated','Total Cases'])
AxF3=FranceII.plot(x='Updated',y='Total Cases',color='navy',fontsize=10)
AxF3.set_xlabel('Days',fontsize='medium',weight='bold')
AxF3.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelAxF3=list(AxF3.get_xticks())
xlabelAxF3=[int(x) for x in xlabelAxF3]
AxF3.set_xticklabels(xlabelAxF3,fontdict=fontdict)
ylabelAxF3=list(AxF3.get_yticks())
ylabelAxF3=[int(x) for x in ylabelAxF3]
AxF3.set_yticklabels(ylabelAxF3,fontdict=fontdict)
data.plot(kind='scatter',x='Updated',y='Total Cases',marker='*',\
          color='black',ax=AxF3,fontsize=10)
AxF3.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()
#prediction Total Deaths
Z,Y=FranceII[['Updated','Total Cases']],FranceII[['Total Deaths']]
X=np.array(FranceII[['Updated']])
Grid1=GridSearchCV(PR,parameters1,cv=4)
Grid1.fit(Z,Y)
yhatD=Grid1.predict(Z)
combined = np.hstack((X, yhatD))
FranceDP=pd.DataFrame(combined,columns=['Updated','Total Deaths'])
AxF4=FranceII.plot(x='Updated',y='Total Deaths',color='navy',fontsize=10)
FranceDP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxF4,fontsize=10)
AxF4.set_xlabel('Days',fontsize='medium',weight='bold')
AxF4.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxF4=list(AxF4.get_xticks())
xlabelAxF4=[int(x) for x in xlabelAxF4]
AxF4.set_xticklabels(xlabelAxF4,fontdict=fontdict)
ylabelAxF4=list(AxF4.get_yticks())
ylabelAxF4=[int(x) for x in ylabelAxF4]
AxF4.set_yticklabels(ylabelAxF4,fontdict=fontdict)
FranceDP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxF4,fontsize=10)
AxF4.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()
#####Germany
GermanyII=Germany[Germany['Updated']>QS_Germany]
X,Y=GermanyII[['Updated']],GermanyII[['Total Cases']]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.3)
xtr,ytr=x_train['Updated'],y_train['Total Cases']
xtst,ytst=x_test['Updated'],y_test['Total Cases']
r2_tr,mse_tr,r2_tst,mse_tst,degree=[],[],[],[],[]
xp,yp=GermanyII['Updated'],GermanyII['Total Cases']
for n in range(1,11):
    f=np.polyfit(xtr,ytr,n)
    p=np.poly1d(f)
    yhat_tr,yhat_tst=p(xtr),p(xtst)
    r2_tr.append(r2_score(ytr, yhat_tr))
    mse_tr.append(mean_squared_error(ytr, yhat_tr))
    r2_tst.append(r2_score(ytst, yhat_tst))
    mse_tst.append(mean_squared_error(ytst, yhat_tst))
    degree.append(n)

bestfit=pd.DataFrame([degree,r2_tr,mse_tr,r2_tst,mse_tst]).transpose()
bestfit.columns=['Degree','R2_tr','MSE_tr','R2_tst','MSE_tst']
columns=['MSE_tr','MSE_tst']
bestfitN=pd.DataFrame(columns=columns)
for cl in columns:
	NM = bestfit[cl]
	NM=np.array(NM)
	NMn=normalize([NM])
	NMn=pd.DataFrame(NMn).transpose()
	NMn.columns=[cl]
	bestfitN[cl]=NMn[cl]
bestfitN['Degree']=bestfit['Degree']
#graph
AxG_f3=bestfit.plot(x='Degree',y='R2_tr',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='R2_tst',xlim=(1,10), ylim=(0,1.2),color='red',marker='o',fontsize=10,ax=AxG_f3)
bestfitN.plot(x='Degree',y='MSE_tr',xlim=(1,10), ylim=(0,1.2),color='black',marker='*',fontsize=10,ax=AxG_f3)
bestfitN.plot(x='Degree',y='MSE_tst',xlim=(1,10), ylim=(0,1.2),color='black',marker='o',fontsize=10,ax=AxG_f3)
AxG_f3.legend(('R2_tr','R2_tst','MSE_tr','MSE_tst'),prop=font)
xlabelAxG_f3=list(AxG_f3.get_xticks())
xlabelAxG_f3=[int(x) for x in xlabelAxG_f3]
AxG_f3.set_xticklabels(xlabelAxG_f3,fontdict=fontdict)
ylabelAxG_f3=list(AxG_f3.get_yticks())
ylabelAxG_f3=[int(x) for x in ylabelAxG_f3]
AxG_f3.set_yticklabels(ylabelAxG_f3,fontdict=fontdict)
plt.show()
#prdiction Total Cases
f=np.polyfit(xp,yp,6)
p=np.poly1d(f)
yhat_II=p(xp)
d=np.array([xp,yhat_II]).T
data=pd.DataFrame(d,columns=['Updated','Total Cases'])
AxG3=GermanyII.plot(x='Updated',y='Total Cases',color='gold',fontsize=10)
AxG3.set_xlabel('Days',fontsize='medium',weight='bold')
AxG3.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelAxG3=list(AxG3.get_xticks())
xlabelAxG3=[int(x) for x in xlabelAxG3]
AxG3.set_xticklabels(xlabelAxG3,fontdict=fontdict)
ylabelAxG3=list(AxG3.get_yticks())
ylabelAxG3=[int(x) for x in ylabelAxG3]
AxG3.set_yticklabels(ylabelAxG3,fontdict=fontdict)
data.plot(kind='scatter',x='Updated',y='Total Cases',marker='*',\
          color='black',ax=AxG3,fontsize=10)
AxG3.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()

#prediction Total Deaths
Z,Y=GermanyII[['Updated','Total Cases']],GermanyII[['Total Deaths']]
X=np.array(GermanyII[['Updated']])
Grid1=GridSearchCV(PR,parameters1,cv=4)
Grid1.fit(Z,Y)
yhatD=Grid1.predict(Z)
combined = np.hstack((X, yhatD))
GermanyDP=pd.DataFrame(combined,columns=['Updated','Total Deaths'])
AxG4=GermanyII.plot(x='Updated',y='Total Deaths',color='gold',fontsize=10)
GermanyDP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxG4,fontsize=10)
AxG4.set_xlabel('Days',fontsize='medium',weight='bold')
AxG4.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxG4=list(AxG4.get_xticks())
xlabelAxG4=[int(x) for x in xlabelAxG4]
AxG4.set_xticklabels(xlabelAxG4,fontdict=fontdict)
ylabelAxG4=list(AxG4.get_yticks())
ylabelAxG4=[int(x) for x in ylabelAxG4]
AxG4.set_yticklabels(ylabelAxG4,fontdict=fontdict)
GermanyDP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxG4,fontsize=10)
AxG4.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()
#####Italy
ItalyII=Italy[Italy['Updated']>QS_Italy]
X,Y=ItalyII[['Updated']],ItalyII[['Total Cases']]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.3)
xtr,ytr=x_train['Updated'],y_train['Total Cases']
xtst,ytst=x_test['Updated'],y_test['Total Cases']
r2_tr,mse_tr,r2_tst,mse_tst,degree=[],[],[],[],[]
xp,yp=ItalyII['Updated'],ItalyII['Total Cases']
for n in range(1,11):
    f=np.polyfit(xtr,ytr,n)
    p=np.poly1d(f)
    yhat_tr,yhat_tst=p(xtr),p(xtst)
    r2_tr.append(r2_score(ytr, yhat_tr))
    mse_tr.append(mean_squared_error(ytr, yhat_tr))
    r2_tst.append(r2_score(ytst, yhat_tst))
    mse_tst.append(mean_squared_error(ytst, yhat_tst))
    degree.append(n)

bestfit=pd.DataFrame([degree,r2_tr,mse_tr,r2_tst,mse_tst]).transpose()
bestfit.columns=['Degree','R2_tr','MSE_tr','R2_tst','MSE_tst']
columns=['MSE_tr','MSE_tst']
bestfitN=pd.DataFrame(columns=columns)
for cl in columns:
	NM = bestfit[cl]
	NM=np.array(NM)
	NMn=normalize([NM])
	NMn=pd.DataFrame(NMn).transpose()
	NMn.columns=[cl]
	bestfitN[cl]=NMn[cl]
bestfitN['Degree']=bestfit['Degree']
#graph
AxI_f3=bestfit.plot(x='Degree',y='R2_tr',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='R2_tst',xlim=(1,10), ylim=(0,1.2),color='red',marker='o',fontsize=10,ax=AxI_f3)
bestfitN.plot(x='Degree',y='MSE_tr',xlim=(1,10), ylim=(0,1.2),color='black',marker='*',fontsize=10,ax=AxI_f3)
bestfitN.plot(x='Degree',y='MSE_tst',xlim=(1,10), ylim=(0,1.2),color='black',marker='o',fontsize=10,ax=AxI_f3)
AxI_f3.legend(('R2_tr','R2_tst','MSE_tr','MSE_tst'),prop=font)
xlabelAxI_f3=list(AxI_f3.get_xticks())
xlabelAxI_f3=[int(x) for x in xlabelAxI_f3]
AxI_f3.set_xticklabels(xlabelAxI_f3,fontdict=fontdict)
ylabelAxI_f3=list(AxI_f3.get_yticks())
ylabelAxI_f3=[int(x) for x in ylabelAxI_f3]
AxI_f3.set_yticklabels(ylabelAxI_f3,fontdict=fontdict)
plt.show()
#prdiction Total Cases
f=np.polyfit(xp,yp,7)
p=np.poly1d(f)
yhat_II=p(xp)
d=np.array([xp,yhat_II]).T
data=pd.DataFrame(d,columns=['Updated','Total Cases'])
AxI3=ItalyII.plot(x='Updated',y='Total Cases',color='darkgreen',fontsize=10)
AxI3.set_xlabel('Days',fontsize='medium',weight='bold')
AxI3.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelAxI3=list(AxI3.get_xticks())
xlabelAxI3=[int(x) for x in xlabelAxI3]
AxI3.set_xticklabels(xlabelAxI3,fontdict=fontdict)
ylabelAxI3=list(AxI3.get_yticks())
ylabelAxI3=[int(x) for x in ylabelAxI3]
AxI3.set_yticklabels(ylabelAxI3,fontdict=fontdict)
data.plot(kind='scatter',x='Updated',y='Total Cases',marker='*',\
          color='black',ax=AxI3,fontsize=10)
AxI3.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()

#prediction Total Deaths
Z,Y=ItalyII[['Updated','Total Cases']],ItalyII[['Total Deaths']]
X=np.array(ItalyII[['Updated']])
Grid1=GridSearchCV(PR,parameters1,cv=4)
Grid1.fit(Z,Y)
yhatD=Grid1.predict(Z)
combined = np.hstack((X, yhatD))
ItalyDP=pd.DataFrame(combined,columns=['Updated','Total Deaths'])
AxI4=ItalyII.plot(x='Updated',y='Total Deaths',color='darkgreen',fontsize=10)
ItalyDP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxI4,fontsize=10)
AxI4.set_xlabel('Days',fontsize='medium',weight='bold')
AxI4.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxI4=list(AxI4.get_xticks())
xlabelAxI4=[int(x) for x in xlabelAxI4]
AxI4.set_xticklabels(xlabelAxI4,fontdict=fontdict)
ylabelAxI4=list(AxI4.get_yticks())
ylabelAxI4=[int(x) for x in ylabelAxI4]
AxI4.set_yticklabels(ylabelAxI4,fontdict=fontdict)
ItalyDP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxI4,fontsize=10)
AxI4.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()
#####Japan
JapanII=Japan[Japan['Updated']>QS_Japan]
X,Y=JapanII[['Updated']],JapanII[['Total Cases']]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.3)
xtr,ytr=x_train['Updated'],y_train['Total Cases']
xtst,ytst=x_test['Updated'],y_test['Total Cases']
r2_tr,mse_tr,r2_tst,mse_tst,degree=[],[],[],[],[]
xp,yp=JapanII['Updated'],JapanII['Total Cases']
for n in range(1,11):
    f=np.polyfit(xtr,ytr,n)
    p=np.poly1d(f)
    yhat_tr,yhat_tst=p(xtr),p(xtst)
    r2_tr.append(r2_score(ytr, yhat_tr))
    mse_tr.append(mean_squared_error(ytr, yhat_tr))
    r2_tst.append(r2_score(ytst, yhat_tst))
    mse_tst.append(mean_squared_error(ytst, yhat_tst))
    degree.append(n)

bestfit=pd.DataFrame([degree,r2_tr,mse_tr,r2_tst,mse_tst]).transpose()
bestfit.columns=['Degree','R2_tr','MSE_tr','R2_tst','MSE_tst']
columns=['MSE_tr','MSE_tst']
bestfitN=pd.DataFrame(columns=columns)
for cl in columns:
	NM = bestfit[cl]
	NM=np.array(NM)
	NMn=normalize([NM])
	NMn=pd.DataFrame(NMn).transpose()
	NMn.columns=[cl]
	bestfitN[cl]=NMn[cl]
bestfitN['Degree']=bestfit['Degree']
#graph
AxJ_f3=bestfit.plot(x='Degree',y='R2_tr',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='R2_tst',xlim=(1,10), ylim=(0,1.2),color='red',marker='o',fontsize=10,ax=AxJ_f3)
bestfitN.plot(x='Degree',y='MSE_tr',xlim=(1,10), ylim=(0,1.2),color='black',marker='*',fontsize=10,ax=AxJ_f3)
bestfitN.plot(x='Degree',y='MSE_tst',xlim=(1,10), ylim=(0,1.2),color='black',marker='o',fontsize=10,ax=AxJ_f3)
AxJ_f3.legend(('R2_tr','R2_tst','MSE_tr','MSE_tst'),prop=font)
xlabelAxJ_f3=list(AxJ_f3.get_xticks())
xlabelAxJ_f3=[int(x) for x in xlabelAxJ_f3]
AxJ_f3.set_xticklabels(xlabelAxJ_f3,fontdict=fontdict)
ylabelAxJ_f3=list(AxJ_f3.get_yticks())
ylabelAxJ_f3=[int(x) for x in ylabelAxJ_f3]
AxJ_f3.set_yticklabels(ylabelAxJ_f3,fontdict=fontdict)
plt.show()
#prdiction Total Cases
f=np.polyfit(xp,yp,5)
p=np.poly1d(f)
yhat_II=p(xp)
d=np.array([xp,yhat_II]).T
data=pd.DataFrame(d,columns=['Updated','Total Cases'])
AxJ3=JapanII.plot(x='Updated',y='Total Cases',color='rebeccapurple',fontsize=10)
AxJ3.set_xlabel('Days',fontsize='medium',weight='bold')
AxJ3.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelAxJ3=list(AxJ3.get_xticks())
xlabelAxJ3=[int(x) for x in xlabelAxJ3]
AxJ3.set_xticklabels(xlabelAxJ3,fontdict=fontdict)
ylabelAxJ3=list(AxJ3.get_yticks())
ylabelAxJ3=[int(x) for x in ylabelAxJ3]
AxJ3.set_yticklabels(ylabelAxJ3,fontdict=fontdict)
data.plot(kind='scatter',x='Updated',y='Total Cases',marker='*',\
          color='black',ax=AxJ3,fontsize=10)
AxJ3.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()

#prediction Total Deaths
Z,Y=JapanII[['Updated','Total Cases']],JapanII[['Total Deaths']]
X=np.array(JapanII[['Updated']])
Grid1=GridSearchCV(PR,parameters1,cv=4)
Grid1.fit(Z,Y)
yhatD=Grid1.predict(Z)
combined = np.hstack((X, yhatD))
JapanDP=pd.DataFrame(combined,columns=['Updated','Total Deaths'])
AxJ4=JapanII.plot(x='Updated',y='Total Deaths',color='rebeccapurple',fontsize=10)
JapanDP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxJ4,fontsize=10)
AxJ4.set_xlabel('Days',fontsize='medium',weight='bold')
AxJ4.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxJ4=list(AxJ4.get_xticks())
xlabelAxJ4=[int(x) for x in xlabelAxJ4]
AxJ4.set_xticklabels(xlabelAxJ4,fontdict=fontdict)
ylabelAxJ4=list(AxJ4.get_yticks())
ylabelAxJ4=[int(x) for x in ylabelAxJ4]
AxJ4.set_yticklabels(ylabelAxJ4,fontdict=fontdict)
JapanDP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxJ4,fontsize=10)
AxJ4.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()
###UK
UKII=UK[UK['Updated']>QS_UK]
X,Y=UKII[['Updated']],UKII[['Total Cases']]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.3)
xtr,ytr=x_train['Updated'],y_train['Total Cases']
xtst,ytst=x_test['Updated'],y_test['Total Cases']
r2_tr,mse_tr,r2_tst,mse_tst,degree=[],[],[],[],[]
xp,yp=UKII['Updated'],UKII['Total Cases']
for n in range(1,11):
    f=np.polyfit(xtr,ytr,n)
    p=np.poly1d(f)
    yhat_tr,yhat_tst=p(xtr),p(xtst)
    r2_tr.append(r2_score(ytr, yhat_tr))
    mse_tr.append(mean_squared_error(ytr, yhat_tr))
    r2_tst.append(r2_score(ytst, yhat_tst))
    mse_tst.append(mean_squared_error(ytst, yhat_tst))
    degree.append(n)

bestfit=pd.DataFrame([degree,r2_tr,mse_tr,r2_tst,mse_tst]).transpose()
bestfit.columns=['Degree','R2_tr','MSE_tr','R2_tst','MSE_tst']
columns=['MSE_tr','MSE_tst']
bestfitN=pd.DataFrame(columns=columns)
for cl in columns:
	NM = bestfit[cl]
	NM=np.array(NM)
	NMn=normalize([NM])
	NMn=pd.DataFrame(NMn).transpose()
	NMn.columns=[cl]
	bestfitN[cl]=NMn[cl]
bestfitN['Degree']=bestfit['Degree']
#graph
AxK_f3=bestfit.plot(x='Degree',y='R2_tr',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='R2_tst',xlim=(1,10), ylim=(0,1.2),color='red',marker='o',fontsize=10,ax=AxK_f3)
bestfitN.plot(x='Degree',y='MSE_tr',xlim=(1,10), ylim=(0,1.2),color='black',marker='*',fontsize=10,ax=AxK_f3)
bestfitN.plot(x='Degree',y='MSE_tst',xlim=(1,10), ylim=(0,1.2),color='black',marker='o',fontsize=10,ax=AxK_f3)
AxK_f3.legend(('R2_tr','R2_tst','MSE_tr','MSE_tst'),prop=font)
xlabelAxK_f3=list(AxK_f3.get_xticks())
xlabelAxK_f3=[int(x) for x in xlabelAxK_f3]
AxK_f3.set_xticklabels(xlabelAxK_f3,fontdict=fontdict)
ylabelAxK_f3=list(AxK_f3.get_yticks())
ylabelAxK_f3=[int(x) for x in ylabelAxK_f3]
AxK_f3.set_yticklabels(ylabelAxK_f3,fontdict=fontdict)
plt.show()
#prdiction Total Cases
f=np.polyfit(xp,yp,7)
p=np.poly1d(f)
yhat_II=p(xp)
d=np.array([xp,yhat_II]).T
data=pd.DataFrame(d,columns=['Updated','Total Cases'])
AxK3=UKII.plot(x='Updated',y='Total Cases',color='red',fontsize=10)
AxK3.set_xlabel('Days',fontsize='medium',weight='bold')
AxK3.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelAxK3=list(AxK3.get_xticks())
xlabelAxK3=[int(x) for x in xlabelAxK3]
AxK3.set_xticklabels(xlabelAxK3,fontdict=fontdict)
ylabelAxK3=list(AxK3.get_yticks())
ylabelAxK3=[int(x) for x in ylabelAxK3]
AxK3.set_yticklabels(ylabelAxK3,fontdict=fontdict)
data.plot(kind='scatter',x='Updated',y='Total Cases',marker='*',\
          color='black',ax=AxK3,fontsize=10)
AxK3.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()

#prediction Total Deaths
Z,Y=UKII[['Updated','Total Cases']],UKII[['Total Deaths']]
X=np.array(UKII[['Updated']])
Grid1=GridSearchCV(PR,parameters1,cv=4)
Grid1.fit(Z,Y)
yhatD=Grid1.predict(Z)
combined = np.hstack((X, yhatD))
UKDP=pd.DataFrame(combined,columns=['Updated','Total Deaths'])
AxK4=UKII.plot(x='Updated',y='Total Deaths',color='red',fontsize=10)
UKDP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxK4,fontsize=10)
AxK4.set_xlabel('Days',fontsize='medium',weight='bold')
AxK4.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxK4=list(AxK4.get_xticks())
xlabelAxK4=[int(x) for x in xlabelAxK4]
AxK4.set_xticklabels(xlabelAxK4,fontdict=fontdict)
ylabelAxK4=list(AxK4.get_yticks())
ylabelAxK4=[int(x) for x in ylabelAxK4]
AxK4.set_yticklabels(ylabelAxK4,fontdict=fontdict)
UKDP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxK4,fontsize=10)
AxK4.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()
####USA
USAII=USA[USA['Updated']>QS_USA]
X,Y=USAII[['Updated']],USAII[['Total Cases']]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.3)
xtr,ytr=x_train['Updated'],y_train['Total Cases']
xtst,ytst=x_test['Updated'],y_test['Total Cases']
r2_tr,mse_tr,r2_tst,mse_tst,degree=[],[],[],[],[]
xp,yp=USAII['Updated'],USAII['Total Cases']
for n in range(1,11):
    f=np.polyfit(xtr,ytr,n)
    p=np.poly1d(f)
    yhat_tr,yhat_tst=p(xtr),p(xtst)
    r2_tr.append(r2_score(ytr, yhat_tr))
    mse_tr.append(mean_squared_error(ytr, yhat_tr))
    r2_tst.append(r2_score(ytst, yhat_tst))
    mse_tst.append(mean_squared_error(ytst, yhat_tst))
    degree.append(n)

bestfit=pd.DataFrame([degree,r2_tr,mse_tr,r2_tst,mse_tst]).transpose()
bestfit.columns=['Degree','R2_tr','MSE_tr','R2_tst','MSE_tst']
columns=['MSE_tr','MSE_tst']
bestfitN=pd.DataFrame(columns=columns)
for cl in columns:
	NM = bestfit[cl]
	NM=np.array(NM)
	NMn=normalize([NM])
	NMn=pd.DataFrame(NMn).transpose()
	NMn.columns=[cl]
	bestfitN[cl]=NMn[cl]
bestfitN['Degree']=bestfit['Degree']
#graph
AxA_f3=bestfit.plot(x='Degree',y='R2_tr',xlim=(1,10), ylim=(0,1.2),color='red',marker='*',fontsize=10)
bestfit.plot(x='Degree',y='R2_tst',xlim=(1,10), ylim=(0,1.2),color='red',marker='o',fontsize=10,ax=AxA_f3)
bestfitN.plot(x='Degree',y='MSE_tr',xlim=(1,10), ylim=(0,1.2),color='black',marker='*',fontsize=10,ax=AxA_f3)
bestfitN.plot(x='Degree',y='MSE_tst',xlim=(1,10), ylim=(0,1.2),color='black',marker='o',fontsize=10,ax=AxA_f3)
AxA_f3.legend(('R2_tr','R2_tst','MSE_tr','MSE_tst'),prop=font)
xlabelAxA_f3=list(AxA_f3.get_xticks())
xlabelAxA_f3=[int(x) for x in xlabelAxA_f3]
AxA_f3.set_xticklabels(xlabelAxA_f3,fontdict=fontdict)
ylabelAxA_f3=list(AxA_f3.get_yticks())
ylabelAxA_f3=[int(x) for x in ylabelAxA_f3]
AxA_f3.set_yticklabels(ylabelAxA_f3,fontdict=fontdict)
plt.show()
#prdiction Total Cases
f=np.polyfit(xp,yp,6)
p=np.poly1d(f)
yhat_II=p(xp)
d=np.array([xp,yhat_II]).T
data=pd.DataFrame(d,columns=['Updated','Total Cases'])
AxA3=USAII.plot(x='Updated',y='Total Cases',color='blue',fontsize=10)
AxA3.set_xlabel('Days',fontsize='medium',weight='bold')
AxA3.set_ylabel('Total Cases',fontsize='medium',weight='bold')
xlabelAxA3=list(AxA3.get_xticks())
xlabelAxA3=[int(x) for x in xlabelAxA3]
AxA3.set_xticklabels(xlabelAxA3,fontdict=fontdict)
ylabelAxA3=list(AxA3.get_yticks())
ylabelAxA3=[int(x) for x in ylabelAxA3]
AxA3.set_yticklabels(ylabelAxA3,fontdict=fontdict)
data.plot(kind='scatter',x='Updated',y='Total Cases',marker='*',\
          color='black',ax=AxA3,fontsize=10)
AxA3.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()

#prediction Total Deaths
Z,Y=USAII[['Updated','Total Cases']],USAII[['Total Deaths']]
X=np.array(USAII[['Updated']])
Grid1=GridSearchCV(PR,parameters1,cv=4)
Grid1.fit(Z,Y)
yhatD=Grid1.predict(Z)
combined = np.hstack((X, yhatD))
USADP=pd.DataFrame(combined,columns=['Updated','Total Deaths'])
AxA4=USAII.plot(x='Updated',y='Total Deaths',color='blue',fontsize=10)
USADP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxA4,fontsize=10)
AxA4.set_xlabel('Days',fontsize='medium',weight='bold')
AxA4.set_ylabel('Total Deaths',fontsize='medium',weight='bold')
xlabelAxA4=list(AxA4.get_xticks())
xlabelAxA4=[int(x) for x in xlabelAxA4]
AxA4.set_xticklabels(xlabelAxA4,fontdict=fontdict)
ylabelAxA4=list(AxA4.get_yticks())
ylabelAxA4=[int(x) for x in ylabelAxA4]
AxA4.set_yticklabels(ylabelAxA4,fontdict=fontdict)
USADP.plot(kind='scatter',x='Updated',y='Total Deaths',marker='*',\
          color='black',ax=AxA4,fontsize=10)
AxA4.legend(('Actual','Predicted'),prop=font,loc='upper left')
plt.show()

