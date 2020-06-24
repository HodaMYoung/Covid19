"""The purpose of this code is to extract data related to Coronavirus published by
World Health Organization(WHO).The WHO publishes PDF version of situation reports to
update the international community about the current status of Covid-19 around the globe.
Two libraries used here to scrape data from PDF documents were tabula and PyPDF2. Since the
features of reports and their corresponding tables were developed gradually during
the course of pandemic, two different Python codes were used to scrape data from PDFs.
For instance, China related Coronavirus information was presented in a separate table than
the rest of affected countries for the first series of reports. This code only covers the
first 38 reports.

Source:
https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports"""

#importing libraries
import pandas as pd#to manipulate DataFrames
import numpy as np#to work with  multidimensional arrays
import tabula#to extract tables from PDF
import PyPDF2#to extract contents of a PDF

"""The names of tables' columns as well as number of columns have modified several times
in the situation reports. These following lists represent the columns' names in the
tables."""

NAMES=['Office','Unnamed: 0','WHO Regional Office']
colrN1=['Country','Total Cases','New Cases','Total Deaths','New Deaths',\
           'Transmission','Updated','First Case Date','Report Date','Last Case Date']
colrN2=['Country', 'Total Cases', 'New Cases', 'Total Deaths',\
       'New Deaths', 'Transmission', 'Updated', 'First Case Date',\
       'Report Date','Last Case Date','Total Travel', 'New Travel','Total Local',\
       'New Local', 'Total Investigation', 'New Investigation']
colrN3=['Country', 'Total Cases', 'New Cases', 'Total Deaths',\
       'New Deaths', 'Transmission', 'Updated', 'First Case Date',\
       'Report Date','Last Case Date','Total Travel', 'New Travel','Total Local',\
       'New Local', 'Total Investigation', 'New Investigation','Total Outside China',\
        'New Outside China']

"""The datafrme dfR contain the page numbers and areas of tables for each report.
Where 'NA' indicates that no separate table rather than the main table was available.
In addition,often the information about deaths and cases was not explicitly presented
in the tables for the early reports, therefore these statistics were assigned to variables
like deaths and UAE_ir."""

dfR=pd.DataFrame({'P0':['NA','NA','NA','NA','NA','NA','NA','NA','NA','NA','NA',\
                        3,3,3,2,3,3,2,3,3,2,3,3,4,3,2,3,3,3,4,3,3,3,2,3,2,2,3],\
                  'A0':['NA','NA','NA','NA','NA','NA','NA','NA','NA','NA','NA',\
                        (120,0,680,600),(120,0,680,600),(120,0,680,600),(120,0,680,600),\
                        (120,0,700,600),(120,0,680,600),(240,0,800,500),\
                        (110,0,680,500),(110,0,620,500),(110,0,620,500),\
                        (110,0,620,500),(110,0,700,500),(110,0,700,500),\
                        (160,0,700,600),(145,0,700,600),(145,0,700,600),\
                        (165,0,710,600),(165,0,710,600),(165,0,710,600),\
                        (165,0,710,600),(165,0,710,600),(160,0,700,600),\
                        (170,0,720,600),(155,0,720,600),(155,0,700,600),\
                        (150,0,690,600),(150,0,690,600)],\
                  'P1':[2,2,3,3,3,3,3,4,3,3,3,4,4,4,3,4,4,3,4,4,3,4,4,5,4,3,4,4,4,\
                        5,4,[4,4],[4],[3],[4,4],[3,3],[3,3],[4,5,5]],\
                  'A1':[(220,0,350,600),(200,0,650,600),\
                        (180,0,900,600),(180,0,700,600),\
                        (170,0,700,600),(210,0,670,600),\
                        (120,0,560,600),(100,0,500,600),\
                        (100,0,520,600),(120,0,620,600),\
                        (120,0,620,600),(80,0,680,600),\
                        (80,0,680,600),(180,0,660,900),(180,0,660,900),\
                        (180,0,680,900),(160,0,660,900),(160,0,660,900),\
                        (160,0,660,900),(160,0,660,900),(160,0,660,900),\
                        (160,0,660,900),(160,0,660,900),(160,0,660,900),\
                        (160,0,660,900),(160,0,660,900),(160,0,660,900),\
                        (160,0,660,900),(160,0,660,900),(160,0,730,900),\
                        (160,0,780,900),[(120,0,700,900),(680,0,720,900)],\
                        [(120, 0, 750, 900)],[(120, 0, 750, 900)],\
                        [(120, 0, 600, 900),(610, 0, 640, 900)],\
                        [(120, 0, 650, 900),(670, 0, 700, 900)],\
                        [(140,0,710,900),(710,0,740,900)],\
                        [(120,0,800,900),(0,0,170,900),(145,0,200,900)]]})

deaths=[6,17,25,41,56]
UAE_ir=['5','5','0','0','0']
#main part
""" The PDF reports were read through a loop where the paths of new reoprt,i.e. path,
and output .csv file extracted from pdf,i.e. pathS are updated in each iteration.
In order to extract table with tabula module following command can be used:
tabula.read_pdf(path,pages=P,area=A)
P: A numeric string or list
A: tuple or list with four elements,i.e. [A,B,C,D].Where, A and B represent the distance
from top and left corner of the page, respectively. C and D are width and length of the
area to be extracted. These information can be achieved by either trial and error method
or through Adobe Acrobat Tools.
It should be noted that tabula library allows the conversion of pdf document to other
formats including csv by .convert_into(path, output_filename, output_format=, pages=)
attribute; nevertheless, the outcomes can be poor.For further information about tabula
library https://pypi.org/project/tabula-py/ is recommended.Another method used here to
scrape a PDF document was opening the file as an object file and then using PyPDF2 module.

It should be noted that after extracting data from the PDF document and storing them in a
new dataframe, post processing is required to deal with missing data as well as format
consistency of datasets. The early WHO reports failed to keep consistency in presenting
countries' names. For instance, the name of a country like the UK were presented in
various formats including, the United Kingdom, The United Kingdom or simply United Kingdom.
Additionally, extra spaces and characters were spotted that must have been removed."""

pathb='covid-19_r'
pathr='CRWHO'
paths=pathr
for n in range(1,39):
    m=n-1#the report number starts from 1, while the dfR index starts from 0
    path=pathb+str(n)+'.pdf'#updating the report path 
    p0,a0=dfR.loc[m,'P0'],dfR.loc[m,'A0']#reading the area and pages
    p1,a1=dfR.loc[m,'P1'],dfR.loc[m,'A1']#reading the area and pages
    #Report Date
    """Following snippet is used to extract the publication date of report in
       PDF documents."""
    if n<7:
       dfRD=tabula.read_pdf(path,pages=1)
       cl=dfRD.columns
       cl1=cl[0].split('\r')
       cl1 = list(map(lambda st: str.replace(st, " ", "-"), cl1)) [-1]
    else:
        pdfFileObj = open(path, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        page=pdfReader.getPage(0)
        cl=page.extractText()
        cl=cl.replace('\n','')
        if n<21:
           expression,ewdth='reported by ',15
        elif n<33:
             expression,ewdth='reported by ',16
        elif n<79:
           expression,ewdth='CET ',16
        elif n<87:
            expression,ewdth='CET,',16
        elif n<102:
            expression,ewdth='CEST,',16
        else:
            expression,ewdth='CEST,',12
        clndx=cl.find(expression)
        cl1=cl[clndx+len(expression):clndx+len(expression)+ewdth]
        cl1=cl1.split()
        if n>59 and n<63:
           cl1[0]=str(int(cl1[0])+1)
        cl1=[x.capitalize() for x in cl1]
        cl1=('-').join(cl1)
        if n==24:
           ndx_death=cl.find('deaths')
           cl2=cl[ndx_death-5:ndx_death-1]
       
    if p0!='NA':
       df_1N=tabula.read_pdf(path,pages=str(p0),area=a0)
       df_1N.dropna(axis=1,inplace=True)
       if n<23:
          df_1N.columns=['Province','Total Cases']
       if n==23:
           df_1N.columns=['Province','Population(in 10,000s)',\
                          'Total Cases','Suspect','Total Deaths']
       if n==24:
           df_1N.columns=['Province','Population(in 10,000s)','Total Cases']
       if n>24 and n<28:
           df_1N.columns=['Province','Population(in 10,000s)','Daily Confirmed',\
                          'Daily Clinically','Daily Cases',\
                          'Daily Suspected Cases','Daily Deaths',\
                          'Total Cases','Total Clinically','Total',\
                          'Total Deaths']
       if n>27:
          df_1N.columns=['Province','Population(in 10,000s)','Daily Confirmed',\
                          'Daily Cases','Daily Deaths',\
                          'Total Cases','Total Deaths']
    
       df_1N['Total Cases'].replace(' ','',regex=True,inplace=True)
       if n==33:
          df_1N['Total Cases']=[x[0:len(x)-1] if x[-1]=='*' else x for x in df_1N['Total Cases']]
       df_1N['Total Cases']=list(map(int,df_1N['Total Cases']))
       if n!=27:
          expr='Total'
       else:
           expr='Totals'
       ndx_total=df_1N[df_1N['Province']==expr].index.to_list()
       if len(ndx_total)>0:
          df_1N.drop(index=[ndx_total[0]],inplace=True)
       if n>24 and n<30:
           dfr0=pd.DataFrame({'Country':['China'],\
                              'Cases':[str(df_1N['Total Cases'].sum())+' ('+\
                                       str(df_1N['Daily Confirmed'].sum())+')'],\
                              'Travel':['0 (0)'],'Local':['0 (0)'],\
                              'Investigation':['0 (0)'],\
                              'Death':[str(df_1N['Total Deaths'].sum())+\
                                       ' ('+str(df_1N['Daily Deaths'].sum())+')']})
       if n>=30:
          dfr0=pd.DataFrame({'Country':['China'],\
                             'Cases':[str(df_1N['Total Cases'].sum())+' ('+\
                                       str(df_1N['Daily Confirmed'].sum())+')'],\
                             'Travel':['0 (0)'],'Outside China':['0 (0)'],\
                             'Local':['0 (0)'],'Investigation':['0 (0)'],\
                             'Death':[str(df_1N['Total Deaths'].sum())+\
                                       ' ('+str(df_1N['Daily Deaths'].sum())+')']})
           
       pathST=paths+str(n)+'_T1.csv'
       df_1N.to_csv(pathST)
    if n>31:
       dfb=pd.DataFrame(columns=['0','1','2','3','4','5','6'])
       for prt in range(0,len(p1)):
           dfM=tabula.read_pdf(path,pages=str(p1[prt]),area=a1[prt])
           if len(dfM.columns)>len(dfb.columns):
              dfM.dropna(axis=1,how='all',inplace=True)
              if n==37 and prt==0:
                 dfM.drop(columns=['Unnamed: 2'],inplace=True)
           if n==38 and prt==1:
                 dfM.drop(index=[dfM.index[-1]],inplace=True)
           if prt==len(p1)-1 and n>34 and len(dfM)>1:
              dfM.iloc[:,0].fillna('',inplace=True)
              dfM.iloc[0,0]=dfM.iloc[0,0]+dfM.iloc[1,0]
              dfM.drop(index=[1],inplace=True)
           dfM.columns=dfb.columns
           ndx_subttl=dfM[dfM['0']==('Subtotal for all regions')].index.to_list()
           if len(ndx_subttl)>0:
               dfM.drop(index=[ndx_subttl[0]],inplace=True)
               dfM.reset_index(drop=True,inplace=True)
           dfb=pd.concat([dfb,dfM],axis=0)
       df=dfb
    else:
       df=tabula.read_pdf(path,pages=str(p1),area=a1)
       
       clmns=df.columns
       ndx_uae=df[df[clmns[0]]=='United Arab Emirates'].index.to_list()
       if len(ndx_uae)>0:
          sub=list(df.iloc[ndx_uae[0],0:len(clmns)-1])
          df.iloc[ndx_uae[0],1:]=sub
    if n<30:   
       df.drop(columns=[clmns[0]],inplace=True)
       
    if len(df.columns)<3 and n<5:
       df.rename(columns={clmns[-1]:'Total Cases'},inplace=True)
       cp=df[clmns[1]].str.split(' – ',n=1,expand=True)
       df['Country']=cp[0]
       df['Province']=cp[1]
       df.drop(columns=[clmns[1]], inplace=True)
       df.dropna(thresh=2,inplace=True)
       df=df.iloc[:,[1,2,0]]

       
    else:
       df.dropna(how='all',axis=1,inplace=True)
       if n==5:
          df.dropna(thresh=2,axis=1,inplace=True)
       if n<5:
          df.columns=['Country','Province','Total Cases']
       elif n<14:
             df.columns=['Country','Total Cases']
       elif n<30:
             df.columns=['Country','Cases','Travel','Local','Investigation','Death']
       else:
             df.columns=['Country','Cases','Travel','Outside China',\
                         'Local','Investigation','Death']

    if n<5:
       df.columns=['Country','Province','Total Cases']
       df['Province'].fillna('Unknown', inplace=True)
       ndxHK=df[df['Province'].str.startswith('Hong')].index.to_list()
       if len(ndxHK)>0:
           df.iloc[ndxHK[0],2]=df.iloc[ndxHK[0]+1,1]
       df['Province'].replace([' Province',' Municipality',' Special Administrative',' Region'],\
			    ['','','',''],regex=True,inplace=True)
       
       df=df[['Country','Province','Total Cases']]
       df.columns=['Country','Province','Total Cases']
       ndxCHN=df[df['Country']=='China'].index.to_list()
       if len(ndxCHN)>0 and n>1:
          df.drop(index=ndxCHN[0],inplace=True)
       df.reset_index(drop=True,inplace=True)
       ndxCHNUN=df[df['Province'].str.startswith('Unspecified')].index.to_list()
       if len(ndxCHNUN)>0:
          df.iloc[ndxCHNUN[0],1]='Unknown'
       df['Country'].fillna('China',inplace=True)
       df.dropna(inplace=True)
       df.reset_index(inplace=True,drop=True)
       df['Total Cases']=list(map(int,df['Total Cases']))
       ndxttl=df[df['Country']=='Total'].index.to_list()
       if len(ndxttl)>0:
          df.drop(index=ndxttl[0],inplace=True)
       df.reset_index(drop=True,inplace=True)
       dfm=df[['Country','Province','Total Cases']]
       pathST=paths+str(n)+'_T1.csv'
       dfm.to_csv(pathST)
       df.drop(columns=['Province'],inplace=True)
       df=df.groupby(by=['Country'])['Total Cases'].sum()
       df=df.reset_index()
    elif n<14:
        ndxNPL=df[df['Country']=='Federal Democratic Republic of'].index.to_list()
        if len(ndxNPL)>0:
            df.iloc[ndxNPL[0],-1]=df.iloc[ndxNPL[0]+1,-1]
            df.iloc[ndxNPL[0],0]='Nepal'
        ndxFR=df[df['Country']=='French Republic'].index.to_list()
        if len(ndxFR)>0:
            df.iloc[ndxFR[0],0]='France'
        df.dropna(inplace=True)
        df.reset_index(drop=True,inplace=True)
        COLUMNS=df.columns
        for COLUMN in COLUMNS:
            df[COLUMN]=list(map(str,df[COLUMN]))
            df[COLUMN]=[x[0:len(x)-1] if x[-1]=='*' else x for x in df[COLUMN]]
        df['Total Cases'].replace(',','',regex=True,inplace=True)                  
        df['Total Cases']=list(map(float,df['Total Cases']))
        df['Total Cases']=list(map(round,df['Total Cases']))
    else:
        df.replace(['\r'],[' '],regex=True,inplace=True)
        df.replace(['The United Kingdom','the The United Kingdom',],\
                   ['United Kingdom','United Kingdom'],\
                   regex=True,inplace=True)
        df.replace(['‡','§','†'],['','',''],regex=True,inplace=True)
        if n==36:
            ndxUKs=df[df['Country']=='United Kingdom*'].index.to_list()
            df.iloc[ndxUKs[0],0]='United Kingdom'
        df['Country'].fillna('',inplace=True)
        df['Death'].fillna(0,inplace=True)
        ndxUS=df[df['Country']=='United States of'].index.to_list()
        if len(ndxUS)>0:
            df.iloc[ndxUS[0],0]='United States of America'
            sub_r=list(df.iloc[ndxUS[0]+1,:])
            if sub_r[0]=='America' or n==16 or n==32:
               df.iloc[ndxUS[0],1:]=sub_r[1:]
            else:
                df.iloc[ndxUS[0],1:]=sub_r[0:len(sub_r)-1]
                
            
            df.iloc[ndxUS[0],:].fillna('0', inplace=True)
            df.drop(index=[ndxUS[0]+1],inplace=True)
            df.reset_index(drop=True,inplace=True)
        ndxUAE=df[df['Country']=='United Arab'].index.to_list()
        if len(ndxUAE)>0:
            df.iloc[ndxUAE[0],0]='United Arab Emirates'
            for UAE in range(1,len(df.columns)):
                if n==15:
                   df.iloc[ndxUAE[0],UAE]=UAE_ir[UAE-1]
                elif n==17:
                     df.iloc[ndxUAE[0],-1]=0
                else:
                   df.iloc[ndxUAE[0],UAE]=df.iloc[ndxUAE[0]+1,UAE-1]
       
        if n<32:
           ndx_chs=df[df['Country'].str.startswith('China')].index.to_list() 
        else:
            ndx_chs=df[df['Country']=='China'].index.to_list()
        if len(ndx_chs)>0:
           df.iloc[ndx_chs[0],0]='China'
           if n==24:
              df.iloc[ndx_chs[0],-1]=cl2 
           if n==14:
              df.iloc[ndx_chs[0],1]=df.iloc[ndx_chs[0]-1,1]+' '+df.iloc[ndx_chs[0]+1,1]
              df.iloc[ndx_chs[0],-1]=df.iloc[ndx_chs[0],-2]
              df.iloc[ndx_chs[0],2],df.iloc[ndx_chs[0],3],df.iloc[ndx_chs[0],4]=0,0,0
           else:
              df.iloc[ndx_chs[0],2],df.iloc[ndx_chs[0],3],df.iloc[ndx_chs[0],4]=0,0,0
    if n>24:
       df=pd.concat([dfr0,df],axis=0)
        
    if n>14 and n<30:
       df['Cases'].replace([' '],[''],regex=True,inplace=True)
    ndxttl=df[df['Country']=='Total'].index.to_list()
    if len(ndxttl)>0:
       df.drop(index=ndxttl[0],inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)
    if n>16:
       df.iloc[-1,0]='Diamond Princess'
    df.replace('Republic of Singapore','Singapore',regex=True,inplace=True)
    df.replace(' SAR','',regex=True,inplace=True)
   
    if n==1:
       dt,p,FD=1,[],[]
       df['New Cases'],df['Total Deaths'],df['New Deaths']=0,0,0
       df['Transmission'],df['Updated']='Imported Cases only',n
       df.loc[0,'Transmission']='Local transmission'
       df.loc[0,'Total Deaths']=6
       pdfFileObj = open(path, 'rb')
       pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
       page=pdfReader.getPage(0)
       cl=page.extractText()
       cl=cl.replace('\n','')
       while dt>0:
             ndxcd=cl.find('On')
             if ndxcd!=-1:
                cl=cl[ndxcd:]
                ndxcd1=cl.find('.')
                if ndxcd1!=-1:
                   p.append(cl[:ndxcd1+1])
                   cl=cl[ndxcd1+1:]
              
             else:
                dt=-1
       del(p[1:3])
       for n_fd in range(len(p)):
            FD.append(('-').join(p[n_fd][2:p[n_fd].find(',')].split()))
       FD_order=[FD[0],FD[-2],FD[-1],FD[1]]
       df['First Case Date']=FD_order
       df['Report Date'],df['Last Case Date']=cl1,0
       df.loc[0,'Transmission']='Local transmission'
        
    elif n<14:
        ndx_china=df[df['Country']=='China'].index.to_list()
        ndx_philippines=df[df['Country']=='Philippines'].index.to_list()
        if n<7:
           df.loc[ndx_china[0],'Total Deaths']=deaths[n-2]
        else:
            ndxDeath=cl.find('deaths')
            df.loc[ndx_china[0],'Total Deaths']=int(cl[ndxDeath-4:ndxDeath])
            
        if n==13:
           df.loc[ndx_philippines[0],'Total Deaths']=1
    else:
        
        df.dropna(inplace=True)
        noc=df['Cases'].str.split('(', n=1, expand=True)
        df['Total Cases']=noc[0]
        df['New Cases']=noc[1]
        nt=df['Travel'].str.split('(', n=1, expand=True)
        df['Total Travel']=nt[0]
        df['New Travel']=nt[1]
        nl=df['Local'].str.split('(', n=1, expand=True)
        df['Total Local']=nl[0]
        df['New Local']=nl[1]
        ni=df['Investigation'].str.split('(', n=1, expand=True)
        df['Total Investigation']=ni[0]
        df['New Investigation']=ni[1]
        df['Death']=list(map(str,df['Death']))
        nd=df['Death'].str.split('(', n=1, expand=True)
        df['Total Deaths']=nd[0]
        df.fillna(0,inplace=True)
        df.drop(columns=['Cases','Travel','Local','Investigation','Death'],inplace=True)
        if n>29:
           nOTC=df['Outside China'].str.split('(', n=1, expand=True)
           df['Total Outside China']=nOTC[0]
           df['New Outside China']=nOTC[1]
           df.drop(columns=['Outside China'],inplace=True)
        df['Total Cases']=list(map(str,df['Total Cases']))
        df['Total Cases']=[x[1:] if x[0]==' ' else x for x in df['Total Cases']]
        df['Total Cases']=[x[0:len(x)-1] if x[-1]==' ' else x for x in df['Total Cases']]
        if n>16 and n<28:
           df['Total Cases']=[x[0:len(x)-2] if x[-1]=='*' else x for x in df['Total Cases']]
        df['Total Cases']=list(map(int,df['Total Cases']))
        df['Total Travel']=list(map(int,df['Total Travel']))
        df['Total Local']=list(map(str,df['Total Local']))
        if n>18 and n<30:
            df['Total Local']=[x[0:len(x)-1] if x[-1]==' ' else x for x in df['Total Local']]
            if n<27:
               trl=3
            else:
                trl=2
            df['Total Local']=[x[0:len(x)-trl] if x[-1]=='*' else x for x in df['Total Local']]
        
        df['Total Local']=list(map(int,df['Total Local']))
        df['New Local']=list(map(str,df['New Local']))
        df['New Local']=[x[0:len(x)-1] if x[-1]==')' else x for x in df['New Local']]
        if n==18:
           df['New Local']=[x[0:len(x)-3] if x[-1]=='*' else x for x in df['New Local']]
        df['New Local']=list(map(int,df['New Local']))
        df['New Travel']=list(map(str,df['New Travel']))
        df['New Travel']=[x[0:len(x)-1] if x[-1]==')' else x for x in df['New Travel']]
        df['New Travel']=list(map(int,df['New Travel']))
        ndx= df[df['Total Investigation']==''].index.to_list()
        df.loc[ndx,'Total Investigation']=0
        df['Total Investigation']=list(map(int,df['Total Investigation']))
        df['New Investigation']=list(map(str,df['New Investigation']))
        df['New Investigation']=[x[0:len(x)-1] if x[-1]==')' else x for x in df['New Investigation']]
        df['New Investigation']=list(map(int,df['New Investigation']))
        df['Total Deaths']=list(map(float,df['Total Deaths']))
        if n>29:
           df['Total Outside China']=list(map(int,df['Total Outside China']))
           df['New Outside China']=list(map(str,df['New Outside China']))
           df['New Outside China']=[x[0:len(x)-1] if x[-1]==')' else x for x in df['New Outside China']]
           df['New Outside China']=list(map(int,df['New Outside China']))
           
        df.reset_index(drop=True,inplace=True)
        
            
    df['Total Deaths'].fillna(0,inplace=True)
    if n!=1:
       pathR=pathr+str(n-1)+'.csv'
       df0=pd.read_csv(pathR)
       df0.drop(columns='Unnamed: 0', inplace=True)
       df0.rename(columns={'Total Cases':'TC','Total Deaths':'TD'},inplace=True)
       dfM=df.merge(df0[['Country','Transmission','TC','TD',\
                      'First Case Date','Last Case Date']],how='outer')
       dfM['New Cases']=dfM['Total Cases']-dfM['TC']
       dfM['New Deaths']=dfM['Total Deaths']-dfM['TD']
       dfM.drop(columns=['TC','TD'], inplace=True)
       dfM['Report Date']=cl1
       dfM['Updated']=n
       dfM['First Case Date'].fillna(cl1,inplace=True)
       dfM['New Cases'].fillna(0,inplace=True)
       dfM['New Deaths'].fillna(0,inplace=True)
       df=dfM
    if n<14:
        colrN=colrN1
    elif n<30:
        colrN=colrN2
        ndxt=df[(df['New Local']>0) |(df['Total Local']>0)].index.to_list()
        df.loc[ndxt,'Transmission']='Local transmission'
    else:
        colrN=colrN3
        ndxt=df[(df['New Local']>0) |(df['Total Local']>0)].index.to_list()
        df.loc[ndxt,'Transmission']='Local transmission'
    df=df[colrN]
    df['Transmission'].fillna('Imported cases only',inplace=True)
    df['Last Case Date'].fillna(0,inplace=True)
    df.replace(['December', 'January','JANUARY','February','March','April'],\
                   ['12','01','01','02','03','04'],regex=True,inplace=True)
       
    pathS=paths+str(n)+'.csv'
    df.to_csv(pathS)

           
             
                
          
       
       


   
    
