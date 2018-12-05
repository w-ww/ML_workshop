
from __future__ import print_function
import json
from pprint import pprint
from IPython.display import Image
import pydot
import itertools
import numpy as np
import matplotlib.pyplot as plt


from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import re
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from hana_ml import dataframe
import pandas as pd
from hdbcli import dbapi
import configparser

def dt_model_to_dot(modelx):    
    dot = pydot.Dot()
    dot.set('rankdir', 'LR')
    dot.set('concentrate', True)
    #dot.set_node_defaults(shape='record')
    
    #create map(ID,Rules)
    dataDict = modelx['DataDictionary']
    transform = modelx['Transformation']
    tree = modelx['TreeModel']
    idName = {}
    for elem in dataDict:
        idName[elem['ID']] = elem['Name']
    idTrans = {}
    for elem in transform:
        tempMap = {}
        try:
            tempDrivedValues = elem['DerivedValues']
            tempOriginValues = elem['OriginValues']
            tempMap=dict(zip(tempDrivedValues,tempOriginValues))
            idTrans[elem['ID']] = tempMap
        except Exception:
            pass
        try:
            tempDrivedValues = [0,1]
            tempOriginValues = ['<= '+str(elem['Intervals'][0]),'> '+str(elem['Intervals'][0])]
            tempMap=dict(zip(tempDrivedValues,tempOriginValues))
            idTrans[elem['ID']] = tempMap
        except Exception:
            pass
    
    nextLayerHasElement = len(tree['Child'])>0  
    node=pydot.Node('root')
    node.set_label('root')
    dot.add_node(node) 
    currentLayer = tree['Child']
    layerId = 0 
    label = 'NULL'
    label2 = 'NULL'
    label3 = 'NULL'
    while nextLayerHasElement:        
        nextLayer = []
        for elem in currentLayer:
            tempNodeList = []
            #save the next layer elem into list
            try:
                nextLayer.extend(elem['Child'])
                tempNodeList = elem['Child']
            except Exception:
                pass
            #add node in next layer into dot
            currentnodeid = id(elem)

            if layerId ==0:
                node=pydot.Node(currentnodeid)
                #label = str(elem['TreeNodePredicate']['FieldPredID'])+'='+str(elem['TreeNodePredicate']['FieldValue'])+'\n'+ str(elem['Score'])
                label = 'RecordCount: '+str(elem['RecordCount'])+'\nScore: '+ str(elem['Score'])+'\nScoreDistribution: '+ str([round(unit, 2) for unit in elem['ScoreDistribution'][1]])     
                node.set_label(label)
                dot.add_node(node) 
                edge = pydot.Edge('root',currentnodeid)
                if elem['TreeNodePredicate']['PredicateType'] == 'SimplePredicate':
                    if 'Operator' in elem['TreeNodePredicate']:

                        label2 = idName[elem['TreeNodePredicate']['FieldPredID']]+' '+elem['TreeNodePredicate']['Operator'].replace('EQUAL',' ').replace('LESSTHAN','<').replace('GREATEROR','>')+' '+str(elem['TreeNodePredicate']['FieldValue'])
                        try:
                            label2 = idName[elem['TreeNodePredicate']['FieldPredID']]+' '+elem['TreeNodePredicate']['Operator'].replace('EQUAL',' ').replace('LESSTHAN','<').replace('GREATEROR','>')+' '+(idTrans[elem['TreeNodePredicate']['FieldPredID']])[elem['TreeNodePredicate']['FieldValue']]  
                        except Exception:
                            pass
                    else:
                        label2 = idName[elem['TreeNodePredicate']['FieldPredID']]+' '+(idTrans[elem['TreeNodePredicate']['FieldPredID']])[elem['TreeNodePredicate']['FieldValue']]  
                    
                else:
                    compoundPredicates = elem['TreeNodePredicate']['Predicates']
                    
                    comOp = str(elem['TreeNodePredicate']['DecisionTreeJsonBooleanOperator'])
                    label2 = 'Compound ('+comOp+')\n'
                    for comPredicate in compoundPredicates:
                        tempLabel = label2
                        comOpShow = comPredicate['Operator'].replace('EQUAL',' ')
                        comOpShow = comOpShow.replace('LESSTHAN','<')
                        comOpShow = comOpShow.replace('GREATEROR','>=')
                        label2 = tempLabel+idName[comPredicate['FieldPredID']]+' '+comOpShow+' '+str(comPredicate['FieldValue'])+'\n'
                        try:
                            label2 = tempLabel+idName[comPredicate['FieldPredID']]+' '+comOpShow+' '+(idTrans[comPredicate['FieldPredID']])[comPredicate['FieldValue']] +'\n' 
                        except Exception:
                            pass
  
                edge.set_label(label2)
                dot.add_edge(edge)
            else:                       
                for tempNode in tempNodeList:   
                    nextnodeid = id(tempNode)
                    node = pydot.Node(nextnodeid)  
                    #label = str(tempNode['TreeNodePredicate']['FieldPredID'])+'='+str(tempNode['TreeNodePredicate']['FieldValue'])+'\n'+ str(tempNode['Score']) 
         
                    label ='RecordCount: '+str(tempNode['RecordCount']) +'\nScore: '+ str(tempNode['Score']) +'\nScoreDistribution: '+ str([round(unit, 2) for unit in tempNode['ScoreDistribution'][1]])  
                    node.set_label(label)
                    dot.add_node(node)
                    edge = pydot.Edge(currentnodeid,nextnodeid)
                    if tempNode['TreeNodePredicate']['PredicateType'] == 'SimplePredicate':
                        if 'Operator' in tempNode['TreeNodePredicate']:

                            label3 = idName[tempNode['TreeNodePredicate']['FieldPredID']]+' '+tempNode['TreeNodePredicate']['Operator'].replace('EQUAL',' ').replace('LESSTHAN','<').replace('GREATEROR','>')+' '+str(tempNode['TreeNodePredicate']['FieldValue'])
                            try:
                                label3 = idName[tempNode['TreeNodePredicate']['FieldPredID']]+' '+tempNode['TreeNodePredicate']['Operator'].replace('EQUAL',' ').replace('LESSTHAN','<').replace('GREATEROR','>')+' '+(idTrans[tempNode['TreeNodePredicate']['FieldPredID']])[tempNode['TreeNodePredicate']['FieldValue']]  
                            except Exception:
                                pass
                        else:
                            label3 = idName[tempNode['TreeNodePredicate']['FieldPredID']]+' '+(idTrans[tempNode['TreeNodePredicate']['FieldPredID']])[tempNode['TreeNodePredicate']['FieldValue']]  
                    
                    else:                        
                        compoundPredicates = tempNode['TreeNodePredicate']['Predicates']
                    
                        comOp = str(tempNode['TreeNodePredicate']['DecisionTreeJsonBooleanOperator'])
                        label3 = 'Compound ('+comOp+')\n'
                        for comPredicate in compoundPredicates:
                            tempLabel = label3
                            comOpShow = comPredicate['Operator'].replace('EQUAL',' ')
                            comOpShow = comOpShow.replace('LESSTHAN','<')
                            comOpShow = comOpShow.replace('GREATEROR','>=')
                            label3 = tempLabel+idName[comPredicate['FieldPredID']]+' '+comOpShow+' '+str(comPredicate['FieldValue'])+'\n'
                            try:
                                label3 = tempLabel+idName[comPredicate['FieldPredID']]+' '+comOpShow+' '+(idTrans[comPredicate['FieldPredID']])[comPredicate['FieldValue']] +'\n' 
                            except Exception:
                                pass
                    edge.set_label(label3)
                    dot.add_edge(edge)

        if len(nextLayer)>0:
            nextLayerHasElement = True
        else:
            nextLayerHasElement = False 
        if layerId == 0:
            layerId = layerId+1
        else:      
            currentLayer = nextLayer

    return dot

def mlp_model_to_dot(model):    
    dot = pydot.Dot()
    dot.set('rankdir', 'LR')
    dot.set('concentrate', True)
    #dot.set_node_defaults(shape='record')
    network = model['NeuralNetwork']
    inputlayer = network['NeuralInputs']
    
    #print 'Input Layer' + str(0)
    for inputneuron in inputlayer:
        #print ' Neuron Id '+str(inputneuron['id'])        
        node = pydot.Node(inputneuron['id'])
        try:
            label = str(inputneuron['DerivedField']['NormContinuous']['field'])+'\n'+str(inputneuron['DerivedField']['NormContinuous']['value'])
            
            node.set_label(label)
        except Exception:
            pass
        try:
            label = str(inputneuron['DerivedField']['NormDiscrete']['field'])+'\n'+str(inputneuron['DerivedField']['NormDiscrete']['value'])
           
            node.set_label(label)
        except Exception: 
            pass
        try:
            norm = inputneuron['DerivedField']['NormContinuous']['Norms']
            label = str(inputneuron['DerivedField']['NormContinuous']['field'])+'\n'+'\n'.join(str(d) for d in norm)
            
            
            node.set_label(label)
        except Exception: 
            pass
        
        dot.add_node(node)
    
    outputlabel = []
    outputlayers = network['NeuralOutputs']

    outputlen = len(outputlayers)
    for outputnode in outputlayers:
        try:
             outputlabel.append(outputnode['DerivedField']['NormDiscrete']['value'])
        except Exception: 
            pass
        try:
             outputlabel.append(outputnode['DerivedField']['NormContinuous']['value'])
        except Exception: 
            pass    
       
    layers = network['NeuralLayers']
    layercount = 1;
    for layer in layers:
        #print str('Layer '+str(layercount))
        layercount = layercount+1
        neurons = layer['Neurons']
        neuronCount = 0
        for neuron in neurons:
            #print ' Neuron Id '+str(neuron['id'])
            node = pydot.Node(neuron['id']) 
            #try activate function
            if layercount > outputlen:
                node.set_label(outputlabel[neuronCount])
                neuronCount = neuronCount+1
            dot.add_node(node)   
            connectionsFrom = neuron['ConnectionsFrom']
            for connectionFrom in connectionsFrom:
                #print '  connection from '+str(connectionFrom['from'])+' | weight '+str(connectionFrom['weight'])
                edge = pydot.Edge(connectionFrom['from'], neuron['id'], weight=connectionFrom['weight'])
                edge.set_label(connectionFrom['weight'])
                dot.add_edge(edge)

 
    return dot

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def pal_filter(start_time,end_time,hana_df):
    to_ts_start_time = 'TO_TIMESTAMP('+"'"+start_time+"'"+','+"'"+'YYYY-MM-DD HH24:MI:SS'+"'"+')'
    to_ts_end_time = 'TO_TIMESTAMP('+"'"+end_time+"'"+','+"'"+'YYYY-MM-DD HH24:MI:SS'+"'"+')'

    return hana_df\
.filter(hana_df.columns[0]+'>='+to_ts_start_time)\
.filter(hana_df.columns[0]+'<='+to_ts_end_time) 

def getHFMin(hana_df):
    firstCol = re.findall('SELECT '+'(.*?)' +' FROM',hana_df.select_statement)[0]

    return hana_df.connection_context.sql(hana_df.select_statement.replace(firstCol,'min('+'\"'+hana_df.columns[0]+'\"'+')',1)).collect().iat[0,0]

def getHFMax(hana_df):
    
    firstCol = re.findall('SELECT '+'(.*?)' +' FROM',hana_df.select_statement)[0]

    return hana_df.connection_context.sql(hana_df.select_statement.replace(firstCol,'max('+'\"'+hana_df.columns[0]+'\"'+')',1)).collect().iat[0,0]


def hana_sample(hana_df,width):
    delta=1e-10
    TIMESTAMP=hana_df.columns[0]
    Y=hana_df.columns[1]
    start_time=getHFMin(hana_df).strftime("%Y-%m-%d %H:%M:%S")
    end_time=getHFMax(hana_df).strftime("%Y-%m-%d %H:%M:%S")
    to_ts_start_time = 'TO_TIMESTAMP('+"'"+start_time+"'"+','+"'"+'YYYY-MM-DD HH24:MI:SS'+"'"+')'
    to_ts_end_time = 'TO_TIMESTAMP('+"'"+end_time+"'"+','+"'"+'YYYY-MM-DD HH24:MI:SS'+"'"+')'
    add_key_sql_statement = 'SELECT round('\
    +str(width)\
    +'*SECONDS_BETWEEN('+TIMESTAMP+','+to_ts_start_time+')/(1e-10+SECONDS_BETWEEN('\
    +to_ts_end_time+','+to_ts_start_time+'))) k,'+TIMESTAMP+','+Y+' FROM '+'('+ hana_df.select_statement +')'
    
    min_max_groupby_sql_statement = 'SELECT k,min('+Y+')  v_min, max('+Y+') v_max,min('+TIMESTAMP+') t_min, max('+TIMESTAMP+') t_max FROM'+'('+add_key_sql_statement+') GROUP BY k'
    
    join_sql_statement = 'SELECT T.'+TIMESTAMP+',T.'+Y+' FROM '+'('+add_key_sql_statement+') T '+'INNER JOIN '\
    +'('+min_max_groupby_sql_statement+') Q ' \
    +'ON T.k=Q.k ' \
    +'AND (T.'+Y+'=Q.v_min OR T.'+Y+'=Q.v_max OR T.'+TIMESTAMP+'=Q.t_min OR T.'+TIMESTAMP+'=Q.t_max)'
    df_sample=dataframe.DataFrame(hana_df.connection_context,join_sql_statement)
    return df_sample


def tsPlot(data_range,w,cba,bit,hana_df):

    start_time=data_range[0].strftime("%Y-%m-%d %H:%M:%S")
    end_time=data_range[1].strftime("%Y-%m-%d %H:%M:%S")

    filtered_df = hana_sample(pal_filter(start_time,end_time,hana_df),w).collect()

    filtered_df.set_index(hana_df.columns[0], inplace=True)
    filtered_df=filtered_df.astype(float)

    if (cba==True):
        filtered_df['avg('+hana_df.columns[1]+')']=filtered_df[hana_df.columns[1]].rolling(bit).mean()
        fit,ax = plt.subplots()
        ax = filtered_df.plot(ax=ax,title=start_time+' - '+end_time,legend=True,figsize=(15,8))
        ax.set_ylabel(hana_df.columns[1])
        ax.legend(loc='upper left')
    elif (cba==False):
        fit,ax = plt.subplots()
        ax = filtered_df.plot(ax=ax,title=start_time+' - '+end_time,legend=False,figsize=(15,8))
        ax.set_ylabel(hana_df.columns[1])


def pal_interact_plot(hana_df):
    start_date = getHFMin(hana_df)
    end_date = getHFMax(hana_df)
    dates = pd.date_range(start_date, end_date, freq='D')
    options = [(date.strftime("%Y-%m-%d"), date) for date in dates]
    index = (0, len(options)-1)
    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        description='Dates',
        continuous_update=False,
        orientation='horizontal',
        layout={'width': '450px'}
    )
    
    float_slider = widgets.FloatSlider(
        value=100,
        min=0.0000000000000000000001,
        max=(len(options)-1)/10,
        step=0.0000001,
        description='Pixel',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.5f',
        layout={'width': '300px'}
    )
    check_box_avg = widgets.Checkbox(
        value=False,
        description='moving average',
        disabled=False
    )
    
    bounded_int_text = widgets.BoundedIntText(
        value=5,
        min=1,
        max=1000,
        step=1,
        description='step',
        disabled=False
    )
    interact(tsPlot,data_range=selection_range_slider,w=float_slider,cba=check_box_avg,bit=bounded_int_text,hana_df=fixed(hana_df), continuous_update=False)
    
    

