# coding=UTF-8
import io
import os
import re
import time
import sys
from unicodedata import normalize

import bokeh
from bokeh.charts import Bar, output_notebook, show
from bokeh.charts import Scatter, output_file, show
from bokeh.plotting import figure, output_file, show
from nltk import FreqDist
from nltk import tokenize
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.probability import FreqDist
from nltk.stem.snowball import SnowballStemmer
from sklearn import metrics
from sklearn import svm
from sklearn import tree
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from twython import Twython

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rn
import seaborn as sea

conexao=None

def main():
    #palavras chaves para buscar no twitter
    palavras_chaves=["violência", "futebol"]
    #caminho da pasta com o arquivo
    pasta="c:/users/usuario/dados"
    alterarPasta(pasta)

    #variável com id do ultimo twitter
    id=930943609800679426
    
    #base de tweets classificados
    nomeArquivo="tweets_classificados.csv"
    
    #quantidade de tweet para buscar
    quantidade=1
    
    #data do inícil da busca
    desde='2014-01-01'
    
    #ler a base de tweets classificados
    df=lerCSV(nomeArquivo)
    
    #faz a conexão com o twitter
    globals()[conexao]=conectarTwitter()
    
    #faz a busca por tweet
    tweetEncontrado=buscar(id, quantidade, palavras_chaves)
    
    #ccontinua o processo de classificação se for encontrado tweet
    
    if (tweetEncontrado):
        #percorre o json de tweet encontrado para pegar usuário, texto e id
        tweet=adicionarNaLista(tweetEncontrado)
        
        #pega o id do tweet
        id=tweet[0]['id']
        
        #cria um dataframe com o novo tweet ainda não classificado
        dataframe =criarDF(tweet)
    
        #trata o texto do tweet
        texto=etl(dataframe['texto'])
        
        #remove a coluna usuário do dataframe classificado
        dft=removerColuna(df)
        
        #trata os textos
        dft['texto']=etl(dft['texto'])
        
        #classificar nova instância de tweet
        c1=classificarDecisionTree(dft['texto'], dft['sentimento'], texto)
        c2=classificarSVM(dft['texto'], dft['sentimento'], texto)
        c3=classificarMultinomialNB(dft['texto'], dft['sentimento'], texto)
        
        #escolhe as duas melhores classificações
        classificacao=escolherMelhorClassificacao(c1, c2, c3)
        
        #preenche a coluna sentimento do dataframe criado com a classificação encontrada
        dataframe['sentimento']=classificacao
        
        #concatena o dataframe criado já classificado com o outro dataframe classificado
        dfNovo=adicionarItem(df, dataframe)
        
        #cria e subscreve a base de tweets classificados com o novo texto também classificado
        criarArquivo(dfNovo, "tweets_classificados")

def alterarPasta(pasta):
    os.chdir(pasta)

def lerCSV(nomeArquivo):
    return pd.read_csv(nomeArquivo, encoding='ISO-8859-1', sep=";", header=0)

def removerPontuacao(texto):
    return re.sub(u'[^a-zA-Z0-9������������������������ ]', ' ', texto)

def removerStopWord(texto):
    stopWord=nltk.corpus.stopwords.words('portuguese')
    sentencaSemStopword = [i for i in texto.split() if not i in stopWord]
    return " ".join(sentencaSemStopword)

def removerAssentuacao(texto):
    return normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')

def removerNumeros(texto):
    return re.sub('[0-9]', '', texto)

def removerURL(texto):
    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*','', texto)

def transformarEmMinusculas(texto):
    return texto.lower()

def aplicarStemming(texto):
    texto=str(texto)
    stemmer = nltk.stem.RSLPStemmer()
    palavras=[]
    for txt in texto.split():
        palavra=stemmer.stem(txt)
        palavras.append(palavra)
    return " ".join(palavras)

def removerColuna(dataframe):
    dataframe=dataframe.drop(dataframe.columns[0], axis=1)
    return dataframe

def removerMensionamento(texto):
    if texto.find("@")>=0:
        texto=texto.split("@")
        texto=str(texto[-1])
        contem=texto.find(" ")
        texto=texto[contem+1:]
        
    return texto

def etl(textos):
    
    tweets=[]
    # pega apenas os valores do dataframe
   
 #trata o texto de cada linha do dataframe
    for texto in textos:
        texto=str(texto)
        
        #remover mensionamento (@usuario)
        texto=removerMensionamento(texto)
        
        #remove URL
        texto=removerURL(texto)
        
        #transforma a frase toda em min�scula
        texto=transformarEmMinusculas(texto)
        
        #remove números
        texto=removerNumeros(texto)
        
        #remove as pontuações
        texto=removerPontuacao(texto)
        
        #retirar assentos
        texto=removerAssentuacao(texto)
        
        #aplica stemming
        texto=aplicarStemming(texto)
        
        #remove stopwords
        texto=removerStopWord(texto)
        
        texto=str(texto)
        
        tweets.append(texto)
    return tweets

def calcularPercentualTotal(df):
    numero_seguros = len(df.loc[df['sentimento'] == 'seguro'])
    numero_inseguros= len(df.loc[df['sentimento'] == 'inseguro'])
    numero_neutros = len(df.loc[df['sentimento'] == 'neutro'])
    
    percentuais=[]
   
    percentuais.append("({:2.2f}%)".format(numero_seguros/(numero_inseguros+numero_seguros+numero_neutros)*100))
    percentuais.append("{:2.2f}%)".format(numero_inseguros/(numero_inseguros+numero_seguros+numero_neutros)*100))
    
    percentuais.append("{:2.2f}%".format(numero_neutros/(numero_inseguros+numero_seguros+numero_neutros)*100))
    return percentuais

def calcularClasses(df):
    numero_seguros = len(df.loc[df['sentimento'] == 'seguro'])
    numero_inseguros= len(df.loc[df['sentimento'] == 'inseguro'])
    numero_neutros = len(df.loc[df['sentimento'] == 'neutro'])
    
    quantidade=[]
    quantidade.append("seguros: {}".format(numero_seguros))
    quantidade.append("inseguros: {}".format(numero_inseguros))
    quantidade.append("neutros: {}".format(numero_neutros))
    
    return quantidade

def classificarDecisionTree(textos, sentimento, texto):
    #classificação com algoritmo tree
    
    #cria um vetor de 1 palavra
    vetor=criarVetor1Palavra()
    
    #pega a frequência das palavras
    textos_freq=vetor.fit_transform(textos)
    
    #cria o modelo
    modelo = tree.DecisionTreeClassifier()
    
    #treina o modelo passando a frequência de palavras de treino e o valor da coluna sentimento
    modelo.fit(textos_freq, sentimento)
    
    #pega a frequência das palavras
    texto_freq=vetor.transform(texto)
    
    #previsão do modelo
    previsao=modelo.predict(texto_freq)
    return previsao

def classificarSVM(textos, sentimento, texto):
    
    #classificação com svm
    
    #cria um vetor de 1 palavra
    vetor = criarVetor1Palavra()
    
    #pega a frequência das palavras
    textos_freq = vetor.fit_transform(textos)
    
    #cria o modelo
    modelo = svm.SVC(gamma=0.001, C=100.)
    
    #treina o modelo passando a frequência de palavras de treino e o valor da coluna sentimento
    modelo.fit(textos_freq, sentimento)
    
    #pega a frequência das palavras
    texto_freq = vetor.transform(texto)
    
    #previsão do modelo
    previsao = modelo.predict(texto_freq)
    
    return previsao

def classificarMultinomialNB(textos, sentimento, texto):
    
    #esta classificação é feita usando o NaiveBayes do skit-learn com MultinomialNB()
    
        #cria um vetor de 1 palavra
    vetor = CountVectorizer(analyzer="word")
    
    #pega a frequência das palavras
    textos_freq = vetor.fit_transform(textos)
    
    #cria o modelo
    modelo = MultinomialNB()
    
    #treina o modelo passando a frequência de palavras de treino e o valor da coluna sentimento
    modelo.fit(textos_freq, sentimento)
    
    #pega a frequência das palavras
    texto_freq = vetor.transform(texto)
    
    #previsão do modelo
    previsao=modelo.predict(texto_freq)
    
    return previsao

def criarVetor1Palavra():
        #a linha abaixo traz o vetor de 1 palavra
    return CountVectorizer(analyzer="word")
    
def criarVetor2Palavras():
        # a linha abaixo tr�z o vetor de 2 em 2 palavras. Obs, o resultado desta foi melhor
    return CountVectorizer(ngram_range=(1,2))

def conectarTwitter():
    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_token_secret = ''
    conectado= Twython(consumer_key, consumer_secret, access_token, access_token_secret)
    return conectado

def buscar(id, quantidade, palavras_chaves):
    resultado = globals()[conexao].search(q=palavras_chaves, since_id=id, result_type='recent', locale='Brasil', lang='pt', count=1)
    return resultado

def adicionarNaLista(tweetsEncontrados):
    tweets=[]
    for tweetEncontrado in tweetsEncontrados["statuses"]:
        tweet={'id': tweetEncontrado['id'], 'usuário': tweetEncontrado['user']['name'], 'texto': tweetEncontrado['text']}
        tweets.append(tweet)
        return tweets

def criarDF(tweets):
    df=pd.DataFrame(tweets, columns=['usuário', 'texto', 'sentimento'])
    return df

def adicionarItem(df, item):
    return df.append(item, ignore_index=True)

def criarArquivo(df, nome):
    criado=df.to_csv(nome+'.csv', sep=";", index=False)
    return criado

def escolherMelhorClassificacao(c1, c2, c3):
    
    #inicializa a classe vazia
    classe=None
    
    #inicializa os contadores da quantidade de classes
    inseguro=0
    seguro=0
    neutro=0
    
    if (c1=="inseguro"):
        inseguro=inseguro+1
        
    if (c1=="seguro"):
        seguro=seguro+1
        
    if (c1=="neutro"):
        neutro=neutro+1
        
    if (c2=="inseguro"):
        inseguro=inseguro+1
        
    if (c2=="seguro"):
        seguro=seguro+1
        
    if (c2=="neutro"):
        neutro=neutro+1
        
    if (c3=="inseguro"):
        inseguro=inseguro+1
        
    if (c3=="seguro"):
        seguro=seguro+1
        
    if (c3=="neutro"):
        neutro=neutro+1
        
    #se as classes forem diferentes, retorna neutro, já que não se sabe se é realmente seguro ou inseguro
    if (seguro==1 & inseguro==1):
        classe="neutro"
        
    #se houver classificações com o mesmo valor
    else:
        
        #cria um dataframe para pegar a maior repetição duma classificação
        classes={"classe": ["seguro", "inseguro", "neutro"], "quantidade": [seguro, inseguro, neutro]}
        dfClasse=pd.DataFrame(classes)
        
    #escolhe a classe repetida
        classe=dfClasse["classe"].loc[dfClasse["quantidade"]>1].values
        
    print(seguro, inseguro, neutro)
    return classe

#define o tempo para fazer nova busca
time.sleep(30)
while (True):
    main()