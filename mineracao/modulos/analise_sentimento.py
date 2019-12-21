# coding=UTF-8

import os
import re
from unicodedata import normalize

import nltk
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from twython import Twython

import pandas as pd
import random as rn


conexao=None
dfNovo=None

def main():
    # pega as variáveis globais necessárias
    global conexao
    global dfNovo
    
    #palavras chaves para buscar no twitter
    palavrasChaves=["política", "político", "futebol", "assalto", "assassino", "ladrão", "violência", "mata", "medo", "crime", "notícia", "noticiário", "operação", "policial", "assédio", "pânico", "roubo", "copa", "famoso", "artista", "favela", "mundo", "brasil", "orientação", "homem", "mulher", "criança", "briga", "segurança", "insegurança", "país", "paz", "jesus", "Deus", "igreja", "remédio", "medicina", "médico", "tiro", "troca", "internet", "Trump", "estados unidos", "guerra", "time", "club", "curso", "casal", "jornal", "rádio", "cidade", "capital", "morte", "trânsito", "intervenção", "campeão", "brasileiro", "bonito", "feio", "lindo", "bonita", "linda", "feia", "magro", "gordo", "magra", "gorda", "presidente", "ex", "guarda", "prefeito", "municipal", "recurso", "ministério", "público", "eleição", "dinheiro", "feliz", "felicidade", "professor", "aluno", "estudante", "ameaça", "manhã", "tarde", "noite", "hoje", "música", "ruin", "jornal", "cego", "deficiente", "poder", "instituto", "greve", "paraliza", "sindicato", "terror", "grupo", "revólver", "tráfico", "droga", "a", "e", "i", "o", "u", "compra", "energia", "trabalha", "trabalho", "coleção", "problema", "péssimo", "coragem", "óleo", "petróleo", "contaminação", "praia", "governo", "bolsonaro", "ministro", "poluição", "pesca", "venezuela", "voluntário", "mancha", "desastre", "ambiente", "ambiental", "peixe", "nordeste", "culpa", "culpado", "competência", "incompetência", "derrubar", "justiça", "mentira", "mentiroso"]
    
    # escolhe uma palavra da lista para buscar no twitter
    palavra=escolherPalavraDalista(palavrasChaves)
    
    #base de tweets classificados
    nomeArquivo="tweets_classificados.csv"
    
    #ler a base de tweets classificados
    df=lerCSV(nomeArquivo)
    
    #faz a conexão com o twitter
    conexao=conectarTwitter()
    
    #faz a busca por tweet
    tweetEncontrado=buscar(palavra)
    
    #ccontinua o processo de classificação se for encontrado tweet
    tweetEncontrado=True
    if (tweetEncontrado):
        
        #percorre o json de tweet encontrado para pegar usuário, texto e id
        tweet=adicionarNaLista(tweetEncontrado)
        
        #trata o texto do tweet
        texto=etl(dataframe['texto'])
        
        #remove a coluna usuário do dataframe classificado
        dft=removerColuna(df)
        
        #trata os textos dos tweets
        dft['texto']=etl(dft['texto'])
        
        #classificar nova instância de tweet
        try:
            classificacao=classificarMultinomialNB(dft['texto'], dft['sentimento'], texto)
            
            #preenche a coluna sentimento do dataframe criado com a classificação encontrada
            dataframe['sentimento']=classificacao
            
            #concatena o dataframe do arquivo de treino com o novo dataframe classificado
            dfNovo=df
            dfNovo=adicionarItem(df, dataframe)
            c=calcularPercentualTotal(df)
            
            # verifica se o novo tweet é duplicado e remove-o caso seja
            dfNovo=removerDuplicadas(dfNovo)
            
            #cria e sobrescreve a base de tweets classificados com o novo texto também classificado
            criarArquivo(dfNovo, nomeArquivo)
            
        except:
            main()

def lerCSV(nomeArquivo):
    return pd.read_csv(nomeArquivo, encoding='ISO-8859-1', sep=";", header=0)

def removerPontuacao(texto):
    return re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôûÂÊÎÔÛàèìòùÀÈÌÒÙãõÇç ]', '', texto)

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
    
    #trata o texto de cada linha do dataframe
    for texto in textos:
        
        #remover mensionamento (@usuario)
        texto=removerMensionamento(texto)
        
        #remove URL
        texto=removerURL(texto)
        
        #transforma a frase toda em minúscula
        texto=transformarEmMinusculas(texto)
        
        #remove as pontuações
        texto=removerPontuacao(texto)
        
        #remove números
        texto=removerNumeros(texto)
        
        #remove stopwords
        texto=removerStopWord(texto)
        
        #aplica stemming
        texto=aplicarStemming(texto)
        
        #retirar assentos
        texto=removerAssentuacao(texto)
        
        tweets.append(texto)
    return tweets

def calcularClasses(df):
    numero_apoios = len(df.loc[df['sentimento'] == 'apoio'])
    numero_criticas= len(df.loc[df['sentimento'] == 'crítica'])
    numero_outros = len(df.loc[df['sentimento'] == 'outros'])
    
    quantidade=[]
    quantidade.append("{}".format(numero_apoios))
    quantidade.append("{}".format(numero_criticas))
    quantidade.append("{}".format(numero_outros))
    
    return quantidade

def calcularPercentualTotal(df):
    #numero_apoios = len(df.loc[df['sentimento'] == 'apoio'])
    numero_criticas= len(df.loc[df['sentimento'] == 'crítica'])
    numero_outros = len(df.loc[df['sentimento'] == 'outros'])
    
    percentuais=[]
    
    #percentuais.append("{:2.2f}".format(numero_apoios/(numero_criticas+numero_apoios+numero_outros)*100))
    percentuais.append("{:2.2f}".format(numero_criticas/(numero_criticas+numero_outros)*100))
    percentuais.append("{:2.2f}".format(numero_outros/(numero_criticas+numero_outros)*100))
    return percentuais

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
    
def conectarTwitter():
    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_token_secret = ''
    conectado= Twython(consumer_key, consumer_secret, access_token, access_token_secret)
    return conectado

def buscar(palavras_chaves):
    global conexao
    resultado = conexao.search(q=palavras_chaves, result_type='recent', locale='Brasil', lang='pt', count=1)
    return resultado

def adicionarNaLista(tweetsEncontrados):
    tweets=[]
    for tweetEncontrado in tweetsEncontrados['statuses']:
        tweet={'id': tweetEncontrado['id'], 'usuário': tweetEncontrado['user']['name'], 'texto': tweetEncontrado['text']}
        tweets.append(tweet)
        return tweets

def criarDF(tweets):
    df=pd.DataFrame(tweets, columns=['usuário', 'texto', 'sentimento'])
    return df

def adicionarItem(df, item):
    return df.append(item, ignore_index=True)

def criarArquivo(df, nome):
    criado=df.to_csv(nome, sep=";", index=False)
    return criado

def escolherMelhorClassificacao(c1, c2, c3, c4, c5):
    critica=0
    
    if (c1=="crítica"):
        critica=critica+1
        
    if (c2=="crítica"):
        critica=critica+1
        
    if (c3=="crítica"):
        critica=critica+1
        
    if (c4=="crítica"):
        critica=critica+1
        
    if (c5=="crítica"):
        critica=critica+1
        
    if (critica>2):
        return "crítica"
    
    return "outros"

# função para pegar o dataframe criado depois da classificação do tweet encontrado
def getDataFrame():
    global dfNovo
    return dfNovo

def escolherPalavraDalista(palavrasChaves):
    # escolhe um número aleatório do intervalo de 0 a tamanho da lista (n-1)
    posicao=rn.sample(range(0, len(palavrasChaves)-1), 1)
    
    # faz um cast da variável posicao do tipo lista para int
    posicao=int(posicao[0])
    
    # retorna a palavra escolhida da lista
    return palavrasChaves[posicao]

def removerDuplicadas(df):
    return df.drop_duplicates(['texto'])


