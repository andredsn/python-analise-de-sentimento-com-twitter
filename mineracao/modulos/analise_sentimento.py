# coding=UTF-8

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
    palavrasChaves=["política", "político", "futebol", "assalto", "assassino", "ladrão", "violência", "mata", "medo", "crime", "notícia", "noticiário", "operação", "policial", "assédio", "pânico", "roubo", "copa", "famoso", "artista", "favela", "mundo", "brasil", "orientação", "homem", "mulher", "criança", "briga", "segurança", "insegurança", "país", "paz", "jesus", "Deus", "igreja", "remédio", "medicina", "médico", "tiro", "troca", "internet", "Trump", "estados unidos", "guerra", "time", "club", "curso", "casal", "jornal", "rádio", "cidade", "capital", "morte", "trânsito", "intervenção", "campeão", "brasileiro", "bonito", "feio", "lindo", "bonita", "linda", "feia", "magro", "gordo", "magra", "gorda", "presidente", "ex", "guarda", "prefeito", "municipal", "recurso", "ministério", "público", "eleição", "dinheiro", "feliz", "felicidade", "professor", "aluno", "estudante", "apeaça", "manhã", "tarde", "noite", "hoje", "música", "ruin", "jornal", "cego", "deficiente", "poder", "instituto", "greve", "paraliza", "sindicato", "terror", "grupo", "revólver", "tráfico", "droga", "a", "e", "i", "o", "u", "compra", "energia", "trabalha", "trabalho", "coleção", "problema", "péssimo", "coragem"]
    
    # escolhe uma palavra da lista para buscar no twitter
    palavra=escolherPalavraDalista(palavrasChaves)
    
    #base de tweets classificados
    nomeArquivo="tweets_classificados"
    
    #ler a base de tweets classificados
    df=lerCSV(nomeArquivo)
    
    #faz a conexão com o twitter
    conexao=conectarTwitter()
    
    #faz a busca por tweet
    tweetEncontrado=buscar(palavra)
    
    #ccontinua o processo de classificação se for encontrado tweet
    
    if (tweetEncontrado):
        
        #percorre o json de tweet encontrado para pegar usuário, texto e id
        tweet=adicionarNaLista(tweetEncontrado)
        
        #cria um dataframe com o novo tweet ainda não classificado
        dataframe =criarDF(tweet)
        
        #trata o texto do tweet
        texto=etl(dataframe['texto'])
        
        #remove a coluna usuário do dataframe classificado
        dft=removerColuna(df)
        
        #trata os textos dos tweets
        dft['texto']=etl(dft['texto'])
        
        #classificar nova instância de tweet com cinco algoritmos
        try:
            c1=classificarDecisionTree(dft['texto'], dft['sentimento'], texto)
            c2=classificarSVM(dft['texto'], dft['sentimento'], texto)
            c3=classificarMultinomialNB(dft['texto'], dft['sentimento'], texto)
            c4 = classificarRandomForestClassifier(dft['texto'], dft['sentimento'], texto)
            c5 = classificarLogisticRegression(dft['texto'], dft['sentimento'], texto)
            
            #escolhe a melhor de 5 classificações
            classificacao=escolherMelhorClassificacao(c1, c2, c3, c4, c5)
            
            #preenche a coluna sentimento do dataframe criado com a classificação encontrada
            dataframe['sentimento']=classificacao
            
            #concatena o dataframe do arquivo de treino com o novo dataframe classificado
            dfNovo=adicionarItem(df, dataframe)
            
            # verifica se o novo tweet é duplicado e remove-o caso seja
            dfNovo=removerDuplicadas(dfNovo)
            
            #cria e sobrescreve a base de tweets classificados com o novo texto também classificado
            #criarArquivo(dfNovo, nomeArquivo)
            
        except:
            main()

def lerCSV(nomeArquivo):
    return pd.read_csv(nomeArquivo+".csv", encoding='ISO-8859-1', sep=";", header=0)

def removerPontuacao(texto):
    return re.sub(u'[^a-zA-Z0-9áéíóúÁÉÍÓÚâêîôûÂÊÎÔÛàèìòùÀÈÌÒÙãõ]+', ' ', texto)

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
        texto=str(texto)
        
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
    numero_inseguros= len(df.loc[df['sentimento'] == 'inseguro'])
    numero_outros = len(df.loc[df['sentimento'] == 'outro'])
    
    percentuais=[]
   
    percentuais.append("{:2.2f}".format(numero_inseguros/(numero_inseguros+numero_outros)*100))
    
    percentuais.append("{:2.2f}".format(numero_outros/(numero_inseguros+numero_outros)*100))
    return percentuais

def calcularClasses(df):
    numero_inseguros= len(df.loc[df['sentimento'] == 'inseguro'])
    numero_outros = len(df.loc[df['sentimento'] == 'outro'])
    
    quantidade=[]
    quantidade.append("{}".format(numero_inseguros))
    quantidade.append("{}".format(numero_outros))
    
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
    modelo=modelo.fit(textos_freq, sentimento)
    
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

def classificarRandomForestClassifier(textos, sentimento, texto):
    
    #classificação com floresta aleatória
    
    #cria um vetor de 1 palavra
    vetor=criarVetor1Palavra()
    
    #pega a frequência das palavras
    textos_freq=vetor.fit_transform(textos)
    
    #pega a frequência das palavras
    texto_freq=vetor.transform(texto)
    
    #cria o modelo
    modelo = RandomForestClassifier(random_state = 42)
    
    #treina o modelo passando a frequência de palavras de treino e o valor da coluna sentimento
    modelo.fit(textos_freq, sentimento.ravel())
    
    #previsão do modelo
    previsao= modelo.predict(texto_freq)
    
    return previsao

def classificarLogisticRegression(textos, sentimento, texto):
    #Regressão Logística
    
        #cria um vetor de 1 palavra
    vetor = criarVetor1Palavra()
    
    #pega a frequência das palavras
    textos_freq = vetor.fit_transform(textos)
    
    #cria o modelo
    modelo=LogisticRegression()
    
    #treina o modelo passando a frequência de palavras de treino e o valor da coluna sentimento
    modelo.fit(textos_freq, sentimento)
    
    #pega a frequência das palavras
    texto_freq = vetor.transform(texto)
    
    #previsão do modelo
    previsao= modelo.predict(texto_freq)
    
    return previsao

def criarVetor1Palavra():
        #a linha abaixo traz o vetor de 1 palavra
    return CountVectorizer(analyzer="word")
    
def conectarTwitter():
    consumer_key='2mDuJh76ceIXYQ76BnrQ2YC2Y'
    consumer_secret = 'yuSnuZoGDmj0DvTDuia6vz992jhfATnJ8OQ6UMbNXBuLK1wknS'
    access_token = '901839813392945152-2euiWPzJJB1SBjULYzA4b6p8D3OsvRA'
    access_token_secret = 'ZRbjml2L2J8KSRavCFGcycTgBfx5nPOpNktv3JCKSFOzL'
    
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
    criado=df.to_csv(nome+'.csv', sep=";", index=False)
    return criado

def escolherMelhorClassificacao(c1, c2, c3, c4, c5):
    inseguro=0
    
    if (c1=="inseguro"):
        inseguro=inseguro+1
        
    if (c2=="inseguro"):
        inseguro=inseguro+1
        
    if (c3=="inseguro"):
        inseguro=inseguro+1
        
    if (c4=="inseguro"):
        inseguro=inseguro+1
        
    if (c5=="inseguro"):
        inseguro=inseguro+1
        
    if (inseguro>2):
        return "inseguro"
    
    return "outro"

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

main()