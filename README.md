﻿# coding = UTF-8

Este projeto faz análise de sentimento de tweets sobre o derramamento de óleo nas praias do Brasil utilizando framework python django através da técnica de classificação.
O django é um framework para desenvolvimento de aplicações web para a linguagem python.

A base desse projeto é a utilização da inteligência artificial através da técnica de classificação, utilizando o algoritmo Naive Bayes para a mineração da opinião dos usuário sobre a mancha de óleo nas praias brasileiras, classificando os tweets em crítica ou outros.

A cada novo tweet classificado, o browser é renderizado, exibindo o último tweet classificado e um gráfico com google chart mostrando o percentual de tweets em cada classe, além do total de tweets no dataset.

Para usar a api do twitter, é necessário estar logado  e criar um app na própria rede social, o qual contém as credenciais que permitem acessar os dados, através do link:
https://apps.twitter.com/app/new

Para preparar o ambiente, instale o Anaconda3-4.4.0-Windows-x86_64

Atenção: o anaconda já traz o próprio python instalado. Portanto, deve-se instalar apenas o anaconda e depois instalar o require.txt, que contém os pacotes necessários para aplicação.

Pegue o anaconda 3-4-4.0 no link:
https://repo.continuum.io/archive/Anaconda3-4.4.0-Windows-x86_64.exe

Ou escolha para seu SO em:
https://repo.continuum.io/archive/

Os pacotes necessários estão no arquivo require.txt que devem ser instalados após a instalação do anaconda.
Para instalar os pacotes do arquivo texto referido, execute no prompt ou terminal: pip install -r require.txt
Não esqueça de apontar o caminho absoluto do require.txt caso você não esteja na mesma pasta do arquivo.

Após a instalação do require.txt, execute no console python:
import nltk
nltk.download()
Quando abrir a janela de download dos plugins do nltk, escolha all para evitar problemas com módulos não encontrados.
ou digite num terminal ou prompt:
python -m nltk.downloader all
O comando acima baixa diretamente sem precisar da interface gráfica do nltk.

Utilize o terminal ou prompt para ir para a raiz do projeto (python-analise-de-sentimento-com-twitter). Digite:
python manage.py runserver e tecle enter.

No browser, digite: localhost:8000/mineracao

A cada 20 segundo, um tweet é buscado, classificado e a página é renderizada novamente.

O resultado dos experimentos com os algoritmos estão na pasta experimentos.

O resultado da avaliação dos algoritmos está no arquivo através do jupyternotebook analise-de-sentimento.ipynb

Vá para a pasta experimento no prompt ou shell e digite jupyternotebook. Abrirá o browser padrão do computador. Clique no arquivo analise-de-sentimento.ipynb para conferir o código fonte e os resultados dos algoritmos. Atenção, para utilizar o jupyternotebook é necessário ter instalado o anaconda ou python.

Leia mais sobre jupyternotebook em jupyter.org/

Leia mais sobre google chart em:
https://developers.google.com/chart/
https://developers.google.com/chart/interactive/docs/gallery