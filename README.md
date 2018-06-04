# coding = UTF-8

Esse projeto faz análise de sentimento de tweets sobre segurança pública no Brasil utilizando framework python django através da técnica de classificação.
O django é um framework para desenvolvimento de aplicações web para a linguagem python.

A base desse projeto é a utilização da inteligência artificial através da técnica de classificação, utilizando 5 algoritmos: RandomForestClassifier, LogisticRegression, decision tree, SVM e Naive Bayes (utilizando a classe MultinomialNB) para a análise de sentimento dos usuário sobre a segurança pública no Brasil.

A cada novo tweet classificado, a página no browser exibe o último tweet classificado, o percentual e total de cada classe. Os três algoritmos classificam um tweet, mas o tweet é inserido em apenas uma classe. Por isso, é verificado o resultado de cada algoritmo e a classificação de maior quantidade é a classificação válida. Por exemplo: se os algoritmos classificarem um texto do tweet respectivamente como: neutro, neutro, inseguro. Prevalecerá a classificação neutro.

Para usar a api do twitter, é necessário ter uma conta e criar um app na própria rede social, o qual contém as credenciais que permitem acessar os dados.

Para preparar o ambiente, instale o Anaconda3-4.4.0-Windows-x86_64

Atenção: o anaconda já traz o próprio python instalado. Portanto, deve-se instalar apenas o anaconda e depois instalar o require.txt.
Pegue o anaconda 3-4-4.0 no link:
https://repo.continuum.io/archive/Anaconda3-4.4.0-Windows-x86_64.exe

Os pacotes necessários estão no arquivo require.txt que devem ser instalados após a instalação do anaconda.
Para instalar os pacotes do arquivo texto referido, execute no prompt ou terminal: pip install -r require.txt
Não esqueça de apontar o caminho absoluto do require.txt caso você não esteja na mesma página do require.txt.
Após a instalação do require.txt, execute no console python:
import nltk
nltk.download()
Quando abrir a janela de download dos plugins do nltk, escolha all para evitar problemas com módulos não encontrados.

Utilize o terminal ou prompt para ir para a raiz do projeto (python-analise-de-sentimento-com-twitter). Digite:
python manage.py runserver e tecle enter.
No browser, digite: localhost:8000/mineracao

A cada 20 segundo, um tweet é buscado, classificado e a página é renderizada novamente.