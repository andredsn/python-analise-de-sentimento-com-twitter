"# python-analise-de-sentimento-com-twitter" 
# coding UTF-8

A base desse projeto é a utilização da inteligência artificial através da técnica de classificação, utilizando três algoritmos: decision tree, SVM e Naive Bayes (utilizando a classe MultinomialNB) para a análise de sentimento dos usuário sobre a segurança pública no Brasil.

A cada novo tweet classificado, a página no browser exibe o percentual e total de cada classe. Os três algoritmos classificam um tweet, mas o tweet é inserido em apenas uma classe. Por isso, é verificado o resultado de cada algoritmo e a classificação de maior quantidade é a classificação válida. Por exemplo: se os algoritmos classificarem um texto do tweet respectivamente como: neutro, neutro, inseguro. Prevalecerá a classificação neutro.

Para usar a api do twitter, é necessário ter uma conta e criar um app na própria rede social, o qual contém as credenciais que permitem acessar os dados.

Os pacotes necessários estão no arquivo require.txt, no qual  os principais pacots são: python 3.6, Django 1.11.7 e Anaconda3-4.4.0-Windows-x86_64

Django é um framework web python.

Atenção: o anaconda já traz o próprio python instalado. Portanto, deve-se instalar apenas o anaconda e depois instalar o require.txt.
Pegue o anaconda 3-4-4.0 no link:
https://repo.continuum.io/archive/Anaconda3-4.4.0-Windows-x86_64.exe

Para instalar os pacotes do arquivo texto referido, execute no prompt ou terminal: pip install -r require.txt
Não esqueça de apontar o caminho absoluto do require.txt caso você não esteja na mesma página do require.txt.