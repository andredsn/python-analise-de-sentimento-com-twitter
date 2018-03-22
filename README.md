"# python-analise-de-sentimento-com-twitter" 

A base desse projeto � a utiliza��o da intelig�ncia artificial atrav�s da t�cnica de classifica��o, utilizando tr�s algoritmos: decision tree, SVM e Naive Bayes (utilizando a classe MultinomialNB) o para a an�lise de sentimento dos usu�rio sobre a seguran�a p�blica no Brasil.

A cada novo tweet classificado, a p�gina no browser exibe o percentual e total de cada classe. Os tr~es algoritmos classificam um tweet, mas o tweet � inserido em apenas uma classe. Por isso, � verificado o resultado de cada algoritmo e a a classifica��o de maior quantidade � a classifica��o v�lida. Por exemplo: se os algoritmos classificarem um texto do tweet respectivamente como: neutro, neutro, inseguro. Prevalecer� a classifica��o neutro.

Para usar a api do twitter, � necess�rio ter uma conta e criar um app na pr�pria rede social, o qual cont�m as credenciais que permitem acessar os dados.

Os pacotes necess�rios est�o no arquivo require.txt, nos qual  os principais pacots s�o: python 3.6, Django 1.11.7 e Anaconda3-4.4.0-Windows-x86_64

Django � um framework web python.

Aten��o: o anaconda j� tr�s o pr�prio python instalado. Portanto, deve-se instalar apenas o anaconda e depois instalar o require.txt.
Pegue o anaconda 3-4-4.0 no link:
https://repo.continuum.io/archive/Anaconda3-4.4.0-Windows-x86_64.exe

Para instalar os pacotes do arquivo texto referido, execute no prompt ou terminal: pip install -r require.txt
N�o esque�a de apontar o caminho absoluto do require.txt.
unicodedata 