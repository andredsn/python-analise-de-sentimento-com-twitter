from django.http import HttpResponse
from django.shortcuts import render
from mineracao.modulos.analise_sentimento import alterarPasta, lerCSV, calcularPercentualTotal

def index(request):
    pasta="c:/users/usuario/dados"
    alterarPasta(pasta)
    df=lerCSV("tweets_classificados.csv")
    percentuais=calcularPercentualTotal(df)
    total=df.shape
    return render(request, 'mineracao/index.html', {"total": total[0], "seguro": percentuais[0], "inseguro": percentuais[1], "neutro": percentuais[2]})