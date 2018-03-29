from django.shortcuts import render
from mineracao.modulos.analise_sentimento import getDataFrame, calcularPercentualTotal,\
    calcularClasses

def index(request):
    df =getDataFrame()
    percentuais=calcularPercentualTotal(df)
    totais=calcularClasses(df)
    total=len(df)
    return render(request, 'mineracao/index.html', {"total": total, "percentualSeguros": percentuais[0], "percentualInseguros": percentuais[1], "percentualNeutros": percentuais[2], "totalSeguros": totais[0], "totalInseguros": totais[1], "totalNeutros": totais[2]})