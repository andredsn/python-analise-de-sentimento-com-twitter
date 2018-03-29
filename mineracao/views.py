from django.shortcuts import render

from mineracao.modulos.analise_sentimento import getDataFrame, calcularPercentualTotal, calcularClasses, main


def index(request):
    main()
    df =getDataFrame()
    percentuais=calcularPercentualTotal(df)
    totais=calcularClasses(df)
    total=len(df)
    
    texto=df['texto'].tail(1).values
    sentimento=df['sentimento'].tail(1).values
    
    return render(request, 'mineracao/index.html', {"total": total, "percentualSeguros": percentuais[0], "percentualInseguros": percentuais[1], "percentualNeutros": percentuais[2], "totalSeguros": totais[0], "totalInseguros": totais[1], "totalNeutros": totais[2], "texto": texto[0], "sentimento": sentimento[0]})