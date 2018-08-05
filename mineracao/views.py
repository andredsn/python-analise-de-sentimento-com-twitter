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
    
    return render(request, 'mineracao/index.html', {"total": total, "percentualInseguros": percentuais[0], "percentualOutros": percentuais[1], "totalInseguros": totais[0], "totalOutros": totais[1], "texto": texto[0], "sentimento": sentimento[0]})

