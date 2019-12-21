from django.shortcuts import render

#from mineracao.modulos.analise_sentimento import getDataFrame, calcularPercentualTotal, calcularClasses, main
from mineracao.modulos.analise_sentimento import calcularClasses,calcularPercentualTotal, getDataFrame, main

def index(request):
    main()
    df =getDataFrame()
    percentuais=calcularPercentualTotal(df)
    totais=calcularClasses(df)
    total=df.shape[0]
    
    texto=df['texto'].tail(1).values
    sentimento=df['sentimento'].tail(1).values
    
    return render(request, 'mineracao/index.html', {"total": total, "percentualCriticas": percentuais[0], "percentualOutros": percentuais[1], "texto": texto[0], "sentimento": sentimento[0]})