#linha comentada caso d� algum erro, descoment�-la para tentar corrigir, j� que foi o eclipse que criou esta linha
#from django.conf.urls import url
from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    url(r'^mineracao/', include('mineracao.urls')),
    url(r'^admin/', admin.site.urls),
]
