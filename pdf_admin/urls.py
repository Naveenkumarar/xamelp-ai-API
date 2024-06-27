from django.urls import path

from.import views

urlpatterns = [
    path('pdfupload', views.pdf_upload, name='pdf_upload'),
    path('question', views.ask_question, name='ask_question'),
    path('getmcq', views.get_mcq, name='get_mcq'),
]

from django.conf import settings
from django.conf.urls.static import static

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
