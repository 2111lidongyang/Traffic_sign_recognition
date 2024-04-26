from django.urls import path

from detect import views

urlpatterns = [
    path('home/', views.home),
    path('output/', views.output)
]
