"""
URL configuration for balance project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.shortcuts import render
from. import views

app_name = 'myapp'




urlpatterns = [
    path("select_kind/", views.select_kind, name = 'select_kind'),
    path("select_card/", views.select_card),
    path("select_insurance/", views.select_insurance),
    path("select_deposit/", views.select_deposit),
    path("select_loan/", views.select_loan),
    path("select_count/", views.select_count),
]
