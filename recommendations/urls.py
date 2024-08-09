from django.contrib import admin
from django.urls import path
from django.shortcuts import render
from. import views

app_name = 'recommendations'

urlpatterns = [
    path("recomm_items/", views.recomm_items, name = 'recomm_items'),
    path("explanation_items/", views.explanation_items, name = 'explanation_items'),
    path("recomm_others/", views.recomm_others, name = 'recomm_others'),
]
