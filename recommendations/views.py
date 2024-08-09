from django.shortcuts import render

# Create your views here.
def recomm_items(request):
    return render(request, 'recomm_items.html')

def explanation_items(request):
    return render(request, 'explanation_items.html')

def recomm_others(request):
    return render(request, 'recomm_others.html')
