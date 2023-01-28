from django.shortcuts import render, redirect
from .models import Chat
from NeuralModel import reply


def index(request):
    object = Chat.objects.all()

    return render(request, 'inbox.html', {'object': object})


def send(request):
    message = request.POST['message']
    response =reply.get_response(message)

    chat = Chat()
    chat.text = message
    chat.person = 1
    chat.save()

    chat = Chat()
    chat.text = response
    chat.person = 2
    chat.save()

    return redirect('index')
