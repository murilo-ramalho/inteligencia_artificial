import json
from tkinter import *
from extract import class_prediction, get_response

from tensorflow.keras.models import load_model

# Carrega modelo e intenções
model = load_model('model.keras')
intents = json.loads(open('intents.json', encoding='utf-8').read())

# Configuração da interface gráfica
base = Tk()
base.title("Chatbot - Lanchonete")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)


def chatbot_response(msg):
    """
    Gera resposta do chatbot.
    """
    ints = class_prediction(msg, model)
    res = get_response(ints, intents)
    return res


def send(event=None):
    """
    Envia mensagem.
    """
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        Chat.config(state=NORMAL)
        Chat.insert(END, f"Você: {msg}\n\n")
        response = chatbot_response(msg)
        Chat.insert(END, f"Bot: {response}\n\n")
        Chat.config(state=DISABLED)
        Chat.yview(END)


def clear_chat():
    """
    Limpa a conversa.
    """
    Chat.config(state=NORMAL)
    Chat.delete("1.0", END)
    Chat.config(state=DISABLED)


# Configuração da interface gráfica
Chat = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
Chat.config(state=DISABLED)

scrollbar = Scrollbar(base, command=Chat.yview)
Chat['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana", 10, 'bold'), text="Enviar", width="12", height=2, command=send)
ClearButton = Button(base, font=("Verdana", 10, 'bold'), text="Limpar", width="12", height=2, command=clear_chat)

EntryBox = Text(base, bd=0, bg="white", width="29", height="2", font="Arial")
EntryBox.bind("<Return>", send)

scrollbar.place(x=376, y=6, height=386)
Chat.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=50, width=260)
SendButton.place(x=6, y=401, height=50)
ClearButton.place(x=6, y=451, height=30)

base.mainloop()
