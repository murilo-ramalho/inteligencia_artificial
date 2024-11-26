"""
    Melhoria do chatbot apresentado em aula

    Na aula 21 apresentada no dia 14/11/2024 foi apresentado um chatbot básico 
    desenvolvido em linguagem python. A tarefa consiste em fazer melhorias nesse 
    chatbot básico. No dia a dia de um desenvolvedor é comum ter que melhorar códigos 
    antigos desenvolvidos por outras pessoas, e esse trabalho une essa tarefa ao tema da disciplina.
    O código para esse chatbot básico pode ser encontrado nos materiais de aula.
    Pode-se utilizar outros pacotes python no projeto para fazer as melhorias.

    Principais problemas a serem resolvidos:
    
    1. Melhoria na lematização do chatbot: Atualmente o chatbot tem problemas em distinguir 
    palavras no masculino e no feminino. Por exemplo ele não reconhece a palavras "obrigado" 
    mas reconhece "obrigada". 
    
    2. Fim dos erros de codificação de caracteres na resposta: Palavras acentuadas são exibidas 
    com caracteres incomuns atualmente. Aparentemente o erro vem do não uso do UTF-8 em 
    todas as partes do projeto. Investiguem se é realmente isso e corrijam.
    3. A interface gráfica GUI não envia o texto ao pressionar Enter na caixa de texto.

    Melhoria do dicionário de intents:
    1. Escolha um tema para que seu chatbot converse. Pode-se simular um negócio (empresa) 
    fictício ou algum outro assunto de seu interesse. 
    2. Procure não repetir o mesmo tema utilizado por outro grupo para evitar problemas de 
    trabalhos iguais já que o código base é o mesmo.

    Outras melhorias necessárias:
    1. Adição de um botão para limpar a conversa na interface gráfica.

    Grupo  
    O trabalho será desenvolvido em grupos de 2 até 4 alunos. Respeitem os tamanhos dos grupos.  

    Entrega do trabalho  
    Cada integrante do grupo deverá entregar o link do GitHub contendo todo o código do projeto.
    O nome dos integrantes deve constar no readme do projeto no GitHub.
    O readme também deverá ter uma explicação sobre o tema que o chatbot conversa.
    O readme deverá conter um tutorial mencionando a instalação dos pacotes e 
    dependências necessárias do projeto.

    Avaliação
    Será levado em consideração o quanto o chatbot básico foi melhorado em 
    relação ao original e aos outros projetos da sala.
    A qualidade da conversação com o chatbot será outro item avaliado.

    FAQ  
    Trabalhos iguais em grupos diferentes: Zero para ambos os grupos. 
    Cuidado com o que vocês compartilham entre si.  
    Usei uma IA para fazer os códigos pra mim, por isso ficou igual ao do outro grupo: 
    Zero para ambos os grupos. Usem as IA (GPT e afins) com responsabilidade. 
    Não sai copiando e colando qualquer coisa que eles respondem... Usem a cabeça...  

""" 



import json
from tkinter import *
from extract import class_prediction, get_response
from tensorflow.keras.models import load_model

model = load_model('model.keras')
intents = json.loads(open('intents.json', encoding='utf-8').read())

base = Tk()
base.title("Chatbot - Lanchonete")
base.geometry("400x500")
base.config(bg="#f2c12e")
base.resizable(width=False, height=False)

def chatbot_response(msg):
    ints = class_prediction(msg, model)
    res = get_response(ints, intents)
    return res

def send(event=None):
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        Chat.config(state=NORMAL)
        Chat.insert(END, f"Você: {msg}\n\n")
        response = chatbot_response(msg)
        Chat.insert(END, f"LanxeBot: {response}\n\n")
        Chat.config(state=DISABLED)
        Chat.yview(END)

def clear_chat():
    Chat.config(state=NORMAL)
    Chat.delete("1.0", END)
    Chat.config(state=DISABLED)

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
