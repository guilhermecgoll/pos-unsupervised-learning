# pos-unsupervised-learning
Contém um projeto de exemplo do Eigenfaces (predição de faces) em Python 3

# Executando o projeto

Após baixar o repositório, é necessário criar um ambiente virtual para instalação das dependências.

## Criação do ambiente virtual

- No diretório ``pos-unsupervised-learning`` execute o seguinte comando pelo shell:

```python -m venv .venv```

## Ativação do ambiente virtual

- Ainda no diretório ``pos-unsupervised-learning`` execute o seguinte comando pelo shell:

```./venv/Scripts/Activate.ps1```

PS: (pt-BR) No Microsoft Windows pode ser necessário habilitar a execução do script Activate.ps1, configurando a política de execução para o usuário. Você pode realizar isto executando o seguinte comando no PowerShell:

PS: (en-US) On Microsoft Windows, it may be required to enable the Activate.ps1 script by setting the execution policy for the user. You can do this by issuing the following PowerShell command:

```PS C:> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser```

See more: https://docs.python.org/3/library/venv.html

## Instalação das dependências e bibliotecas necessárias

- Ainda no diretório ``pos-unsupervised-learning`` execute o seguinte comando no shell:

```pip install --no-cache-dir -r pca-eigenfaces\requirements.txt```

## Execução do projeto

- Execute o comando abaixo no shell:

```python pca-eigenfaces/main.py```

## Exemplo de saída gerada pelo programa:
```
(.venv) PS G:\Pós\11. Aprendizado não supervisionado\pos-unsupervised-learning> & "g:/Pós/11. Aprendizado não supervisionado/pos-unsupervised-learning/.venv/Scripts/python.exe" "g:/Pós/11. Aprendizado não supervisionado/pos-unsupervised-learning/pca-eigenfaces/main.py"
10 componentes principais, acurácia: 95.9349593495935%
11 componentes principais, acurácia: 96.7479674796748%
12 componentes principais, acurácia: 97.5609756097561%
13 componentes principais, acurácia: 97.5609756097561%
14 componentes principais, acurácia: 96.7479674796748%
15 componentes principais, acurácia: 96.7479674796748%
16 componentes principais, acurácia: 96.7479674796748%
17 componentes principais, acurácia: 96.7479674796748%
18 componentes principais, acurácia: 96.7479674796748%
19 componentes principais, acurácia: 96.7479674796748%
20 componentes principais, acurácia: 96.7479674796748%
```


# Referências

https://docs.opencv.org/4.4.0/d3/df2/tutorial_py_basic_ops.html