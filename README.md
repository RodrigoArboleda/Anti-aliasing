# Colorização de imagens (de tons de cinza para RGB)
## Objetivo
O objetivo principal do projeto é transformar imagens em escalas de cinza para imagens coloridas do tipo RGB, através da utilização de uma rede neural convencional (CNN). Para isso é passado uma imagem em escala de cinza, e comparado o resultado final com sua versão RGB. Com isso os pesos devem ser ajustados até produzir uma rede capaz de colorir imagens.
## Images de entrada
Temos como entrada imagens em escala de cinza que devem ser convertidas em imagens RGB pela rede neural. Essas imagens possuem todas uma dimensão de 32x32 pixels.
### Fonte  das imagens
Essas imagens foram obtidas do repositório CIFAR 10 e Linnaeus 5. Vale dizer que essas imagens são divididas em várias classes, entretanto essas classificações não são relevantes para o projeto. Por isto, todas as imagens são colocadas juntas realizando o tratamento de nomes para não possuir imagens com nomes repetidos.
Fonte do conjunto de dados [CIFAR10](https://github.com/YoongiKim/CIFAR-10-images)
Fonte do conjunto de dados [Linnaeus](http://chaladze.com/l5/)
Como o conjunto de dados no site oficial do CIFAR 10 vem em um formato diferente do Linnaeus 5, utilizando a versão do github de Yoongi Kim, pois está no formato adequado.
### Tratando nomes das imagens
Para realizar este tratamento é acessado todos os arquivos baixados de forma recursiva. Como cada arquivo de imagem está dentro da pasta da sua respectiva classe, é gerado um novo nome a ele que é a concatenação do nome de seu diretório (pasta da classe) mais o seu nome original, então é salvo em um novo diretório Esse passo de mudança de nome é muito importante pois algumas imagens possuem o mesmo nome, dificultando o trabalho. Este procedimento pode ser encontrado no arquivo: organize_data.py no repositório do GitHub.
### Convertendo para escala de cinzas
As imagens do CIFAR 10 e do Linnaeus 5 são em RGB. As imagens em RGB são importantes para o treinamento do modelo, mas temos que fornecer como entrada imagens em escala de cinza. Para isso é preciso realizar uma conversão que ocorre de forma linear. A equação linear utilizada para tal é a seguinte:   
   
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/rgb_form.jpg)   
   
Sendo o valor do canal R (Red - Vermelho) multiplicado por 0.299. O do canal G (Green - Verde) é multiplicado por 0.587. E por fim do canal B (Blue - Azul) por 0.114. No fim temos apenas um valor que representa o valor deste pixel em escala de cinzas. Essa combinação linear para conversão em escala e cinzas foi retirada da especificação do trabalho 5 da disciplina de processamento de imagem do professor Moacir Ponti - ICMC/USP.   

![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/imgGray_RGB.jpg)
   
## Etapas para geração da imagem RGB
### Importação das bibliotecas e inicialização de variáveis
Para a execução do projeto, é necessário a instalação e importação das seguintes bibliotecas, utilizamos o PyTorch para a  implementação da rede neural e para realização dos testes.
```
import torch
import torch.nn as nn
import torchvision.datasets as dataset_loader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from skimage.io import imread_collection
 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
```
### Conversão da imagem
Para realizar a conversão, será utilizado uma rede neural convolucional. Para isso, primeiro iremos definir sua arquitetura, realizar seu treinamento e só então realizar o processo de conversão de imagens.
#### Arquitetura da Rede
A rede realiza primeiro um encoder para extrair os parâmetros da imagem. Isso ocorre utilizando uma camada de convolução seguida de uma ReLU ( Retificador) duas vezes. Após isso, uma camada de pooling é utilizada para reduzir o tamanho pela metade. Esse procedimento é realizado mais uma vez. Para finalizar a fazer de enconding é aplicada mais uma camada de convolução.   
Após isso é temos o decoder que irá gerar a saída da rede. Essa etapa irá consistir de uma camada linear, seguida de um ReLU e outra camada linear. Então teremos a saída da nossa rede com a imagem em RGB, abaixo tem uma representação gráfica da arquitetura da nossa rede.   
   
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/encoder.jpg)   
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/decoder.jpg)   
   
#### Organização das imagens para o treino da rede
Para realizar o treinamento da rede neural, nosso dataset foi dividido em 2, um para treino e outro para teste. Além disso, o grupo de imagens para treino foi dividido em batches. Essa divisão em batches é importante para o treino da rede neural para o cálculo do gradiente e ajuste dos parâmetros em cada época.   
Dado o grupo de imagens em escalas de cinza para treino completo, nós transformamos o conjunto em um conjunto de dados com média 0 e desvio padrão 1, pois este processo auxilia o treinamento da nossa rede neural.   
Realizamos um processo parecido para as respectivas imagens RGB, realizamos uma normalização nos valores dos pixels para sempre estarem no intervalo [0,1], não possibilitamos o valor negativo a nossa imagem RGB, para se ter uma completa noção de qual intensidade do canal temos, pois se colocarmos média 0 e desvio padrão 1, não conseguimos ter uma compreensão de como os valores são mapeados ao intervalo original [0,255].
#### Treinamento da rede
Para a realização do treinamento, utilizamos como função de loss ( função de erro) da rede a função MSE, que é definida por:   
   
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/loss_form.jpg)   
   
Onde N é a quantidade de imagens, fié a i-ésima imagem RGB, que é gerada utilizando a rede neural a partir da i-ésima imagem em escalas de cinza do conjunto, e yi  sendo a imagem RGB verdadeira que a i-ésima imagem em escalas de cinza.   
Em relação aos hiperparâmetros utilizados para o treinamento, utilizou um batch size de tamanho 100, learning rate de 0.0001, e treinou a rede neural durante 2000 épocas. Foi utilizado o otimizador Adam para o treinamento.
### Análise dos Resultados
Para analisar o desempenho da rede neural e verificar se está convergindo para o resultado esperado, foi utilizado o valor da função Loss tanto para o conjunto de treino, quanto para o conjunto de testes.
#### Gráfico de Loss do Treino e do Teste
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/tran_test_graphic%20.jpg)   
   
|                    |     ERRO FINAL     |
| ------------------ | ------------------ |
| CONJUNTO DE TREINO | 2.3465220669750124 |
| CONJUNTO DE TESTE  | 1.4805386755615473 |   
   
Podemos notar que a partir da época 250, começamos a ter um overfitting da rede neural para o conjunto de treino, pois a Loss no conjunto de teste passa a aumentar a partir deste ponto, começa a aumentar de valor.
#### Exemplos de Imagens Gerada
Abaixo, temos um conjunto de exemplo de imagens geradas do conjunto de testes, tendo as imagens em escalas de cinza, a imagem RGB gerada pela rede e a imagem RGB original
##### Imagens em Escalas de Cinza
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/gray_imgs.jpg)
##### Imagens RGB geradas pela Rede Neural
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/predict_imgs.jpg)
##### Imagem RGB Original
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/rgb_imgs.jpg)
### Testes ao Decorrer do Desenvolvimento
#### Testado outra Arquitetura
Inicialmente, se tentou utilizar uma versão alterada da rede U-Net para solucionar o problema, pois pelo que pesquisamos, é uma rede famosa por realizar segmentação, o que é um estilo de rede que tem de entrada uma imagem, e como saída outra imagem, porém este tipo de arquitetura não teve resultados nada agradáveis, como se pode ver abaixo:
##### Tabela com o Erro
|                    |     ERRO FINAL     |
| ------------------ | ------------------ |
| CONJUNTO DE TREINO | 28.65520852804184  |
| CONJUNTO DE TESTE  | 6.70705895498395   |   
   
Podemos ver abaixo alguns resultados nada satisfatórios desta arquitetura:
##### Imagens em Escalas de Cinza
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/gray_imgs.jpg)
##### Imagens RGB geradas pela Rede Neural
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/teste_outra_arquitetura.png)
##### Imagem RGB Original
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/rgb_imgs.jpg)
#### Variando Alguns Detalhes da Arquitetura
Como podemos ver na arquitetura principal, temos uma camada que passa por uma Fully Connected Layer de tamanho 512, esta camada tem um número de dimensões menores que a camada anterior e posterior dela, então tentamos mudar o tamanho desta camada para 2048, para ficar do mesmo tamanho da camada anterior, a seguir tem os resultados deste experimento:
##### Tabela com o Erro
|                    |     ERRO FINAL     |
| ------------------ | ------------------ |
| CONJUNTO DE TREINO | 0.8541442838031799 |
| CONJUNTO DE TESTE  | 1.3705658428370953 |   
   
Com esta variação, notamos que se tem um overfitting ainda maior do que a camada com tamanho 512, podemos ver abaixo alguns resultados desta alteração. Podemos notar que temos algumas cores mais fidedignas que a versão original, porém temos a impressão de que as imagens estão manchadas.
##### Imagens em Escalas de Cinza
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/gray_imgs.jpg)
##### Imagens RGB geradas pela Rede Neural
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/tabela%202048.png)
##### Imagem RGB Original
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/rgb_imgs.jpg)
## Referências
[U-Net - Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)   
[Implementing original U-Net from scratch using PyTorch](https://www.youtube.com/watch?v=u1loyDCoGbE)   
[CIFAR-10 images](https://github.com/YoongiKim/CIFAR-10-images)   
[Linnaeus 5 dataset](http://chaladze.com/l5/)   
[Stanford University School of Engineering Lecture | Deep Learning Software](https://www.youtube.com/watch?v=6SlgtELqOWc&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=8)   
## Alunos
- Luan Ícaro Pinto Arcanjo - 10799230
- Victor Graciano de Souza Correia - 10431201
- Rodrigo Cesar Arboleda - 10416722
