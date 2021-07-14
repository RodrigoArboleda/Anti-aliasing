# Colorização de imagens (de tons de cinza para RGB)
## Objetivo
O objetivo principal do projeto é transformar imagens em escalas de cinza para imagens coloridas do tipo RGB, através da utilização de uma rede neural convencional (CNN). Para isso é passado uma imagem em escala de cinza, e comparado o resultado final com sua versão RGB. Com isso os pesos devem ser ajustados até produzir uma rede capaz de colorir imagens.
## Integrantes e suas Respectivas Funções Principais
### Luan Ícaro Pinto Arcanjo - NUSP: 10799230
Responsável por pesquisar e desenhar a arquitetura da Rede Neural, e pelo código de demonstração.

### Rodrigo Cesar Arboleda - NUSP: 10416722
Responsável por realizar o código de treinamento da rede neural, apresentar os resultados e pelo salvamento do estado da rede para sua utilização a posteriori.

### Victor Graciano de Souza Correia - NUSP: 10431201
Responsável por realizar o carregamento do conjunto de dados e pelo tratamento dos mesmos, para passá-los no formato adequado a Rede Neural. 

## Images de entrada
Temos como entrada imagens em escala de cinza que devem ser convertidas em imagens RGB pela rede neural. Essas imagens possuem todas uma dimensão de 32x32 pixels.
### Fonte  das imagens
Essas imagens foram obtidas do repositório CIFAR 10 e Linnaeus 5. Vale dizer que essas imagens são divididas em várias classes, entretanto essas classificações não são relevantes para o projeto. Por isto, todas as imagens são colocadas juntas realizando o tratamento de nomes para não possuir imagens com nomes repetidos
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
Para realizar o treinamento da rede neural, nosso dataset foi dividido em 2, um para treino e outro para validação. Além disso, o grupo de imagens para treino foi dividido em batches. Essa divisão em batches é importante para o treino da rede neural para o cálculo do gradiente e ajuste dos parâmetros em cada época.   
Dado o grupo de imagens em escalas de cinza para treino completo, nós transformamos o conjunto em um conjunto de dados com média 0 e desvio padrão 1, pois este processo auxilia o treinamento da nossa rede neural.   
Realizamos um processo parecido para as respectivas imagens RGB, realizamos uma normalização nos valores dos pixels para sempre estarem no intervalo [0,1], não possibilitamos o valor negativo a nossa imagem RGB, para se ter uma completa noção de qual intensidade do canal temos, pois se colocarmos média 0 e desvio padrão 1, não conseguimos ter uma compreensão de como os valores são mapeados ao intervalo original [0,255].
#### Treinamento da rede
Para a realização do treinamento, utilizamos como função de loss ( função de erro) da rede a função MSE, que é definida por:   
   
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/loss_form.jpg)   
   
Onde N é a quantidade de imagens, fié a i-ésima imagem RGB, que é gerada utilizando a rede neural a partir da i-ésima imagem em escalas de cinza do conjunto, e yi  sendo a imagem RGB verdadeira que a i-ésima imagem em escalas de cinza.   
Em relação aos hiperparâmetros utilizados para o treinamento, utilizou um batch size de tamanho 100, learning rate de 0.0001, e treinou a rede neural durante 2000 épocas. Foi utilizado o otimizador Adam para o treinamento.
### Análise dos Resultados
Para analisar o desempenho da rede neural e verificar se está convergindo para o resultado esperado, foi utilizado o valor da função Loss tanto para o conjunto de treino, quanto para o conjunto de validação.
#### Gráfico de Loss do Treino e do Teste
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/tran_test_graphic1.png)   
   
|                       |     ERRO FINAL     |
| --------------------- | ------------------ |
|  CONJUNTO DE TREINO   | 2.3465220669750124 |
| CONJUNTO DE VALIDAÇÃO | 1.4805386755615473 |   
   
Podemos notar que a partir da época 250, começamos a ter um overfitting da rede neural para o conjunto de treino, pois a Loss no conjunto de validação passa a aumentar a partir deste ponto, começa a aumentar de valor.
#### Exemplos de Imagens Gerada
Abaixo, temos um conjunto de exemplo de imagens geradas do conjunto de validação, tendo as imagens em escalas de cinza, a imagem RGB gerada pela rede e a imagem RGB original.
##### Imagens em Escalas de Cinza
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/exemplo1.png)
##### Imagens RGB geradas pela Rede Neural após 500 épocas
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/exemplo2.png)
##### Imagens RGB geradas pela Rede Neural após 2000 épocas.
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/exemplo3.png)
##### Imagem RGB Original
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/exemplo4.png)
### Testando o modelo em imagens Diferentes
Todos os testes realizados até o momento foram feitos testes somente baseados nas imagens do Dataset, o que faz o modelo tender a aprender apenas os padrões de cores das imagens relacionadas às imagens proveniente do conjunto de dados, o que nos faz questionar como o modelo performa em imagens retiradas de outras fontes, por isso iremos iremos mostrar a seguir exemplos de imagens que não relacionadas ao dataset.
#### Imagens não Relacionadas ao conjunto de imagens
##### Wolverine
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/volv_exp.png)
##### José Carioca
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/ze_exp.png)
##### Buzz Lightyear
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/buzz_exp.png)
##### Monkey D. Luffy
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/luffy_exp.png)
##### Análise
Podemos ver que o modelo tem um resultado interessante em imagens relacionadas ao dataset, o primeiro ponto que podemos notar, é que o modelo não distorce as imagens, porém aplica um blur a elas. 
Podemos ver que o modelo conseguiu inferir de forma satisfatória as cores do wolverine, aprendendo que sua roupa é majoritariamente amarela. Em relação Buzz Lightyear, o modelo parece aprender a paleta de cores da imagem com exceção da cor de pêssego do rosto do personagem, e acaba não organizando muito bem onde as cores estão, fazendo a imagem gerada ficar manchada de roxo.
Já em relação ao Zé Carioca, o modelo apenas detecta que na imagem há verde em algum lugar, enquanto com a imagem do pirata que estica, a imagem fica apenas bege.
### Testes ao Decorrer do Desenvolvimento
#### Testado outra Arquitetura
Inicialmente, se tentou utilizar uma versão alterada da rede U-Net para solucionar o problema, pois pelo que pesquisamos, é uma rede famosa por realizar segmentação, o que é um estilo de rede que tem de entrada uma imagem, e como saída outra imagem, porém este tipo de arquitetura não teve resultados nada agradáveis, como se pode ver abaixo:
##### Tabela com o Erro
|                       |     ERRO FINAL     |
| --------------------- | ------------------ |
|  CONJUNTO DE TREINO   | 28.65520852804184  |
| CONJUNTO DE VALIDAÇÃO | 6.70705895498395   |   
   
Podemos ver abaixo alguns resultados nada satisfatórios desta arquitetura:
##### Imagens em Escalas de Cinza
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/gray_imgs.jpg)
##### Imagens RGB geradas pela Rede Neural
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/teste_outra_arquitetura.png)
##### Imagem RGB Original
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/rgb_imgs.jpg)
#### Variando Alguns Detalhes da Arquitetura
A arquitetura definida inicialmente, no relatório parcial, fora uma arquitetura, onde todas camadas tinham metade das dimensões da arquitetura final, por exemplo, a camada Fully Connected tinha dimensão 512, e as camadas convolucionais de 32 filtros, tinham apenas 16.   
Essa arquitetura, no primeiro teste, teve um resultado bem interessante quando olhávamos as imagens criadas inicialmente, em um momento onde não tínhamos o código para salvar o estado da rede. Posteriormente, com a funcionalidade de salvar a rede, tentamos reproduzir este mesmo estado diversas vezes, porém sem sucesso, o que nos fez escolhermos por mudar a arquitetura, pois com a mudança, nós conseguimos resultados mais satisfatórios do que os testes falhos. Abaixo há a tabela de erros deste modelo inicial.
##### Tabela com o Erro
|                       |     ERRO FINAL     |
| --------------------- | ------------------ |
|  CONJUNTO DE TREINO   | 0.8541442838031799 |
| CONJUNTO DE VALIDAÇÃO | 1.3705658428370953 |   
   
Com este modelo inicial, podemos notar que temos um resultado em números, parelho a solução final. Abaixo podemos ver alguns resultados desta arquitetura. Podemos notar que o teste que não conseguimos reproduzir têm um resultado bem satisfatório, as cores , porém os testes após o código final, acaba sendo pior do que a solução final.
##### Imagens em Escalas de Cinza
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/exemplo5.png)
##### Imagens RGB geradas pela Rede Neural após o Código FInal
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/exemplo6.png)
##### Imagens RGB geradas pelo teste não Reprodutível.
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/exemplo7.png)
##### Imagem RGB Original
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/exemplo8.png)
#### Testado o otimizador Reduce LR On Plateau
Como mencionado anteriormente, a rede neural está sofrendo overfitting pelo conjunto de treinamento, para se testar se este problema é por causa de um learning rate baixo, resultando em cair em mínimos locais facilmente, ou se a arquitetura desenhada está mesmo no limite de sua capacidade, utilizamos o otimizador Reduce LR On Plateau sugerido pelo professor no relatório parcial da disciplina.   
O otimizador Reduce LR On Plateau altera o learning rate dinamicamente ao decorrer do treinamento, utilizando um learning rate alto inicialmente, e caso, durante um período a rede estiver  em plateau, o otimizador abaixa o valor do learning rate para realizar movimento no gradientes mais finos. Para este learning rate inicial, utilizamos 0.001 como valor deste hiperparâmetro, o que é 10 vezes maior que o learning rate utilizado nos outros testes.
##### Gráficos da Loss
![alt text](https://github.com/RodrigoArboleda/Grayscale-to-RGB/blob/main/img/tran_test_graphic2.png)
##### Tabela com o Erro
|                       |     ERRO FINAL     |
| --------------------- | ------------------ |
|  CONJUNTO DE TREINO   | 2.8273078962229192 |
| CONJUNTO DE VALIDAÇÃO | 1.3493332248181105 |   
   
Como podemos ver, com este otimizador, acabamos tendo um comportamento bem estranho na função de custo do conjunto de validação, ela acaba não tendo um comportamento nada esperado, e também acabamos caindo em um mínimo local rapidamente, após cerca de 300 épocas, e este mínimo local acaba tendo um erro ainda maior no conjunto de treino e de validação dos testes anteriores, o que nos fez escolher por não utilizarmos este otimizador no modelo final.
## Notebook
O código principal e sua execução pode ser conferida no notebook denominado pdi_trabalho.ipynb  na pasta source do GitHub

O código de Demonstração e sua execução pode ser coferido no notebook denominado Demonstracao.ipynb na pasta source do GitHub

## Apresentação

Para se acessar o vídeo com a apresentação do projeto, [clique aqui.](https://drive.google.com/file/d/1ONwIo9DE7bmXMlP0oLNzAsV7bkn9YAWY/view?usp=sharing)

## Referências
[U-Net - Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)   
[Implementing original U-Net from scratch using PyTorch](https://www.youtube.com/watch?v=u1loyDCoGbE)   
[CIFAR-10 images](https://github.com/YoongiKim/CIFAR-10-images)   
[Linnaeus 5 dataset](http://chaladze.com/l5/)   
[Stanford University School of Engineering Lecture | Deep Learning Software](https://www.youtube.com/watch?v=6SlgtELqOWc&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=8)   
