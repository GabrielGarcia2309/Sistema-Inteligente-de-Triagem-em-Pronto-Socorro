# Sistema Inteligente de Triagem em Pronto-Socorro

Este projeto aplica a Ciência de Dados e a Inteligência Artificial para otimizar o fluxo de atendimento em prontos-socorros, mitigando o risco de deterioração clínica em filas de espera. O sistema atua desde a leitura dos sinais vitais até a tomada de decisão matemática de priorização.

*Autores do Projeto:* João Gabriel Aquino Ferreira, Gabriel Garcia Colares e Rômulo Emanuel Marinho Barbosa

---

## O Problema 
Em hospitais lotados, a triagem puramente baseada na ordem de chegada (FIFO) ignora a biologia humana e pode custar vidas. O tempo de espera atua como um agravante exponencial silencioso. A proposta deste sistema é traduzir sinais clínicos em linguagem matemática, calculando dinamicamente quem deve ser atendido primeiro para minimizar o dano colateral de toda a fila. O objetivo central é utilizar os dados como ferramenta de intervenção prática para gerar valor e proteção ao coletivo.

## Arquitetura do Sistema
O projeto foi estruturado de forma modular para garantir o desacoplamento entre a inferência estatística e o algoritmo de otimização de rotas.

### Módulo 1: Redes Bayesianas (Probabilidade Clínica)
Para suprir a ausência de dados reais restritos, foi gerada uma base sintética de 1.000 pacientes, fundamentada nas diretrizes e pesos do Protocolo de Manchester.
* *Variáveis Observáveis:* 'Febre', 'SaturacaoO2', 'PressaoArterial', 'FrequenciaCardiaca', 'NivelDor' e 'IdadeDoencaCronica'.
* *Variável Alvo:* 'Gravidade' ('Alta', 'Media', 'Baixa').
* *Ferramenta:* A biblioteca pgmpy foi utilizada para treinar as Tabelas de Probabilidade Condicional (CPTs) e realizar a inferência exata de risco (Variable Elimination).

### Módulo 2: Busca Heurística (Algoritmo A*)
Os pacientes na fila de espera tornam-se "nós" em uma árvore de busca do algoritmo A*.
* *Função de Risco (Exponencial):* O perigo de cada paciente é calculado por $P \times e^{\frac{t}{\tau}}$, onde $P$ é a probabilidade de gravidade alta, $t$ é o tempo de espera e $\tau = 30.0$ é a constante de tolerância.
* *Heurística (h(n)):* Soma dos riscos atuais de todos os pacientes na fila.
* *Custo da Ação (g(n)):* O custo de atender um paciente é o risco acumulado de todos os outros que continuam aguardando.

### Pipeline de Integração
Uma função receber_paciente foi desenvolvida para conectar os módulos. O sistema recebe o dicionário de sintomas da triagem, extrai o valor de P(Alta) do motor bayesiano e instancializa automaticamente o objeto do paciente na fila dinâmica de testes.

---

## Resultados e Experimentos

O sistema avaliou e comparou três abordagens de tomada de decisão. As visualizações finais foram geradas utilizando a biblioteca seaborn.

1.  *Cenário Pequeno (5 Pacientes):* * O algoritmo A* encontrou a rota ótima visitando apenas 7 nós na árvore de busca, alcançando um Custo Total de *6.90*.
    * Este valor superou matematicamente a Estratégia Gulosa (custo *7.92) e o FIFO (custo **10.55*).

2.  *Cenário Médio (25 Pacientes):* * *Limitação Computacional:* A escalabilidade revelou a barreira da explosão combinatória. Ordenar 25 pacientes exige o cálculo de $25!$ permutações. Como esperado na teoria de grafos, a busca exata (A* puro) demanda alto custo de memória para filas dessa magnitude, sendo um limite conhecido da abordagem.
    * *O Colapso do FIFO:* Restritos à comparação de algoritmos mais rápidos neste cenário de estresse, o Custo Total do FIFO explodiu para *14.646.26, enquanto a Estratégia Gulosa conteve os danos com um custo de **2.854.06*. A matemática prova que a ordem de chegada (FIFO) é cega à biologia: ao deixar pacientes graves no fundo da fila, a função exponencial multiplica o risco de mortalidade em larga escala.

---

## Como Executar

Para reproduzir as análises localmente (por exemplo, via Terminal no macOS rodando scripts Python) ou diretamente em nuvem via Google Colab:

1. Clone o repositório ou baixe o arquivo .ipynb.
2. Instale as bibliotecas necessárias executando a primeira célula do notebook ou no seu terminal:
   pip install pgmpy numpy pandas matplotlib seaborn networkx
3. Execute as células sequencialmente. O código finaliza com os testes de estresse de fila e a plotagem automatizada da comparação de estratégias.
