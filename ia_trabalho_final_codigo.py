# -*- coding: utf-8 -*-
"""IA_Trabalho_Final_codigo

Gabriel Garcia Colares 581964
João Gabriel Aquino Ferreira 582424
Rômulo Emanuel Marinho Barbosa 579586

"""

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import math
import heapq
import random

# ==========================================
# Módulo 1: Redes Bayesianas
# ==========================================

print("[INFO] Inicializando a Rede Bayesiana...")

# Definindo a estrutura da rede (causas -> efeito)
modelo_triagem = DiscreteBayesianNetwork([
    ('Febre', 'Gravidade'),
    ('SaturacaoO2', 'Gravidade'),
    ('PressaoArterial', 'Gravidade'),
    ('FrequenciaCardiaca', 'Gravidade'),
    ('NivelDor', 'Gravidade'),
    ('IdadeDoencaCronica', 'Gravidade')
])

print("Nós da rede:", modelo_triagem.nodes())
print("Arestas estruturadas:", modelo_triagem.edges())

def visualizar_rede(modelo, titulo="Rede Bayesiana"):
    plt.figure(figsize=(12, 6))
    pos = nx.circular_layout(modelo)
    nx.draw_networkx_nodes(modelo, pos, node_color='lightblue', node_size=2200)
    nx.draw_networkx_edges(modelo, pos, edge_color='gray', arrows=True,
                           arrowstyle='-|>', arrowsize=20, node_size=2200)
    nx.draw_networkx_labels(modelo, pos, font_size=12, font_weight='bold')
    plt.title(titulo, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('01_rede_bayesiana.png', dpi=300, bbox_inches='tight')
    print("[INFO] Gráfico salvo como: 01_rede_bayesiana.png")
    print("[AVISO] Feche a janela do gráfico da Rede Bayesiana para continuar a execução...")
    plt.show()

visualizar_rede(modelo_triagem, "Modelo Triagem")

# Fixando a semente aleatória para o resultado ser sempre o mesmo ao testar
np.random.seed(42)
num_pacientes = 1000

# Gerando os dados vitais (sintomas) de forma aleatória com proporções realistas
saturacao = np.random.choice(['Normal_Alta', 'Reduzida', 'Critica'], size=num_pacientes, p=[0.80, 0.15, 0.05])
dor = np.random.choice(['Leve_Moderada', 'Intensa'], size=num_pacientes, p=[0.7, 0.3])
febre = np.random.choice(['Sem_Febre_Leve', 'Alta'], size=num_pacientes, p=[0.6, 0.4])
freq_cardiaca = np.random.choice(['Normal', 'Alterada'], size=num_pacientes, p=[0.8, 0.2])
pressao = np.random.choice(['Normal', 'Anormal_Choque'], size=num_pacientes, p=[0.85, 0.15])
idade_doenca = np.random.choice(['Saudavel_Jovem', 'Idoso_Comorbidade'], size=num_pacientes, p=[0.6, 0.4])

# Criando o DataFrame
df_pacientes = pd.DataFrame({
    'SaturacaoO2': saturacao,
    'NivelDor': dor,
    'Febre': febre,
    'FrequenciaCardiaca': freq_cardiaca,
    'PressaoArterial': pressao,
    'IdadeDoencaCronica': idade_doenca
})

def calcular_gravidade(row):
    pontos = 0
    if row['SaturacaoO2'] == 'Critica': pontos += 3
    elif row['SaturacaoO2'] == 'Reduzida': pontos += 1

    if row['PressaoArterial'] == 'Anormal_Choque': pontos += 2
    if row['NivelDor'] == 'Intensa': pontos += 1
    if row['Febre'] == 'Alta': pontos += 1
    if row['FrequenciaCardiaca'] == 'Alterada': pontos += 1
    if row['IdadeDoencaCronica'] == 'Idoso_Comorbidade': pontos += 1

    if pontos >= 4:
        return 'Alta'
    elif pontos >= 2:
        return 'Media'
    else:
        return 'Baixa'

# Aplicando a regra para criar a coluna alvo
df_pacientes['Gravidade'] = df_pacientes.apply(calcular_gravidade, axis=1)

# Substituído display() por print() padrão do terminal
print("\n--- Primeiros 5 Pacientes da Base Sintética ---")
print(df_pacientes.head())
print("\n--- Distribuição de Gravidade no PS Sintético ---")
print(df_pacientes['Gravidade'].value_counts(normalize=True).round(3))

# Fazendo a rede "aprender" as Tabelas de Probabilidade (CPTs) com os nossos dados
modelo_triagem.fit(df_pacientes)

# Validando se as probabilidades somam 1.0 corretamente
print("\nModelo válido?", modelo_triagem.check_model())

# Criando o motor de inferência
inferencia_triagem = VariableElimination(modelo_triagem)

print("\nEstimativa de Gravidade para o Paciente Teste:")
resultado = inferencia_triagem.query(
    variables=['Gravidade'],
    evidence={
        'Febre': 'Alta',
        'SaturacaoO2': 'Reduzida',
        'PressaoArterial': 'Anormal_Choque',
        'NivelDor': 'Intensa',
        'IdadeDoencaCronica': 'Idoso_Comorbidade'
    }
)
print(resultado)


# ==========================================
# Módulo 2: O algoritmo A*
# ==========================================

print("\n[INFO] Inicializando os testes de Busca e Priorização...")

class Paciente:
    def __init__(self, id_paciente, p_alta, tempo_esperando):
        self.id_paciente = id_paciente
        self.p_alta = p_alta
        self.tempo_esperando = tempo_esperando
        self.tau = 30.0 

    def calcular_risco_atual(self):
        f_tempo = math.exp(self.tempo_esperando / self.tau)
        return self.p_alta * f_tempo

    def __repr__(self):
        return f"Paciente(id={self.id_paciente}, p_alta={self.p_alta:.2f}, tempo={self.tempo_esperando}min, risco={self.calcular_risco_atual():.2f})"

fila_inicial = [
    Paciente(id_paciente="Ana", p_alta=0.85, tempo_esperando=10),
    Paciente(id_paciente="Bruno", p_alta=0.60, tempo_esperando=30),
    Paciente(id_paciente="Carla", p_alta=0.20, tempo_esperando=5),
    Paciente(id_paciente="Diego", p_alta=0.45, tempo_esperando=20),
    Paciente(id_paciente="Elena", p_alta=0.10, tempo_esperando=45)
]

print("\nFila inicial com Risco Exponencial:")
for p in fila_inicial:
    print(p)

class EstadoFila:
    def __init__(self, pacientes_restantes, ordem_atendimento, custo_g, tempo_atual):
        self.pacientes_restantes = pacientes_restantes 
        self.ordem_atendimento = ordem_atendimento     
        self.custo_g = custo_g                         
        self.tempo_atual = tempo_atual                 

    def heuristica_h(self):
        soma_riscos = sum(p.calcular_risco_atual() for p in self.pacientes_restantes)
        return soma_riscos

    def custo_f(self):
        return self.custo_g + self.heuristica_h()

    def gerar_sucessores(self, tempo_por_atendimento=10):
        sucessores = []
        for i, paciente_escolhido in enumerate(self.pacientes_restantes):
            risco_da_espera = sum(p.calcular_risco_atual() for p in self.pacientes_restantes if p.id_paciente != paciente_escolhido.id_paciente)
            novo_custo_g = self.custo_g + risco_da_espera

            nova_fila = []
            for p in self.pacientes_restantes:
                if p.id_paciente != paciente_escolhido.id_paciente:
                    novo_paciente = Paciente(p.id_paciente, p.p_alta, p.tempo_esperando + tempo_por_atendimento)
                    nova_fila.append(novo_paciente)

            nova_ordem = self.ordem_atendimento + [paciente_escolhido.id_paciente]
            novo_estado = EstadoFila(nova_fila, nova_ordem, novo_custo_g, self.tempo_atual + tempo_por_atendimento)
            sucessores.append(novo_estado)

        return sucessores

    def __lt__(self, outro):
        return self.custo_f() < outro.custo_f()

def simular_estrategia_basica(fila_inicial, estrategia="FIFO"):
    import copy
    pacientes_restantes = copy.deepcopy(fila_inicial)
    ordem_atendimento = []
    custo_total = 0.0
    tempo_por_atendimento = 10

    while pacientes_restantes:
        if estrategia == "FIFO":
            pacientes_restantes = sorted(pacientes_restantes, key=lambda p: p.tempo_esperando, reverse=True)
        elif estrategia == "Gulosa":
            pacientes_restantes = sorted(pacientes_restantes, key=lambda p: p.p_alta, reverse=True)

        paciente_escolhido = pacientes_restantes.pop(0)
        ordem_atendimento.append(paciente_escolhido.id_paciente)

        risco_da_espera = sum(p.calcular_risco_atual() for p in pacientes_restantes)
        custo_total += risco_da_espera

        for p in pacientes_restantes:
            p.tempo_esperando += tempo_por_atendimento

    return ordem_atendimento, custo_total

ordem_fifo, custo_fifo = simular_estrategia_basica(fila_inicial, "FIFO")
ordem_gulosa, custo_gulosa = simular_estrategia_basica(fila_inicial, "Gulosa")

print(f"\nEstratégia FIFO: Ordem {ordem_fifo} | Custo Total: {custo_fifo:.2f}")
print(f"Estratégia Gulosa: Ordem {ordem_gulosa} | Custo Total: {custo_gulosa:.2f}")

def busca_a_estrela(fila_inicial):
    import copy
    estado_inicial = EstadoFila(
        pacientes_restantes=copy.deepcopy(fila_inicial),
        ordem_atendimento=[],
        custo_g=0.0,
        tempo_atual=0
    )

    fronteira = []
    heapq.heappush(fronteira, estado_inicial)
    nos_explorados = 0

    while fronteira:
        estado_atual = heapq.heappop(fronteira)
        nos_explorados += 1

        if not estado_atual.pacientes_restantes:
            return estado_atual.ordem_atendimento, estado_atual.custo_g, nos_explorados

        sucessores = estado_atual.gerar_sucessores(tempo_por_atendimento=10)
        for suc in sucessores:
            heapq.heappush(fronteira, suc)

    return None, 0, nos_explorados

ordem_a_estrela, custo_a_estrela, nos_visitados = busca_a_estrela(fila_inicial)

print(f"Estratégia A*: Ordem {ordem_a_estrela} | Custo Total: {custo_a_estrela:.2f}")
print(f"Nós explorados na árvore de busca: {nos_visitados}")

# Cenário Médio
random.seed(42)
num_pacientes_medio = 25
fila_media = []

for i in range(num_pacientes_medio):
    id_pac = f"Pac_{i+1}"
    p_alta_sorteada = random.uniform(0.05, 0.95)
    tempo_sorteado = 0
    novo_paciente = Paciente(id_paciente=id_pac, p_alta=p_alta_sorteada, tempo_esperando=tempo_sorteado)
    fila_media.append(novo_paciente)

print(f"\nCenário Médio gerado com {len(fila_media)} pacientes.")
for p in fila_media[:3]:
    print(p)

ordem_fifo_media, custo_fifo_medio = simular_estrategia_basica(fila_media, "FIFO")
ordem_gulosa_media, custo_gulosa_medio = simular_estrategia_basica(fila_media, "Gulosa")

print(f"Cenário Médio - Custo Total FIFO: {custo_fifo_medio:.2f}")
print(f"Cenário Médio - Custo Total Gulosa: {custo_gulosa_medio:.2f}")

def receber_paciente(id_paciente, sintomas_coletados, tempo_espera_atual, motor_inferencia, fila_atual):
    print(f"\n--- Novo paciente chegou na triagem: {id_paciente} ---")
    resultado_inferencia = motor_inferencia.query(
        variables=['Gravidade'],
        evidence=sintomas_coletados
    )
    idx_alta = resultado_inferencia.state_names['Gravidade'].index('Alta')
    p_alta_calculada = resultado_inferencia.values[idx_alta]

    print(f"A Rede Bayesiana calculou P(Alta) = {p_alta_calculada:.4f}")
    novo_paciente = Paciente(id_paciente=id_paciente, p_alta=p_alta_calculada, tempo_esperando=tempo_espera_atual)
    fila_atual.append(novo_paciente)
    print(f"Paciente {id_paciente} inserido na fila de espera com sucesso!\n")
    return fila_atual

fila_dinamica = []
sintomas_paciente_A = {
    'SaturacaoO2': 'Normal_Alta',
    'PressaoArterial': 'Normal',
    'NivelDor': 'Leve_Moderada',
    'Febre': 'Sem_Febre_Leve',
    'FrequenciaCardiaca': 'Normal',
    'IdadeDoencaCronica': 'Saudavel_Jovem'
}

fila_dinamica = receber_paciente(
    id_paciente="Maria",
    sintomas_coletados=sintomas_paciente_A,
    tempo_espera_atual=0,
    motor_inferencia=inferencia_triagem,
    fila_atual=fila_dinamica
)


# ==========================================
# Módulo Extra: Visualizações
# ==========================================

print("[INFO] Renderizando os gráficos estatísticos comparativos...")
sns.set_theme(style="white", context="notebook")

# GRÁFICO 1: CENÁRIO PEQUENO
dados_peq = pd.DataFrame({
    'Estratégia': ['Gulosa', 'FIFO', 'A*'],
    'Custo': [custo_gulosa, custo_fifo, custo_a_estrela]
})

plt.figure(figsize=(8, 5))
ax1 = sns.barplot(data=dados_peq, x='Estratégia', y='Custo', hue='Estratégia', palette='mako', legend=False)

for container in ax1.containers:
    ax1.bar_label(container, fmt='%.2f', padding=3, fontweight='bold')

plt.title('Custo Total de Risco - Cenário Pequeno (5 Pacientes)', fontsize=14, pad=15)
plt.ylabel('Risco Acumulado', fontsize=12)
plt.xlabel('')
sns.despine()
plt.savefig('02_cenario_pequeno.png', dpi=300, bbox_inches='tight')
print("[INFO] Gráfico salvo como: 02_cenario_pequeno.png")
print("[AVISO] Feche a janela do primeiro gráfico de desempenho para exibir o próximo...")
plt.show()

# GRÁFICO 2: CENÁRIO MÉDIO
dados_med = pd.DataFrame({
    'Estratégia': ['Gulosa', 'FIFO'],
    'Custo': [custo_gulosa_medio, custo_fifo_medio]
})

plt.figure(figsize=(8, 5))
ax2 = sns.barplot(data=dados_med, x='Estratégia', y='Custo', hue='Estratégia', palette='mako', legend=False)

for container in ax2.containers:
    ax2.bar_label(container, fmt='%.2f', padding=3, fontweight='bold')

plt.title('Custo Total de Risco - Cenário Médio (25 Pacientes)', fontsize=14, pad=15)
plt.ylabel('Risco Acumulado', fontsize=12)
plt.xlabel('')
sns.despine()
plt.savefig('03_cenario_medio.png', dpi=300, bbox_inches='tight')
print("[INFO] Gráfico salvo como: 03_cenario_medio.png")
plt.show()

print("\n[INFO] Execução finalizada com sucesso!")
