# Relatório de Projeto: Coverage Path Planning (5x5 e 10x10)

## 1. Descrição do Problema e Estratégia Escolhida

O objetivo deste projeto é treinar um agente via Reinforcement Learning (PPO) para realizar a cobertura completa (Coverage Path Planning) de ambientes de grid com obstáculos. O desafio principal decorre da **observabilidade parcial**: o agente possui apenas uma janela de visão restrita do seu redor e não tem acesso ao mapa global, devendo tomar decisões com base no que enxerga no momento e no que já explorou.

**Estratégia Escolhida:**
Após os testes iniciais demonstrarem que o agente sofria de severa "escassez de recompensas" e falta de informação espacial em grids maiores, adotou-se uma estratégia combinada de **Expansão Visual Segura** e **Reward Shaping Progressivo**:

1. **Expansão do Field of View (FOV):** A janela de observação local do agente (`self._neighbors`) foi ampliada do padrão 3x3 para um grid **5x5**. Isso mantém o conceito central do desafio (observabilidade parcial do mapa global), mas fornece ao MLP informação periférica suficiente para evitar "becos sem saída" e contornar obstáculos antes de colidir.
2. **Reward Shaping Progressivo:** 
   - A recompensa por descobrir células novas escala proporcionalmente ao que já foi coberto: `reward += 1.0 + (5.0 * coverage_ratio)`.
   - A penalidade por revisitar células foi consideravelmente reduzida de `-0.3` para `-0.05`. Isso é crucial em ambientes maiores onde o agente *precisa* cruzar o mapa por áreas já visitadas para explorar o outro lado.

---

## 2. Resultados Obtidos e Análise

Os modelos foram testados ao longo de 100 episódios cada para aferir a robustez de generalização.

| Tamanho do Grid | Configuração | Taxa de Cobertura Completa (%) | Cobertura Média (%) | Passos Médios | Recompensa Média |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **5x5** | Original (3x3, Reward Simples) | 97.0% | 99.86% | 44.2 | 18.5 |
| **5x5** | Atualizado (5x5, Reward Shaping) | 96.0% | 99.82% | **40.5** | 167.1 |
| **10x10** | Original (3x3, Reward Simples) | 77.0% | 99.44% | 321.3 | -14.0 |
| **10x10** | Atualizado (5x5, Reward Shaping) | 41.0% | 98.31% | 421.3 | 307.5 |

### Análise de Desempenho: O Ambiente 5x5
Para o grid de dimensões menores, a cobertura se aproxima fortemente de 100%. É notável que, embora a taxa de conclusão absoluta tenha oscilado de 97% para 96% no modelo atualizado, ele completou as rotas **quase 10% mais rápido** (40.5 passos vs 44.2). O aumento no Field of View para 5x5 provou ser eficaz para a otimização de caminhos locais.

### Análise de Desempenho e Limitações: O Ambiente 10x10
Embora a estratégia tenha sido capaz de alavancar enormemente a recompensa bruta do agente (subindo de -14 para 307.5 no teste), ocorreu o fenômeno de **Reward Hacking**. 
O agente explorou a combinação de bônus exploratório escalável com a baixa penalização por tempo e por revisitação. Ele aprendeu que pode colher uma vasta soma de recompensas limpando confortavelmente ~95% do mapa e que o esforço para encontrar os últimos 5% de blocos isolados (e ganhar o bônus de término de +100) não era matematicamente vantajoso. 

### Possíveis Melhorias para Implementações Futuras
Para contornar esse comportamento e viabilizar os testes oficiais em mapas de 20x20:
1. Reestruturar a função de custo para implementar *Potential-Based Distance Shaping*. Em vez de depender apenas do FOV restrito e de penalidades temporais brandas, o ambiente deve recompensar micromovimentos que diminuam a distância euclidiana até a célula inexplorada mais próxima, criando uma "gravidade" que orienta o agente em grandes matrizes.
2. Migrar da arquitetura estática MLP para um modelo LSTM (`RecurrentPPO`), fornecendo ao agente memória temporal do caminho que já construiu.
