# Pipeline de Machine Learning - Olist

## Arquivos de dados

O projeto utiliza:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`

Os CSVs podem ficar em uma pasta `data/` ou na mesma pasta do arquivo `olist_ml_pipeline.py`.

## Modelos

### Classificação
- Random Forest Classifier

### Regressão
- Regressão Linear Múltipla
- Regressão Polinomial de grau 2

## Fluxo

1. Carregamento dos CSVs.
2. Conversão das datas.
3. Seleção de pedidos entregues.
4. Agregação dos itens por `order_id`.
5. Junção dos datasets.
6. Criação das features.
7. Criação dos alvos `atrasado` e `days_atraso`.
8. EDA e identificação de possíveis outliers.
9. Divisão 70/30 em treino e teste.
10. Otimização do Random Forest com `RandomizedSearchCV`.
11. Treinamento final do Random Forest com todo o treino.
12. Geração de probabilidades out-of-fold para evitar vazamento na segunda etapa.
13. Escolha do limiar de classificação usando o treino.
14. Treinamento das regressões somente nos pedidos realmente atrasados.
15. Avaliação e exportação das métricas e gráficos.

## Execução

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_olist_ml.txt
python olist_ml_pipeline.py
```

No Linux Mint, o comando de ativação acima é adequado para Bash.

## Resultados gerados

A execução cria:

- `outputs/figures/` para gráficos;
- `outputs/tables/` para tabelas CSV;
- `outputs/resumo_resultados.txt` para um resumo do experimento.

## Observação sobre otimização

Para não tornar o treinamento pesado demais em computadores modestos, a busca de hiperparâmetros do Random Forest utiliza uma amostra estratificada do conjunto de treinamento. Depois de encontrar os melhores parâmetros, o modelo final é treinado usando todas as linhas do conjunto de treinamento.

Isso não é uma redução do conjunto usado na avaliação final: o conjunto de teste continua separado e intocado até a avaliação.

## Observação sobre outliers

Os valores extremos do dataset não são apagados automaticamente. O código exporta possíveis anomalias para `outputs/tables/possiveis_anomalias_para_inspecao.csv`, permitindo que a decisão de tratamento seja documentada no relatório em vez de ser escondida no pré-processamento.
